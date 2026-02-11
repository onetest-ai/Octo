"""CLI subcommands for skills marketplace — search, install, manage."""
from __future__ import annotations

import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import click

from octo.config import SKILLS_CACHE_DIR, SKILLS_CACHE_TTL, SKILLS_DIR, SKILLS_REGISTRY_URL


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

_CACHE_FILE = SKILLS_CACHE_DIR / "registry.json"


def _fetch_registry(force: bool = False) -> list[dict]:
    """Fetch registry.json from GitHub, with local cache (TTL-based)."""
    if not force and _CACHE_FILE.exists():
        age = time.time() - _CACHE_FILE.stat().st_mtime
        if age < SKILLS_CACHE_TTL:
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))

    try:
        req = urllib.request.Request(SKILLS_REGISTRY_URL)
        # Use GITHUB_TOKEN for higher rate limits if available
        import os
        token = os.getenv("GITHUB_TOKEN", "")
        if token:
            req.add_header("Authorization", f"token {token}")

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return data
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        # Fall back to cached version if available
        if _CACHE_FILE.exists():
            click.echo(f"Warning: could not fetch registry ({exc}), using cached version.")
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        click.echo(f"Error: could not fetch skills registry: {exc}")
        raise SystemExit(1) from exc


def _find_in_registry(registry: list[dict], name: str) -> dict | None:
    """Find a skill by name in the registry."""
    for entry in registry:
        if entry["name"] == name:
            return entry
    return None


def _download_skill_files(skill_name: str, files: list[str]) -> Path:
    """Download all files for a skill from GitHub raw URLs."""
    import os
    base_url = SKILLS_REGISTRY_URL.rsplit("/", 1)[0]  # strip registry.json
    skill_base = f"{base_url}/skills/{skill_name}"

    dest = SKILLS_DIR / skill_name
    dest.mkdir(parents=True, exist_ok=True)

    token = os.getenv("GITHUB_TOKEN", "")
    for rel_path in files:
        url = f"{skill_base}/{rel_path}"
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"token {token}")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()

            out_file = dest / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(content)
        except urllib.error.URLError as exc:
            click.echo(f"  Warning: failed to download {rel_path}: {exc}")

    return dest


def _installed_skill_meta(skill_dir: Path) -> dict | None:
    """Parse minimal metadata from an installed SKILL.md."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return None

    import yaml
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {"name": skill_dir.name, "version": "0.0.0", "description": ""}

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {"name": skill_dir.name, "version": "0.0.0", "description": ""}

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except Exception:
        meta = {}

    return {
        "name": meta.get("name", skill_dir.name),
        "version": meta.get("version", "0.0.0"),
        "description": meta.get("description", ""),
        "author": meta.get("author", ""),
        "tags": meta.get("tags", []),
    }


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------

@click.group(name="skills")
def skills() -> None:
    """Manage Octo skills — search, install, update, remove."""


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@skills.command()
@click.argument("query", default="")
@click.option("--tag", "-t", default="", help="Filter by tag")
def search(query: str, tag: str) -> None:
    """Search the skills marketplace."""
    registry = _fetch_registry()

    results = []
    for entry in registry:
        if tag and tag.lower() not in [t.lower() for t in entry.get("tags", [])]:
            continue
        if query:
            q = query.lower()
            name_match = q in entry["name"].lower()
            desc_match = q in entry.get("description", "").lower()
            tag_match = any(q in t.lower() for t in entry.get("tags", []))
            if not (name_match or desc_match or tag_match):
                continue
        results.append(entry)

    if not results:
        click.echo("No skills found matching your query.")
        return

    # Sort: exact name prefix first, then alphabetical
    q_lower = query.lower() if query else ""
    results.sort(key=lambda e: (not e["name"].lower().startswith(q_lower), e["name"]))

    # Table output
    click.echo(f"\n{'Name':<20} {'Version':<10} {'Tags':<30} Description")
    click.echo("-" * 90)
    for entry in results:
        tags_str = ", ".join(entry.get("tags", []))[:28]
        desc = entry.get("description", "")[:40]
        click.echo(f"{entry['name']:<20} {entry.get('version', '?'):<10} {tags_str:<30} {desc}")
    click.echo(f"\n{len(results)} skill(s) found.")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@skills.command()
@click.argument("name")
def info(name: str) -> None:
    """Show detailed information about a skill."""
    registry = _fetch_registry()
    entry = _find_in_registry(registry, name)

    if not entry:
        click.echo(f"Skill '{name}' not found in registry.")
        return

    click.echo(f"\n  Name:        {entry['name']}")
    click.echo(f"  Version:     {entry.get('version', '?')}")
    click.echo(f"  Author:      {entry.get('author', '?')}")
    click.echo(f"  Description: {entry.get('description', '')}")
    click.echo(f"  Tags:        {', '.join(entry.get('tags', []))}")

    # Dependencies
    deps = entry.get("dependencies", {})
    has_deps = any(deps.get(k) for k in ("python", "npm", "mcp", "system"))
    if has_deps:
        click.echo("\n  Dependencies:")
        for kind in ("python", "npm", "mcp", "system"):
            items = deps.get(kind, [])
            if items:
                if kind == "mcp":
                    names = [m.get("server", "?") for m in items]
                    click.echo(f"    {kind}: {', '.join(names)}")
                else:
                    click.echo(f"    {kind}: {', '.join(items)}")

    # Requirements
    reqs = entry.get("requires", [])
    if reqs:
        click.echo("\n  Requirements:")
        for r in reqs:
            if "command" in r:
                click.echo(f"    command: {r['command']} — {r.get('reason', '')}")
            if "env" in r:
                click.echo(f"    env: {r['env']} — {r.get('reason', '')}")

    # Stats
    stats = entry.get("stats", {})
    if stats:
        click.echo(f"\n  Stats:")
        click.echo(f"    Commits:      {stats.get('commits', '?')}")
        click.echo(f"    Contributors: {stats.get('contributors', '?')}")
        click.echo(f"    Created:      {stats.get('created', '?')}")
        click.echo(f"    Last updated: {stats.get('last_updated', '?')}")

    # Installed?
    installed_dir = SKILLS_DIR / name
    if installed_dir.is_dir():
        local = _installed_skill_meta(installed_dir)
        local_ver = local.get("version", "?") if local else "?"
        click.echo(f"\n  Installed:   yes (v{local_ver})")
    else:
        click.echo(f"\n  Install:     octo skills install {name}")

    click.echo()


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

@skills.command()
@click.argument("name")
@click.option("--no-deps", is_flag=True, help="Skip automatic dependency installation")
@click.option("--local", "local_path", default="", help="Install from local path instead of registry")
def install(name: str, no_deps: bool, local_path: str) -> None:
    """Install a skill from the marketplace (or local path)."""
    dest = SKILLS_DIR / name

    if local_path:
        # Local install — copy from path
        src = Path(local_path)
        if not (src / "SKILL.md").is_file():
            click.echo(f"Error: {src}/SKILL.md not found.")
            raise SystemExit(1)

        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        click.echo(f"Installed '{name}' from {src}")
    else:
        # Registry install
        registry = _fetch_registry()
        entry = _find_in_registry(registry, name)

        if not entry:
            click.echo(f"Skill '{name}' not found in registry. Try: octo skills search")
            raise SystemExit(1)

        files = entry.get("files", ["SKILL.md"])
        click.echo(f"Installing '{name}' v{entry.get('version', '?')}...")

        if dest.exists():
            shutil.rmtree(dest)

        _download_skill_files(name, files)
        click.echo(f"Installed to {dest}")

    if not no_deps:
        _install_deps(name)


def _install_deps(name: str) -> None:
    """Install dependencies for a skill."""
    try:
        from octo.dependencies import DependencyInstaller
    except ImportError:
        click.echo("Warning: dependency installer not available.")
        return

    import yaml
    skill_md = SKILLS_DIR / name / "SKILL.md"
    if not skill_md.is_file():
        return

    text = skill_md.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        return
    try:
        full_meta = yaml.safe_load(parts[1]) or {}
    except Exception:
        return

    deps_data = full_meta.get("dependencies", {})
    if not any(deps_data.get(k) for k in ("python", "npm", "mcp", "system")):
        click.echo("No dependencies to install.")
        return

    installer = DependencyInstaller(deps_data, full_meta.get("requires", []))
    installer.install_all()


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

@skills.command()
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def remove(name: str, yes: bool) -> None:
    """Remove an installed skill."""
    dest = SKILLS_DIR / name

    if not dest.is_dir():
        click.echo(f"Skill '{name}' is not installed.")
        return

    if not yes:
        click.confirm(f"Remove skill '{name}'?", abort=True)

    shutil.rmtree(dest)
    click.echo(f"Removed '{name}'.")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@skills.command(name="list")
def list_skills() -> None:
    """List all installed skills."""
    if not SKILLS_DIR.is_dir():
        click.echo("No skills installed.")
        return

    entries = []
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = _installed_skill_meta(d)
        if meta:
            entries.append(meta)

    if not entries:
        click.echo("No skills installed.")
        return

    click.echo(f"\n{'Name':<20} {'Version':<10} Description")
    click.echo("-" * 70)
    for e in entries:
        desc = e.get("description", "")[:40]
        click.echo(f"{e['name']:<20} {e.get('version', '?'):<10} {desc}")
    click.echo(f"\n{len(entries)} skill(s) installed.")


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

@skills.command()
@click.argument("name", default="")
@click.option("--all", "update_all", is_flag=True, help="Update all installed skills")
@click.option("--no-deps", is_flag=True, help="Skip dependency installation after update")
def update(name: str, update_all: bool, no_deps: bool) -> None:
    """Update installed skill(s) from the marketplace."""
    registry = _fetch_registry(force=True)

    if not name and not update_all:
        click.echo("Specify a skill name or use --all.")
        return

    targets: list[str] = []
    if update_all:
        if SKILLS_DIR.is_dir():
            targets = [d.name for d in sorted(SKILLS_DIR.iterdir()) if d.is_dir()]
    else:
        targets = [name]

    updated = 0
    for skill_name in targets:
        local_dir = SKILLS_DIR / skill_name
        if not local_dir.is_dir():
            click.echo(f"  {skill_name}: not installed, skipping.")
            continue

        entry = _find_in_registry(registry, skill_name)
        if not entry:
            click.echo(f"  {skill_name}: not in registry (local-only), skipping.")
            continue

        local = _installed_skill_meta(local_dir)
        local_ver = local.get("version", "0.0.0") if local else "0.0.0"
        remote_ver = entry.get("version", "0.0.0")

        if local_ver == remote_ver:
            click.echo(f"  {skill_name}: already at v{local_ver}")
            continue

        click.echo(f"  {skill_name}: updating v{local_ver} → v{remote_ver}...")
        files = entry.get("files", ["SKILL.md"])
        shutil.rmtree(local_dir, ignore_errors=True)
        _download_skill_files(skill_name, files)
        if not no_deps:
            _install_deps(skill_name)
        updated += 1

    click.echo(f"\n{updated} skill(s) updated.")
