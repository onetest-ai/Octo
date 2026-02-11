# Octo Skills Marketplace - Implementation Plan

**Created:** 2026-02-11  
**Status:** Draft  
**Goal:** Transform Octo skills from local-only markdown files into a GitHub-based, dependency-aware marketplace with zero backend infrastructure

---

## 📋 Overview

Create a skills marketplace that:
- Uses GitHub as distribution platform (no backend needed)
- Supports dependency declarations (Python, npm, MCP servers, system packages)
- Tracks skill quality via git stats (commits, contributors, activity)
- Provides CLI for search/install/manage
- Enables community contributions via PRs

**Core Philosophy:** Skills = Markdown + YAML frontmatter + optional scripts (NOT heavy npm packages like OpenClaw)

---

## 🎯 Phase 1: Repository & Structure (Week 1)

### Task 1.1: Create GitHub Repository
**Goal:** Setup `onetest-ai/skills` as the central registry

**Repository Structure:**
```
onetest-ai/skills/
├── README.md                    # Marketplace overview
├── CONTRIBUTING.md              # How to submit skills
├── registry.json                # Auto-generated skill index
├── skills/
│   ├── quick/
│   │   ├── SKILL.md            # Agent instructions with frontmatter
│   │   └── README.md           # Human documentation
│   ├── verify/
│   │   └── SKILL.md
│   ├── map-codebase/
│   │   └── SKILL.md
│   └── discuss/
│       └── SKILL.md
├── templates/
│   ├── SKILL_TEMPLATE.md       # Template for new skills
│   └── skill-metadata-schema.json
├── scripts/
│   ├── build_registry.py       # Generate registry.json
│   ├── validate_skill.py       # Pre-commit validation
│   └── stats.py                # Calculate git-based stats
└── .github/
    └── workflows/
        ├── update-registry.yml  # Daily registry rebuild
        └── validate-pr.yml      # PR skill validation
```

### Deliverables:
1. **README.md** - Marketplace introduction, usage instructions
2. **CONTRIBUTING.md** - Submission guidelines, validation rules
3. **templates/SKILL_TEMPLATE.md** - Boilerplate for new skills with annotated frontmatter

---

## 🔧 Phase 2: Registry Build System (Week 1-2)

### Task 2.1: Build Registry Generator
**File:** `scripts/build_registry.py`

**Purpose:** Auto-generate `registry.json` from `skills/` folder with git-based stats

**Features:**
- Parse YAML frontmatter from each SKILL.md
- Calculate git stats:
  - Commit count (activity level)
  - Contributor count (collaboration signal)
  - Created date (first commit)
  - Last updated (most recent commit)
- Output structured JSON for CLI consumption

**Key Functions:**
```python
def parse_frontmatter(skill_md: Path) -> dict
def get_git_stats(skill_path: Path) -> dict
def build_registry() -> None
```

### Task 2.2: GitHub Actions Automation
**Files:**
- `.github/workflows/update-registry.yml` - Daily registry rebuild
- `.github/workflows/validate-pr.yml` - PR validation

**Triggers:**
- Push to `main` (skills/** changes)
- Daily cron (00:00 UTC)
- Manual workflow dispatch

**Actions:**
1. Checkout with full git history
2. Install Python + dependencies (PyYAML)
3. Run `build_registry.py`
4. Commit registry.json if changed (skip CI to prevent loops)

### Task 2.3: Skill Validation
**File:** `scripts/validate_skill.py`

**Security Checks:**
- Detect dangerous commands (`rm -rf`, `curl | bash`, `eval()`)
- Validate YAML frontmatter structure
- Check required fields (name, version, description)
- Verify semver format
- Ensure no direct disk writes

**Runs on:** Every PR that modifies `skills/**`

---

## 💻 Phase 3: CLI Implementation (Week 2-3)

### Task 3.1: Skills CLI Commands
**File:** `octo/cli_skills.py` (new file)

**Commands:**

#### `octo skills search [query]`
- Search by name/description
- Filter by `--tag`
- Display: name, version, description, stats, dependencies

#### `octo skills info <name>`
- Detailed skill information
- Full dependency list
- Requirements
- Activity stats
- Installation command

#### `octo skills install <name>`
- Download SKILL.md from GitHub
- Save to `~/.octo/skills/<name>/`
- `--deps` flag: auto-install dependencies
- `--local` flag: install from local path (for testing)

#### `octo skills remove <name>`
- Delete skill directory
- Run pre-uninstall hook if exists

#### `octo skills list`
- Show all installed skills
- Parse version from SKILL.md

#### `octo skills update [name]`
- Fetch latest version from registry
- Re-install if version changed

**Registry Fetching:**
- URL: `https://raw.githubusercontent.com/onetest-ai/skills/main/registry.json`
- Cache for 1 hour (optional optimization)

### Task 3.2: Wire to Main CLI
**File:** `octo/cli.py`

Add:
```python
from octo.cli_skills import skills
cli.add_command(skills)
```

---

## 🔌 Phase 4: Dependency Management (Week 3-4)

### Task 4.1: Dependency Installer
**File:** `octo/dependencies.py` (new file)

**Class:** `DependencyInstaller`

**Methods:**

#### `check_requirements() -> bool`
- Verify required commands exist (`git`, `npm`, etc.)
- Check for required environment variables

#### `install_python() -> bool`
- Parse `dependencies.python` list
- Run `pip install <package>` for each
- Capture errors

#### `install_npm() -> bool`
- Parse `dependencies.npm` list
- Run `npm install -g <package>`
- Requires npm installed

#### `configure_mcp() -> bool`
- Parse `dependencies.mcp` list
- Load existing `~/.mcp.json`
- Append new MCP server configs
- Save and notify user to restart

#### `prompt_system_packages()`
- Display `dependencies.system` list
- Detect OS (macOS/Linux/Windows)
- Suggest install command (`brew`, `apt`, `pacman`)
- **Never auto-install** (security risk)

#### `install_all(interactive: bool) -> bool`
- Orchestrate all installation steps
- Report errors clearly

### Task 4.2: YAML Frontmatter Schema

**Full Specification:**
```yaml
---
# Required fields
name: skill-name              # Identifier (lowercase-with-dashes)
version: 1.0.0                # Semantic versioning
author: github-username       # Author identifier
description: Brief summary    # One-liner for search results
tags: [cat1, cat2]            # For filtering

# Dependencies
dependencies:
  python:                     # pip packages
    - playwright>=1.40.0
    - pyyaml~=6.0
  npm:                        # npm packages (global install)
    - @playwright/browser-chromium
  mcp:                        # MCP servers
    - server: playwright      # Server name in .mcp.json
      package: "@modelcontextprotocol/server-playwright"
      args: ["--browser", "chromium"]
      env:                    # Optional environment variables
        PLAYWRIGHT_BROWSERS_PATH: /usr/local/bin
  system:                     # System packages (prompt only)
    - chromium
    - git

# Pre-checks (validated before install)
requires:
  - command: git              # Required CLI command
    reason: "Needed for repository cloning"
  - env: OPENAI_API_KEY       # Required env var
    reason: "Used for LLM calls"

# Optional hooks
hooks:
  post_install: scripts/setup.sh     # Run after install
  pre_uninstall: scripts/cleanup.sh  # Run before removal

# Security/Permissions (future)
permissions:
  filesystem: read            # read|write|execute
  network: true               # Allow network calls
  shell: true                 # Allow subprocess execution
  mcp_servers: [playwright]   # Which MCP servers used
---
```

---

## 📦 Phase 5: Migration (Week 4)

### Task 5.1: Migrate Existing Skills

**Current Skills to Migrate:**
1. `quick` - Fast ad-hoc fixes
2. `verify` - Conversational testing
3. `map-codebase` - Parallel codebase analysis
4. `discuss` - Requirements gathering

**Migration Steps for Each Skill:**
1. Create `onetest-ai/skills/skills/<name>/` directory
2. Move SKILL.md and add YAML frontmatter
3. Create README.md (human documentation)
4. Document any dependencies
5. Add to registry via `build_registry.py`

**Frontmatter Template:**
```yaml
---
name: quick
version: 1.0.0
author: onetest-ai
description: Execute ad-hoc tasks quickly without heavyweight planning
tags: [development, quick-fix, utility]

dependencies:
  python: []    # No external deps
  npm: []
  mcp: []
  system: []

requires: []    # No special requirements

permissions:
  filesystem: write
  network: false
  shell: true
---
```

### Task 5.2: Update Octo Core

**File:** `octo/agent.py` (or wherever skills are loaded)

**Changes:**
- Auto-discover skills from `~/.octo/skills/` on startup
- Parse SKILL.md frontmatter for metadata
- Populate `/skills` command dynamically
- Use skill names from frontmatter, not folder names

---

## 📊 Phase 6: Stats & Discovery (Week 4-5)

### Task 6.1: Enhanced Registry Stats

**Git-Based Metrics:**
```json
"stats": {
  "commits": 23,              // Activity level
  "contributors": 5,          // Collaboration indicator
  "created": "2025-11-15",    // Age
  "last_updated": "2026-02-10" // Maintenance signal
}
```

**Future: Optional Telemetry** (opt-in)
```json
"telemetry_stats": {
  "installs": 342,            // From telemetry backend
  "active_users": 89          // Unique users last 30 days
}
```

### Task 6.2: Search Ranking

**Factors (in order):**
1. **Relevance** - Query match in name/description
2. **Maintenance** - Recent commits
3. **Activity** - High commit count
4. **Collaboration** - Multiple contributors
5. **Maturity** - Version >= 1.0.0

---

## 🛡️ Phase 7: Security & Quality (Week 5)

### Task 7.1: Security Validation

**Automated Checks:**
- Scan for dangerous shell commands
- Check for eval/exec in Python
- Detect curl piping
- Flag direct disk writes
- Validate package names (no typosquatting)

**Manual Review:**
- All new skills reviewed by maintainer
- PR requires approval before merge
- Community reporting for malicious skills

### Task 7.2: Skill Permissions (Future)

**Declaration in Frontmatter:**
```yaml
permissions:
  filesystem: read    # read|write|execute
  network: true
  shell: false
  mcp_servers: [github, playwright]
```

**Enforcement:**
- User prompted on first install
- Permissions stored in `~/.octo/skills/<name>/permissions.json`
- Sandbox mode (future): restrict actual capabilities

---

## 📚 Phase 8: Documentation (Week 5-6)

### Task 8.1: User Documentation

**Files to Create:**
1. `docs/skills-marketplace.md` - User guide
2. `docs/creating-skills.md` - Author guide
3. `docs/dependency-management.md` - Dependency deep-dive

**Topics:**
- How to search and install skills
- Understanding skill stats
- Creating your first skill
- Dependency best practices
- Security considerations
- Troubleshooting

### Task 8.2: Video Demos (Optional)

- "Installing Your First Skill"
- "Creating a Custom Skill"
- "Contributing to the Marketplace"

---

## 🚀 Phase 9: Launch (Week 6)

### Task 9.1: Announce

**Channels:**
- GitHub README update
- Twitter/X announcement
- Discord/Slack communities
- Dev.to blog post
- Product Hunt launch

### Task 9.2: Gather Feedback

- Create feedback issue template
- Monitor GitHub issues
- Track CLI usage (if telemetry enabled)
- Community survey after 30 days

---

## 📅 Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **1. Repository Setup** | Week 1 | GitHub repo, templates, docs |
| **2. Registry Build** | Week 1-2 | Auto-generated registry.json |
| **3. CLI Commands** | Week 2-3 | `octo skills` CLI fully working |
| **4. Dependencies** | Week 3-4 | Auto-install Python/npm/MCP |
| **5. Migration** | Week 4 | 4 existing skills in registry |
| **6. Stats & Discovery** | Week 4-5 | Enhanced search ranking |
| **7. Security** | Week 5 | Validation + permissions |
| **8. Documentation** | Week 5-6 | Complete user/author guides |
| **9. Launch** | Week 6 | Public announcement |

**Total:** ~6 weeks for full implementation

---

## 🎯 Success Metrics

**Month 1:**
- [ ] 10+ skills in registry
- [ ] 50+ installs
- [ ] 3+ community contributions

**Month 3:**
- [ ] 25+ skills
- [ ] 200+ installs
- [ ] 10+ contributors

**Month 6:**
- [ ] 50+ skills
- [ ] 500+ installs
- [ ] Active community discussions

---

## 🔮 Future Enhancements

### Phase 10+: Advanced Features
- **Skill Versioning** - Install specific versions
- **Dependency Locking** - `skills-lock.json` for reproducibility
- **Skill Aliases** - `octo install quick-fix` → `quick`
- **Skill Updates** - `octo skills update --all`
- **Private Registries** - Enterprise skill repos
- **Skill Collections** - "web-automation-bundle"
- **Telemetry Dashboard** - Public stats page
- **Skill Templates** - `octo skills create <name>` scaffolding
- **Testing Framework** - Automated skill testing
- **Skill Marketplace UI** - Web interface for browsing

---

## 💡 Key Design Decisions

### Why GitHub (not npm)?
- ✅ Skills are markdown, not code packages
- ✅ Zero distribution cost
- ✅ Familiar workflow (PR-based)
- ✅ Built-in versioning (git tags)
- ✅ No packaging overhead

### Why Git Stats (not downloads)?
- ✅ No backend required
- ✅ Reflects real maintenance
- ✅ Privacy-friendly
- ✅ Can add telemetry later if needed

### Why YAML Frontmatter (not separate manifest)?
- ✅ Single file per skill
- ✅ Human-readable
- ✅ Easy to parse
- ✅ Standard in static site generators

### Why Opt-In Dependencies?
- ✅ Security (user controls what's installed)
- ✅ Transparency (user sees all deps)
- ✅ Flexibility (use --deps or manual)

---

## 📝 Implementation Notes

### Registry URL
```
https://raw.githubusercontent.com/onetest-ai/skills/main/registry.json
```

### Skill Installation Path
```
~/.octo/skills/<skill-name>/SKILL.md
```

### MCP Configuration
```
~/.mcp.json (auto-updated by installer)
```

### Telemetry (Future)
```
https://telemetry.onetest-ai.dev/skill-install (Cloudflare Worker)
```

---

## 🐛 Known Limitations

1. **GitHub Rate Limits** - Unauthenticated: 60 req/hour (should be fine for typical usage)
2. **No Semantic Search** - Registry search is keyword-based only
3. **No Dependency Resolution** - No conflict detection between skills
4. **Manual System Packages** - Can't auto-install OS packages (security)
5. **MCP Restart Required** - After installing MCP-dependent skills

---

## 🤝 Comparison: Octo vs OpenClaw

| Feature | OpenClaw | Octo Skills Marketplace |
|---------|----------|------------------------|
| **Distribution** | npm packages (700+ skills) | GitHub repo (markdown) |
| **Size** | Megabytes per skill | Kilobytes per skill |
| **Language** | JavaScript/TypeScript only | Language-agnostic |
| **Backend** | Required (ClawHub API) | Zero backend (GitHub only) |
| **Versioning** | npm semver | Git tags |
| **Installation** | `npm install` | `octo skills install` |
| **Dependencies** | package.json | YAML frontmatter |
| **Stats** | Download counts (API) | Git commits (calculated) |
| **Portability** | Requires Node.js | Plain markdown files |
| **Security** | npm audit | Custom validation |

**Conclusion:** Octo's approach is lighter, more portable, and doesn't require backend infrastructure.

---

## 📞 Support & Contact

- **GitHub Issues:** `onetest-ai/skills`
- **Discussions:** GitHub Discussions
- **Discord:** (if exists)

---

**End of Implementation Plan**

*This document will evolve as implementation progresses. Check git history for changes.*
