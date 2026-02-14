"""S3-compatible storage backend.

Requires the [s3] extra: pip install octo-agent[s3]

Works with AWS S3, MinIO, and any S3-compatible service.
"""
from __future__ import annotations

import fnmatch
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _require_boto3() -> Any:
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError(
            "S3 storage requires boto3. Install with: pip install octo-agent[s3]"
        )


class S3Storage:
    """S3-compatible object storage backend.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix for all objects (e.g. "octi/" for project isolation).
        endpoint_url: S3 endpoint URL (for MinIO or custom S3).
        access_key: AWS access key ID.
        secret_key: AWS secret access key.
        region: AWS region (default: us-east-1).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: str = "",
        access_key: str = "",
        secret_key: str = "",
        region: str = "us-east-1",
    ) -> None:
        boto3 = _require_boto3()

        kwargs: dict[str, Any] = {"region_name": region}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if access_key:
            kwargs["aws_access_key_id"] = access_key
        if secret_key:
            kwargs["aws_secret_access_key"] = secret_key

        self._client = boto3.client("s3", **kwargs)
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""

    def _key(self, path: str) -> str:
        """Build full S3 key from relative path."""
        return self._prefix + path

    async def read(self, path: str) -> str:
        """Read object content as text."""
        import botocore.exceptions
        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=self._key(path),
            )
            return response["Body"].read().decode("utf-8")
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"S3 object not found: {path}")
            raise

    async def write(self, path: str, content: str) -> None:
        """Write text content to an object."""
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._key(path),
            Body=content.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

    async def append(self, path: str, content: str) -> None:
        """Append text — read + write (S3 doesn't support true append)."""
        try:
            existing = await self.read(path)
        except FileNotFoundError:
            existing = ""
        await self.write(path, existing + content)

    async def exists(self, path: str) -> bool:
        """Check if an object exists."""
        import botocore.exceptions
        try:
            self._client.head_object(
                Bucket=self._bucket,
                Key=self._key(path),
            )
            return True
        except botocore.exceptions.ClientError:
            return False

    async def list_dir(self, prefix: str = "") -> list[str]:
        """List objects under a prefix (non-recursive)."""
        full_prefix = self._key(prefix)
        if full_prefix and not full_prefix.endswith("/"):
            full_prefix += "/"

        response = self._client.list_objects_v2(
            Bucket=self._bucket,
            Prefix=full_prefix,
            Delimiter="/",
        )
        results = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            # Strip our prefix to return relative paths
            if self._prefix and key.startswith(self._prefix):
                key = key[len(self._prefix):]
            results.append(key)
        return results

    async def delete(self, path: str) -> None:
        """Delete an object. No error if missing."""
        self._client.delete_object(
            Bucket=self._bucket,
            Key=self._key(path),
        )

    async def glob(self, pattern: str) -> list[str]:
        """Find objects matching a glob pattern."""
        # List all objects under prefix and filter with fnmatch
        paginator = self._client.get_paginator("list_objects_v2")
        results = []
        for page in paginator.paginate(
            Bucket=self._bucket,
            Prefix=self._prefix,
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self._prefix and key.startswith(self._prefix):
                    key = key[len(self._prefix):]
                if fnmatch.fnmatch(key, pattern):
                    results.append(key)
        return results

    def __repr__(self) -> str:
        return (
            f"S3Storage(bucket={self._bucket!r}, "
            f"prefix={self._prefix!r})"
        )
