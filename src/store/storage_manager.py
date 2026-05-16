"""
storage_manager.py
──────────────────
Production-grade caching abstraction for ML pipeline storage.

Contract
────────
  • Every artifact is addressed by (stage, filename).
  • Reads:  LOCAL first → S3 fallback → auto-populate local cache.
  • Writes: write-through (local + S3), skip local if already cached.
  • Callers never touch storage clients directly.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import yaml

from src.core.logger import pipeline_logger
from src.pipeline.pipeline_enum import PipelineEnum, StorageType


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

class StorageBackend(ABC):
    """Minimal stable interface every backend must satisfy."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def list_prefix(self, prefix: str) -> list[str]:
        """Return all keys that start with *prefix*."""
        ...

    @abstractmethod
    def copy_key(self, src_key: str, dst_key: str) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL BACKEND
# ─────────────────────────────────────────────────────────────────────────────

class LocalBackend(StorageBackend):
    def __init__(self, base_path: str | Path = "/app/ladybug/assets"):
        self.root = Path(base_path)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def list_prefix(self, prefix: str) -> list[str]:
        search_root = self.root / prefix
        if not search_root.exists():
            return []
        return [
            str(p.relative_to(self.root))
            for p in search_root.rglob("*")
            if p.is_file()
        ]

    def copy_key(self, src_key: str, dst_key: str) -> None:
        src = self._path(src_key)
        dst = self._path(dst_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# S3 BACKEND
# ─────────────────────────────────────────────────────────────────────────────

class S3Backend(StorageBackend):
    def __init__(self, s3_client, bucket: str):
        self._client = s3_client
        self._bucket = bucket

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def read_bytes(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self._bucket, Key=key)
        return obj["Body"].read()

    def write_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def list_prefix(self, prefix: str) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def copy_key(self, src_key: str, dst_key: str) -> None:
        self._client.copy_object(
            Bucket=self._bucket,
            CopySource={"Bucket": self._bucket, "Key": src_key},
            Key=dst_key,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZERS
# ─────────────────────────────────────────────────────────────────────────────

class _Serializer:
    @staticmethod
    def df_to_bytes(df: pd.DataFrame, fmt: str) -> bytes:
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        if fmt == "jsonl":
            return df.to_json(orient="records", lines=True, force_ascii=False).encode()
        raise ValueError(f"Unknown DataFrame format: {fmt!r}")

    @staticmethod
    def bytes_to_df(data: bytes, fmt: str) -> pd.DataFrame:
        if fmt == "csv":
            return pd.read_csv(io.BytesIO(data))
        if fmt == "jsonl":
            return pd.read_json(io.BytesIO(data), lines=True)
        raise ValueError(f"Unknown DataFrame format: {fmt!r}")

    @staticmethod
    def dict_to_bytes(data: dict, fmt: str) -> bytes:
        if fmt == "json":
            return json.dumps(data, ensure_ascii=False, indent=2).encode()
        if fmt == "yaml":
            return yaml.dump(data, allow_unicode=True).encode()
        raise ValueError(f"Unknown dict format: {fmt!r}")

    @staticmethod
    def bytes_to_dict(data: bytes, fmt: str) -> dict:
        if fmt == "json":
            return json.loads(data.decode())
        if fmt == "yaml":
            return yaml.safe_load(data.decode())
        raise ValueError(f"Unknown dict format: {fmt!r}")


# ─────────────────────────────────────────────────────────────────────────────
# STORAGE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class StorageManager:
    """
    Unified read/write layer with automatic local caching.

    All artifacts are identified by (stage, filename).
    Full storage key: {project}/{version}/{stage}/{filename}

    Parameters
    ----------
    project       : model name (e.g. "commment-intent")
    version       : pipeline run version (e.g. "v3")
    local_backend : LocalBackend instance, or None to disable
    s3_backend    : S3Backend instance, or None to disable
    """

    PIPELINE_FOLDERS = [
        "raw_data",
        "processed_data",
        "training",
        "training/training_data",
        "training/training_configs",
        "training/model_output",
        "training/evaluation",
        "mlflow",
        "mlflow/artifacts",
        "model",
        "logs",
        "benchmark",
    ]

    def __init__(
        self,
        project: str,
        version: str,
        local_backend: LocalBackend | None = None,
        s3_backend: S3Backend | None = None,
        logger: logging.Logger | None = None,
    ):
        if not local_backend and not s3_backend:
            raise ValueError("At least one storage backend must be provided.")

        self.project = project
        self.version = version
        self._local = local_backend
        self._s3    = s3_backend
        self._log   = logger or pipeline_logger

        self._log.info(
            f"[{PipelineEnum.STORAGE}] StorageManager ready "
            f"(local={'✓' if self._local else '✗'}, "
            f"s3={'✓' if self._s3 else '✗'})"
        )

    # ── Key construction ─────────────────────────────────────────────────────

    def _key(self, stage: str, filename: str) -> str:
        """Single source of truth for storage path construction."""
        return f"{self.project}/{self.version}/{stage}/{filename}"

    # ── Core primitives ──────────────────────────────────────────────────────

    def _resolve(self, stage: str, filename: str) -> bytes:
        """
        LOCAL hit → return immediately.
        S3 hit    → populate local cache → return.
        """
        key = self._key(stage, filename)

        if self._local and self._local.exists(key):
            self._log.debug(f"[{PipelineEnum.STORAGE}] [{StorageType.LOCAL}] cache hit: {key}")
            return self._local.read_bytes(key)

        if self._s3 and self._s3.exists(key):
            self._log.debug(f"[{PipelineEnum.STORAGE}] [{StorageType.S3}] fetching: {key}")
            data = self._s3.read_bytes(key)
            if self._local:
                self._local.write_bytes(key, data)
                self._log.debug(f"[{PipelineEnum.STORAGE}] [{StorageType.LOCAL}] cached: {key}")
            return data

        raise FileNotFoundError(
            f"Artifact not found in any backend: {key}\n"
            f"  local={'checked' if self._local else 'disabled'}, "
            f"  s3={'checked' if self._s3 else 'disabled'}"
        )

    def _write(self, stage: str, filename: str, data: bytes, *, overwrite: bool = False) -> None:
        """
        Write-through: local (guarded) + S3 (always).
        Local write is skipped if artifact already exists, unless overwrite=True.
        """
        key = self._key(stage, filename)

        if self._local:
            if overwrite or not self._local.exists(key):
                self._local.write_bytes(key, data)
                self._log.debug(f"[{PipelineEnum.STORAGE}] [{StorageType.LOCAL}] wrote: {key}")
            else:
                self._log.debug(f"[{PipelineEnum.STORAGE}] [{StorageType.LOCAL}] skipped (cached): {key}")

        if self._s3:
            self._s3.write_bytes(key, data)
            self._log.debug(f"[{PipelineEnum.STORAGE}] [{StorageType.S3}] wrote: {key}")

    # ── Existence / discovery ─────────────────────────────────────────────────

    def exists(self, stage: str, filename: str) -> bool:
        key = self._key(stage, filename)
        if self._local and self._local.exists(key):
            return True
        if self._s3 and self._s3.exists(key):
            return True
        return False

    def stage_exists(self, stage: str) -> bool:
        """Return True if any file exists inside a stage folder."""
        prefix = self._key(stage, "")
        if self._local and self._local.list_prefix(prefix):
            return True
        if self._s3 and self._s3.list_prefix(prefix):
            return True
        return False

    # ── DataFrame ─────────────────────────────────────────────────────────────

    def save_df(self, df: pd.DataFrame, stage: str, filename: str) -> None:
        fmt = "jsonl" if filename.endswith(".jsonl") else "csv"
        self._write(stage, filename, _Serializer.df_to_bytes(df, fmt))
        self._log.info(
            f"[{PipelineEnum.STORAGE}] saved DataFrame → {stage}/{filename} ({len(df)} rows)"
        )

    def load_df(self, stage: str, filename: str) -> pd.DataFrame:
        fmt = "jsonl" if filename.endswith(".jsonl") else "csv"
        df = _Serializer.bytes_to_df(self._resolve(stage, filename), fmt)
        self._log.info(
            f"[{PipelineEnum.STORAGE}] loaded DataFrame ← {stage}/{filename} ({len(df)} rows)"
        )
        return df

    # ── JSON ──────────────────────────────────────────────────────────────────

    def save_json(self, payload: dict, stage: str, filename: str) -> None:
        self._write(stage, filename, _Serializer.dict_to_bytes(payload, "json"))
        self._log.info(f"[{PipelineEnum.STORAGE}] saved JSON → {stage}/{filename}")

    def load_json(self, stage: str, filename: str) -> dict:
        return _Serializer.bytes_to_dict(self._resolve(stage, filename), "json")

    # ── YAML ──────────────────────────────────────────────────────────────────

    def save_yaml(self, payload: dict, stage: str, filename: str) -> None:
        self._write(stage, filename, _Serializer.dict_to_bytes(payload, "yaml"))
        self._log.info(f"[{PipelineEnum.STORAGE}] saved YAML → {stage}/{filename}")

    def load_yaml(self, stage: str, filename: str) -> dict:
        return _Serializer.bytes_to_dict(self._resolve(stage, filename), "yaml")

    # ── Raw bytes ─────────────────────────────────────────────────────────────

    def save_bytes(self, data: bytes, stage: str, filename: str) -> None:
        self._write(stage, filename, data)

    def load_bytes(self, stage: str, filename: str) -> bytes:
        return self._resolve(stage, filename)

    # ── Disk / RunPod helpers ─────────────────────────────────────────────────

    def write_yaml_to_disk(self, payload: dict, abs_path: str) -> None:
        """Write a YAML dict directly to an absolute local path (RunPod workspace)."""
        path = Path(abs_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(payload, allow_unicode=True), encoding="utf-8")
        self._log.info(f"[{PipelineEnum.STORAGE}] YAML written to disk: {abs_path}")

    def download_to_disk(self, stage: str, filename: str, abs_path: str) -> None:
        """Resolve an artifact and write it to an absolute local path."""
        data = self._resolve(stage, filename)
        path = Path(abs_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._log.info(f"[{PipelineEnum.STORAGE}] downloaded {stage}/{filename} → {abs_path}")

    def upload_folder_to_s3(self, local_dir: str, stage: str) -> int:
        """
        Upload every file in local_dir to S3 under stage/.
        Returns the number of files uploaded.
        """
        if not self._s3:
            raise RuntimeError("upload_folder_to_s3 requires an S3 backend.")

        local_root = Path(local_dir)
        if not local_root.exists():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")

        uploaded = 0
        for file_path in local_root.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(local_root)
            key = self._key(stage, str(relative))
            self._s3.write_bytes(key, file_path.read_bytes())
            uploaded += 1

        self._log.info(
            f"[{PipelineEnum.STORAGE}] [{StorageType.S3}] "
            f"uploaded {uploaded} files → {stage}/"
        )
        return uploaded

    def copy_stage(self, src_stage: str, dst_stage: str) -> int:
        """
        Copy all artifacts from src_stage into dst_stage within the same
        project/version. Used to snapshot benchmark data into a versioned folder.
        """
        backend = self._s3 or self._local
        src_prefix = self._key(src_stage, "")
        keys = backend.list_prefix(src_prefix)
        copied = 0
        for src_key in keys:
            filename = src_key[len(src_prefix):]
            dst_key = self._key(dst_stage, filename)
            backend.copy_key(src_key, dst_key)
            copied += 1

        self._log.info(
            f"[{PipelineEnum.STORAGE}] copied {copied} files: {src_stage}/ → {dst_stage}/"
        )
        return copied

    def create_folder_structure(self, folders: list[str] | None = None) -> None:
        """
        Create .keep placeholder markers for every folder.
        Defaults to PIPELINE_FOLDERS if not provided.
        """
        folders = folders or self.PIPELINE_FOLDERS
        for folder in folders:
            if not self.exists(folder, ".keep"):
                self._write(folder, ".keep", b"")
        self._log.info(
            f"[{PipelineEnum.STORAGE}] folder structure created ({len(folders)} folders)"
        )