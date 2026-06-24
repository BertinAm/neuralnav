"""Shared helper for pulling trained artifacts from the Hugging Face Hub
repo that notebooks/01_intent_classification.ipynb and
notebooks/02_retrieval_and_ner.ipynb push to, when they aren't present in
the local models/ directory (e.g. fresh clone, fresh Docker build).
"""
import os
from pathlib import Path

HF_REPO_ID = os.environ.get("HF_REPO_ID", "unixio/neuralnav-intent-models")
HF_TOKEN = os.environ.get("HF_TOKEN")


def ensure_file(local_path: Path, hub_filename: str) -> Path | None:
    """Return local_path if it already exists, otherwise try downloading
    hub_filename from HF_REPO_ID into local_path. Returns None if neither
    works (no token, repo/file missing, or network unavailable)."""
    if local_path.exists():
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID, filename=hub_filename, token=HF_TOKEN
        )
    except Exception:
        return None

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(Path(downloaded).read_bytes())
    return local_path


def ensure_dir(local_dir: Path, hub_prefix: str) -> Path | None:
    """Same as ensure_file but for a directory of files (e.g. bert_intent/),
    downloaded via snapshot_download with an allow_patterns filter."""
    if local_dir.exists() and any(local_dir.iterdir()):
        return local_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    try:
        snapshot_path = snapshot_download(
            repo_id=HF_REPO_ID,
            token=HF_TOKEN,
            allow_patterns=[f"{hub_prefix}/*"],
        )
    except Exception:
        return None

    downloaded_dir = Path(snapshot_path) / hub_prefix
    if not downloaded_dir.exists():
        return None

    import shutil

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(downloaded_dir, local_dir, dirs_exist_ok=True)
    return local_dir
