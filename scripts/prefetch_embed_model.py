#!/usr/bin/env python3
"""Pre-download MELON's local embedding model into ``$HF_HOME``.

MELON's fully-local detection backend (``MELON_EMBED_PROVIDER=sentence-transformers``)
needs a Hugging Face embedding model. This script downloads it ahead of time so
the first benchmark run doesn't block on the download, and so it can run offline.

The model is resolved from (in order): a command-line argument, the
``MELON_EMBED_MODEL`` environment variable, or the default
``BAAI/bge-large-en-v1.5``. Files are cached under ``$HF_HOME`` (defaults to
``~/.cache/huggingface``).

Usage:
    # uses MELON_EMBED_MODEL from your environment / .env, else the default
    python scripts/prefetch_embed_model.py

    # explicit model
    python scripts/prefetch_embed_model.py BAAI/bge-m3

    # explicit cache location for just this run
    HF_HOME=/data/hf_cache python scripts/prefetch_embed_model.py
"""

import os
import sys

# The OpenAI default isn't a Hugging Face repo, so treat it as "not set" here.
OPENAI_EMBED_DEFAULT = "text-embedding-3-large"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def resolve_model(argv: list[str]) -> str:
    """Resolve the embedding model to download from CLI args / env / default."""
    if len(argv) > 1 and argv[1].strip():
        return argv[1].strip()
    env_model = (os.getenv("MELON_EMBED_MODEL") or "").strip()
    if env_model and env_model != OPENAI_EMBED_DEFAULT:
        return env_model
    return DEFAULT_MODEL


def main() -> None:
    # Load a local .env if present so HF_HOME / MELON_EMBED_MODEL are picked up
    # even without `set -a && source .env`.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    model = resolve_model(sys.argv)
    hf_home = os.getenv("HF_HOME")
    print(f"Embedding model : {model}")
    print(f"HF_HOME         : {hf_home or '(unset -> ~/.cache/huggingface)'}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed. Install MELON's deps first:\n"
            "    cd agentdojo && pip install -e \".[melon]\""
        )

    path = snapshot_download(model)
    print(f"Done. Cached at : {path}")


if __name__ == "__main__":
    main()
