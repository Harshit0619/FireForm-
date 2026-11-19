"""Download the Whisper model the desktop build ships with.

Run before PyInstaller:  python scripts/fetch_whisper_model.py
Writes models/faster-whisper-small/, which fireform-backend.spec bundles.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = "faster-whisper-small"
FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.*"]

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "models" / MODEL
    snapshot_download(f"Systran/{MODEL}", local_dir=target, allow_patterns=FILES)
    print(f"Saved Systran/{MODEL} to {target}")
