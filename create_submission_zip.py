"""Create the official VoiceOps 2026 capstone submission ZIP file."""

import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ZIP_NAME = ROOT / "sumitsaini03_voiceops.zip"

# Ensure demo.mp4 exists (if user has not recorded their 2-min demo yet, create a placeholder)
demo_file = ROOT / "demo.mp4"
if not demo_file.exists():
    with open(demo_file, "wb") as f:
        # Write a minimal MP4 container header / placeholder note
        f.write(b"DEMO_MP4_PLACEHOLDER: Replace this file with your 2-minute recorded demo.mp4 video before final live judging.")

# Files and directories to package strictly matching Capstone page 14 specification
INCLUDED_FILES = [
    "README.md",
    ".env.example",
    "requirements.txt",
    "run.py",
    "report.pdf",
    "demo_script.md",
    "demo.mp4",
]

INCLUDED_DIRS = [
    "src",
    "sample_outputs",
]

# Excluded patterns (Keep secrets out: .env, API keys, virtual environments, dataset copy)
EXCLUDED_PATTERNS = [
    ".env",
    "*.pkl",
    "__pycache__",
    "*.pyc",
    ".git",
    "data",
    "tickets_created.csv",
    ".pytest_cache",
]

def should_exclude(rel_path: str) -> bool:
    for pat in EXCLUDED_PATTERNS:
        if pat.startswith("*") and rel_path.endswith(pat[1:]):
            return True
        if f"/{pat}/" in f"/{rel_path}/" or rel_path == pat or rel_path.startswith(f"{pat}/"):
            return True
    return False

def build_zip():
    print(f"Creating submission zip: {ZIP_NAME.name}...")
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        # Add single files
        for fname in INCLUDED_FILES:
            fpath = ROOT / fname
            if fpath.exists():
                print(f"  Adding file: {fname}")
                z.write(fpath, arcname=fname)
            else:
                print(f"  [WARNING] File missing: {fname}")

        # Add directories recursively
        for dname in INCLUDED_DIRS:
            dpath = ROOT / dname
            if dpath.exists() and dpath.is_dir():
                for root, dirs, files in os.walk(dpath):
                    for file in files:
                        full_path = Path(root) / file
                        rel_path = full_path.relative_to(ROOT).as_posix()
                        if not should_exclude(rel_path):
                            print(f"  Adding: {rel_path}")
                            z.write(full_path, arcname=rel_path)

    print(f"\nSuccessfully generated {ZIP_NAME} ({os.path.getsize(ZIP_NAME):,} bytes)!")

if __name__ == "__main__":
    build_zip()
