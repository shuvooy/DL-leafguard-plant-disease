#!/usr/bin/env python3
"""Check that the checked-out LeafGuard repository has its deployment assets."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "leafguard.py",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "DATASETS.md",
    "MODEL_CARD.md",
    "requirements.txt",
    "requirements-train.txt",
    ".streamlit/config.toml",
    ".github/workflows/ci.yml",
    "deploy/model_config.json",
    "deploy/model_int8.onnx",
    "deploy/SHA256SUMS",
    "deploy/MODEL_CARD.md",
    "deploy/requirements.txt",
    "docs/METHODOLOGY.md",
    "docs/RESULTS.md",
    "docs/DEPLOYMENT.md",
    "tests/test_leafguard.py",
    "results/summary_metrics.json",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        fail("missing required files: " + ", ".join(missing))

    empty = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file() and path.stat().st_size == 0]
    if empty:
        fail("zero-byte regular files: " + ", ".join(empty))

    cfg = json.loads((ROOT / "deploy/model_config.json").read_text(encoding="utf-8"))
    if cfg.get("run") != "convnext_tiny" or len(cfg.get("classes", [])) != 109:
        fail("deployment config is not the 109-class convnext_tiny export")

    model = ROOT / "deploy/model_int8.onnx"
    if model.stat().st_size < 20_000_000:
        fail("INT8 ONNX model is unexpectedly small")
    if (ROOT / "deploy/model.onnx").exists():
        fail("optional FP32 model should not be in this GitHub-ready package")

    samples = sorted((ROOT / "deploy/samples").glob("*.jpg"))
    if len(samples) != 10:
        fail(f"expected 10 JPG sample photos, found {len(samples)}")
    for sample in samples:
        try:
            with Image.open(sample) as image:
                image.verify()
        except Exception as exc:
            fail(f"invalid sample image {sample.name}: {exc}")

    if (ROOT / "requirements.txt").read_text() != (ROOT / "deploy/requirements.txt").read_text():
        fail("root and deploy requirements.txt files do not match")

    # Verify the published model/config/sample checksum manifest.
    for line in (ROOT / "deploy/SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relpath = line.split(maxsplit=1)
        relpath = relpath.lstrip("* ")
        target = ROOT / relpath
        if not target.is_file():
            fail(f"checksum target is missing: {relpath}")
        import hashlib
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != expected:
            fail(f"checksum mismatch: {relpath}")

    # Check relative links and embedded images in Markdown; external URLs are left online.
    broken = []
    pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for markdown in ROOT.rglob("*.md"):
        content = markdown.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(content):
            target = match.group(1).split("#", 1)[0].strip()
            if not target or urlparse(target).scheme in {"http", "https", "mailto", "data"} or target.startswith("#"):
                continue
            if not (markdown.parent / target).resolve().exists():
                broken.append(f"{markdown.relative_to(ROOT)} -> {target}")
    if broken:
        fail("broken local Markdown links: " + "; ".join(broken))

    print("LeafGuard package verification passed")
    print(f"- model: {cfg['run']} / {len(cfg['classes'])} classes / {model.stat().st_size:,} bytes")
    print(f"- sample images: {len(samples)} (all decode successfully)")
    print("- no zero-byte regular files")
    print("- local Markdown links resolve")
    print("- root and deploy requirements match")
    print("- SHA256SUMS manifest verifies")


if __name__ == "__main__":
    main()
