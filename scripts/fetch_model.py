#!/usr/bin/env python3
"""Copy or export the multilingual-e5-base OpenVINO model.

Two modes:
    fetch_model.py --from /path/to/Multilingual-E5-Base
        Copy an already converted OpenVINO IR directory (fast, offline).

    fetch_model.py --download
        Download intfloat/multilingual-e5-base from HuggingFace and export it
        with optimum-cli (requires requirements-model.txt).

The default destination is ~/.local/share/npu-search/models/Multilingual-E5-Base.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

MODEL_NAME = "Multilingual-E5-Base"


def data_dir() -> str:
    base = os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share"))
    return os.path.join(base, "npu-search")


def default_dest() -> str:
    return os.environ.get("NPU_MODEL_DIR", os.path.join(data_dir(), "models", MODEL_NAME))


def model_xml(dest: str) -> str:
    nested = os.path.join(dest, "openvino", "openvino_model.xml")
    if os.path.isfile(nested):
        return nested
    return os.path.join(dest, "openvino_model.xml")


def validate(dest: str) -> bool:
    required = [
        model_xml(dest),
        os.path.join(dest, "tokenizer.json"),
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        print("[ERROR] Model is incomplete, missing:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return False
    return True


def copy_model(source: str, dest: str, force: bool) -> int:
    source = os.path.abspath(os.path.expanduser(source))
    if not validate(source):
        return 1

    if os.path.abspath(source) == os.path.abspath(dest):
        print(f"[INFO] Model already at {dest}")
        return 0

    if os.path.exists(dest) and not force:
        print(f"[ERROR] Destination exists: {dest} (use --force to replace)", file=sys.stderr)
        return 1
    if os.path.exists(dest) and force:
        shutil.rmtree(dest)

    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    print(f"[INFO] Copying {source} -> {dest}")
    shutil.copytree(source, dest)
    print("[INFO] Done.")
    return 0 if validate(dest) else 1


def download_model(dest: str, force: bool) -> int:
    optimum = os.path.join(os.path.dirname(sys.executable), "optimum-cli")
    if not os.path.isfile(optimum):
        optimum = shutil.which("optimum-cli")
    if not optimum:
        print(
            "[ERROR] optimum-cli not found. Install the export dependencies first:\n"
            "  pip install -r requirements-model.txt",
            file=sys.stderr,
        )
        return 1

    if os.path.exists(dest) and not force:
        print(f"[ERROR] Destination exists: {dest} (use --force to replace)", file=sys.stderr)
        return 1
    if os.path.exists(dest) and force:
        shutil.rmtree(dest)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    command = [
        optimum,
        "export",
        "openvino",
        "--model", "intfloat/multilingual-e5-base",
        "--task", "feature-extraction",
        dest,
    ]
    print("[INFO] Running:", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        return result.returncode
    return 0 if validate(dest) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the multilingual-e5-base OpenVINO model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="source", help="Existing OpenVINO model directory to copy")
    group.add_argument("--download", action="store_true", help="Download and export via optimum-cli")
    parser.add_argument("--dest", default=default_dest(), help="Destination directory")
    parser.add_argument("--force", action="store_true", help="Replace an existing destination")
    args = parser.parse_args()

    dest = os.path.abspath(os.path.expanduser(args.dest))
    print(f"[INFO] Destination: {dest}")

    if args.source:
        raise SystemExit(copy_model(args.source, dest, args.force))
    if args.download:
        raise SystemExit(download_model(dest, args.force))

    parser.print_help()
    print("\nUse --from to copy an existing model or --download to export a new one.")


if __name__ == "__main__":
    main()
