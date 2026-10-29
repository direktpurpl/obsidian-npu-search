#!/usr/bin/env python3
"""INT8 quantization for the Multilingual-E5-Base OpenVINO IR model.

Produces a drop-in replacement model directory (tokenizer + configs are
copied) that can be selected with ``--model-dir`` or picked up automatically
by ``scripts/npu_vault_search.py``.

Two variants:

  * ``static``      — full static INT8 (weights + activations) via
                      ``nncf.quantize()`` with a calibration set of vault chunks;
  * ``weight-only`` — ``nncf.compress_weights()``: INT8 weights, FP activations
                      (4x smaller model, usually no measurable quality loss).

Examples:
    PY=/home/direkt/.local/share/npu-search/venv/bin/python
    $PY scripts/quantize_int8.py --mode static
    $PY scripts/quantize_int8.py --mode weight-only --out-dir .../Multilingual-E5-Base-INT8-WO
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import random
import shutil
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
ENGINE_PATH = os.path.join(SCRIPT_DIR, "npu_vault_search.py")

MODEL_NAME = "Multilingual-E5-Base"
DEFAULT_VAULT = os.path.expanduser("~/Документы/learn")
COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "config.json",
)

def data_dir() -> str:
    base = os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share"))
    return os.path.join(base, "npu-search")


def load_engine():
    """Import the engine module to reuse its chunking logic and constants."""
    spec = importlib.util.spec_from_file_location("npu_vault_search", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load engine from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["npu_vault_search"] = module
    spec.loader.exec_module(module)
    return module


def resolve_fp_dir(custom: str) -> str:
    if custom:
        return os.path.expanduser(custom)
    return os.path.join(data_dir(), "models", MODEL_NAME)


def resolve_xml(model_dir: str) -> str:
    nested = os.path.join(model_dir, "openvino", "openvino_model.xml")
    if os.path.isfile(nested):
        return nested
    return os.path.join(model_dir, "openvino_model.xml")


def collect_texts(vault: str, engine, limit: int, seed: int = 13) -> list[str]:
    """Random vault notes -> section chunks, same slicing as the real index."""
    notes = glob.glob(os.path.join(vault, "**", "*.md"), recursive=True)
    if not notes:
        raise SystemExit(f"no markdown notes found in {vault}")
    rng = random.Random(seed)
    rng.shuffle(notes)

    texts: list[str] = []
    for path in notes:
        if len(texts) >= limit * 3:
            break
        rel = os.path.relpath(path, vault)
        try:
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        title = engine.note_title(rel)
        for chunk in engine.split_note_into_chunks(title, content):
            body = chunk["text"].strip()
            if len(body) < 40:
                continue
            texts.append(body)
            # E5 expects "query:" for queries — mix a few in for activation ranges.
            if rng.random() < 0.25:
                texts.append("query: " + (chunk["section"] or title))
    rng.shuffle(texts)
    return texts[:limit]


def make_dataset(texts: list[str], tokenizer, max_len: int, fixed_len: int | None = None):
    import nncf

    def transform(text):
        encoded = tokenizer.encode(text)
        length = min(len(encoded.ids), max_len)
        if fixed_len:
            length = fixed_len
        elif length < max_len:
            length = min(max_len, max(16, ((length + 31) // 32) * 32))
        ids = np.zeros((1, length), dtype=np.int64)
        mask = np.zeros((1, length), dtype=np.int64)
        keep = min(len(encoded.ids), length)
        ids[0, :keep] = encoded.ids[:keep]
        mask[0, :keep] = encoded.attention_mask[:keep]
        return {"input_ids": ids, "attention_mask": mask}

    return nncf.Dataset(texts, transform)


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8 quantization of the E5 OpenVINO model")
    parser.add_argument("--model-dir", default="", help="Source FP model directory")
    parser.add_argument("--out-dir", default="", help="Output directory for the INT8 model")
    parser.add_argument("--mode", choices=("static", "weight-only"), default="static")
    parser.add_argument("--vault-path", default=DEFAULT_VAULT, help="Vault used for calibration texts")
    parser.add_argument("--calib-count", type=int, default=300, help="Calibration samples (default 300)")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Tokenizer max length (default 512)")
    parser.add_argument("--subset-size", type=int, default=200, help="NNCF subset_size (default 200)")
    args = parser.parse_args()

    import nncf
    import openvino as ov
    from tokenizers import Tokenizer

    src_dir = resolve_fp_dir(args.model_dir)
    xml_path = resolve_xml(src_dir)
    if not os.path.isfile(xml_path):
        raise SystemExit(f"source model not found: {xml_path}")

    suffix = "INT8" if args.mode == "static" else "INT8-WO"
    out_dir = os.path.expanduser(args.out_dir) or os.path.join(data_dir(), "models", f"{MODEL_NAME}-{suffix}")

    tokenizer = Tokenizer.from_file(os.path.join(src_dir, "tokenizer.json"))
    tokenizer.enable_truncation(max_length=args.max_seq_len)

    engine = load_engine()
    texts = collect_texts(args.vault_path, engine, args.calib_count)
    print(f"[quantize] source: {src_dir}")
    print(f"[quantize] calibration samples: {len(texts)}")
    print(f"[quantize] mode: {args.mode}")

    core = ov.Core()
    model = core.read_model(xml_path)
    input_names = [inp.get_any_name() for inp in model.inputs]
    print(f"[quantize] inputs: {input_names}")

    if args.mode == "static":
        dataset = make_dataset(texts, tokenizer, args.max_seq_len)
        try:
            model = nncf.quantize(
                model,
                dataset,
                preset=nncf.QuantizationPreset.PERFORMANCE,
                subset_size=min(args.subset_size, len(texts)),
                fast_bias_correction=True,
            )
            note("STATIC_DYNAMIC")
        except Exception as exc:  # dynamic shapes rejected -> static quantize, then restore
            print(f"[quantize] dynamic-shape PTQ failed ({str(exc).splitlines()[0][:120]}), "
                  "retrying with static shape", file=sys.stderr)
            model = core.read_model(xml_path)
            model.reshape({name: [1, args.max_seq_len] for name in input_names})
            dataset = make_dataset(texts, tokenizer, args.max_seq_len, fixed_len=args.max_seq_len)
            model = nncf.quantize(
                model,
                dataset,
                preset=nncf.QuantizationPreset.PERFORMANCE,
                subset_size=min(args.subset_size, len(texts)),
                fast_bias_correction=True,
            )
            note("STATIC_SHAPE")
    else:
        model = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT8_ASYM,
            ignored_scope=None,
        )
        note("WEIGHT_ONLY")

    # Keep the runtime contract of the FP model: dynamic [-1, -1] inputs.
    try:
        shapes = {inp.get_any_name(): inp.get_partial_shape() for inp in model.inputs}
        if any(str(shape) != "[?,?]" for shape in shapes.values()):
            model.reshape({name: [-1, -1] for name in shapes})
            print("[quantize] reshaped back to dynamic [-1, -1]")
    except Exception as exc:
        print(f"[quantize] reshape to dynamic failed: {exc}", file=sys.stderr)

    os.makedirs(os.path.join(out_dir, "openvino"), exist_ok=True)
    out_xml = os.path.join(out_dir, "openvino", "openvino_model.xml")
    ov.save_model(model, out_xml, compress_to_fp16=False)
    for name in COPY_FILES:
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))

    print(f"[quantize] output: {out_dir}")
    print(f"[quantize] size FP: {dir_size_mb(src_dir):.0f} MB")
    print(f"[quantize] size INT8: {dir_size_mb(out_dir):.0f} MB")


def note(text: str) -> None:
    print(f"[quantize] variant: {text}")


if __name__ == "__main__":
    main()
