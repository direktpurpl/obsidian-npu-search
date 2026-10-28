#!/usr/bin/env python3
"""
Standalone semantic search engine for Obsidian.

Runs intfloat/multilingual-e5-base (768-dim) through OpenVINO on:
  * Intel NPU (AI Boost)  -> preferred when available
  * Intel GPU             -> fallback
  * CPU                   -> always available

The engine has no dependency on any external project: the model is looked up
in the XDG data directory (or NPU_MODEL_DIR / --model-dir), the vector index
lives next to this script, and the vault can be read either through the
Obsidian Local REST API or directly from disk (--vault-path).

Examples:
    python npu_vault_search.py --query "docker deployment" --top-k 5
    python npu_vault_search.py --query "backup" --filter dev
    python npu_vault_search.py --build-index --vault-path ~/vault
    python npu_vault_search.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_REST_URL = os.environ.get("OBSIDIAN_REST_URL", "http://127.0.0.1:27123")
INDEX_NAME = "vault_embeddings.npz"
MODEL_NAME = "Multilingual-E5-Base"
SNIPPET_CHARS = 1200
MAX_SEQ_LEN = 512
EXCLUDED_DIRS = {
    ".git",
    ".obsidian",
    ".stversions",
    ".trash",
    "cache",
    "cashe",
    "node_modules",
    "templates",
}


def data_dir() -> str:
    base = os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share"))
    return os.path.join(base, "npu-search")


def resolve_model_xml(model_dir: str) -> str:
    """Return the OpenVINO XML path (optimum-cli may export flat or in openvino/)."""
    nested = os.path.join(model_dir, "openvino", "openvino_model.xml")
    if os.path.isfile(nested):
        return nested
    flat = os.path.join(model_dir, "openvino_model.xml")
    if os.path.isfile(flat):
        return flat
    return nested


def resolve_model_dir(custom_path: str = "") -> str:
    """Locate the OpenVINO Multilingual-E5-Base directory."""
    candidates = []
    if custom_path:
        candidates.append(os.path.expanduser(custom_path))
    env_path = os.environ.get("NPU_MODEL_DIR", "")
    if env_path:
        candidates.append(os.path.expanduser(env_path))

    default = os.path.join(data_dir(), "models", MODEL_NAME)
    candidates.extend([
        default,
        os.path.join(PLUGIN_DIR, "models", MODEL_NAME),
    ])

    for candidate in candidates:
        tokenizer = os.path.join(candidate, "tokenizer.json")
        if os.path.isfile(resolve_model_xml(candidate)) and os.path.isfile(tokenizer):
            return candidate
    return candidates[0]


def resolve_index_file(custom_path: str = "", vault_path: str = "") -> str:
    if custom_path:
        return os.path.expanduser(custom_path)
    env_path = os.environ.get("NPU_INDEX_FILE", "")
    if env_path:
        return os.path.expanduser(env_path)

    vault = resolve_vault_path(vault_path)
    key = hashlib.sha1(vault.encode("utf-8")).hexdigest()[:12] if vault else "default"
    return os.path.join(data_dir(), "indexes", f"{key}.npz")


def resolve_cache_dir() -> str:
    env_path = os.environ.get("NPU_CACHE_DIR", "")
    if env_path:
        return os.path.expanduser(env_path)
    return os.path.join(data_dir(), "cache", "openvino")


def resolve_vault_path(custom_path: str = "") -> str:
    """Vault root used for direct filesystem access (optional)."""
    if custom_path:
        return os.path.abspath(os.path.expanduser(custom_path))
    env_path = os.environ.get("OBSIDIAN_VAULT_PATH", "")
    if env_path:
        return os.path.abspath(os.path.expanduser(env_path))

    # <vault>/.obsidian/plugins/npu-search -> <vault>
    candidate = os.path.dirname(os.path.dirname(os.path.dirname(PLUGIN_DIR)))
    if os.path.isdir(os.path.join(candidate, ".obsidian")):
        return candidate

    # Running from a vault directory is also a valid hint.
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, ".obsidian")):
        return cwd
    return ""


def resolve_api_key(args_key: str = "", vault_dir: str = "") -> str:
    """Resolve API key from args, env or the Local REST API plugin config."""
    if args_key:
        return args_key
    env_key = os.environ.get("OBSIDIAN_API_KEY", "")
    if env_key:
        return env_key

    candidates = [vault_dir, os.environ.get("OBSIDIAN_VAULT_PATH", ""), resolve_vault_path()]
    for vault in candidates:
        if not vault:
            continue
        cfg_path = os.path.join(vault, ".obsidian", "plugins", "obsidian-local-rest-api", "data.json")
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                key = json.load(fh).get("apiKey", "")
                if key:
                    return key
        except Exception:
            pass
    return ""


class ObsidianClient:
    """Minimal Obsidian Local REST API client."""

    def __init__(self, base_url: str = DEFAULT_REST_URL, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(self, path: str, method: str = "GET", headers: dict | None = None) -> bytes:
        url = f"{self.base_url}/{path.lstrip('/')}"
        req_headers = {"Authorization": f"Bearer {self.api_key}"}
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(url, headers=req_headers, method=method)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read()

    def list_files_recursive(self, directory: str = "") -> list[str]:
        encoded_dir = urllib.parse.quote(directory, safe="/")
        path = f"vault/{encoded_dir}" if encoded_dir else "vault/"
        try:
            data = json.loads(self._request(path).decode("utf-8"))
        except Exception:
            return []

        results = []
        for item in data.get("files", []):
            if item.endswith("/"):
                sub = item.rstrip("/")
                if sub.lower() in EXCLUDED_DIRS or sub.startswith("."):
                    continue
                results.extend(self.list_files_recursive(f"{directory}/{sub}" if directory else sub))
            elif item.endswith(".md"):
                results.append(f"{directory}/{item}" if directory else item)
        return results

    def get_file_content(self, filename: str) -> str:
        encoded = urllib.parse.quote(filename, safe="/")
        try:
            return self._request(f"vault/{encoded}").decode("utf-8", errors="replace")
        except Exception:
            return ""


def scan_vault(vault_path: str) -> list[dict]:
    """Recursively list markdown notes with mtime/size (fast, no REST)."""
    root = os.path.abspath(os.path.expanduser(vault_path))
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d.lower() not in EXCLUDED_DIRS and not d.startswith(".")
        )
        for name in sorted(filenames):
            if not name.lower().endswith(".md"):
                continue
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            entries.append({"path": rel, "mtime_ns": st.st_mtime_ns, "size": st.st_size})
    return entries


def read_note(client: ObsidianClient, vault_path: str, rel_path: str) -> str:
    if vault_path:
        try:
            with open(os.path.join(vault_path, rel_path), "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception:
            return ""
    return client.get_file_content(rel_path)


def clean_markdown(text: str) -> str:
    text = text.lstrip("\ufeff")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    return text.strip()


class Embedder:
    """OpenVINO inference with automatic NPU -> GPU -> CPU fallback."""

    def __init__(self, device: str = "auto", model_dir: str = ""):
        self.requested_device = (device or "auto").strip() or "auto"
        self.model_dir = resolve_model_dir(model_dir)
        self.device = "CPU"
        self.static_shape = False
        self.load_time_ms = 0.0
        self.available_devices: list[str] = []
        self._load_model()

    def _device_candidates(self) -> list[str]:
        if self.requested_device.lower() != "auto":
            requested = self.requested_device
            upper = requested.upper()
            if upper == "CPU":
                return ["CPU"]
            # Prefer the explicitly requested device, but still fall back to CPU.
            return [requested, "CPU"]

        available = self.available_devices
        candidates = []
        for preferred in ("NPU", "GPU", "CPU"):
            matches = [d for d in available if d.upper() == preferred or d.upper().startswith(preferred + ".")]
            if matches:
                candidates.extend(matches)
            elif preferred == "CPU":
                candidates.append("CPU")
        return candidates or ["CPU"]

    def _load_model(self) -> None:
        started = time.time()
        import openvino as ov
        from tokenizers import Tokenizer

        xml_path = resolve_model_xml(self.model_dir)
        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(
                f"OpenVINO model not found: {xml_path}. "
                "Run scripts/fetch_model.py or pass --model-dir."
            )
        if not os.path.isfile(tok_path):
            raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)

        core = ov.Core()
        cache_dir = resolve_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        try:
            core.set_property({"CACHE_DIR": cache_dir})
        except Exception:
            pass

        try:
            self.available_devices = list(core.get_available_devices())
        except Exception:
            self.available_devices = []

        errors = []
        for candidate in self._device_candidates():
            try:
                if candidate.upper().startswith("NPU"):
                    model = core.read_model(xml_path)
                    names = [inp.get_any_name() for inp in model.inputs]
                    model.reshape({name: [1, MAX_SEQ_LEN] for name in names})
                    compiled = core.compile_model(model, candidate)
                    static = True
                else:
                    model = core.read_model(xml_path)
                    compiled = core.compile_model(model, candidate)
                    static = False
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue

            self.compiled = compiled
            self.device = candidate
            self.static_shape = static
            self.input_names = [inp.get_any_name() for inp in compiled.inputs]
            if static:
                self.tokenizer.enable_padding(length=MAX_SEQ_LEN)
            else:
                self.tokenizer.enable_padding()
            self.load_time_ms = round((time.time() - started) * 1000, 2)
            if candidate != self.requested_device and self.requested_device.lower() != "auto":
                sys.stderr.write(
                    f"[WARNING] Device '{self.requested_device}' failed, using '{candidate}'\n"
                )
            return

        raise RuntimeError("Could not compile the model: " + " | ".join(errors))

    @property
    def embedding_dim(self) -> int:
        shape = self.compiled.output(0).get_partial_shape()
        return int(shape[-1].get_length())

    def embed_text(self, text: str) -> np.ndarray:
        encoded = self.tokenizer.encode(text)
        inputs = {}
        for name in self.input_names:
            if name == "input_ids":
                inputs[name] = np.array([encoded.ids], dtype=np.int64)
            elif name == "attention_mask":
                inputs[name] = np.array([encoded.attention_mask], dtype=np.int64)
            elif name == "token_type_ids":
                inputs[name] = np.array([encoded.type_ids], dtype=np.int64)

        result = self.compiled(inputs)
        last_hidden_state = result[self.compiled.output(0)]

        if last_hidden_state.ndim == 2:
            embedding = last_hidden_state[0]
        else:
            mask = np.array([encoded.attention_mask], dtype=np.float32)[:, :, None]
            summed = np.sum(last_hidden_state * mask, axis=1)
            counts = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
            embedding = (summed / counts)[0]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)


def load_index(index_file: str):
    data = np.load(index_file, allow_pickle=True)
    files = [str(f) for f in data["files"]]
    vectors = data["vectors"].astype(np.float32)
    mtimes = data["mtimes"].astype(np.int64) if "mtimes" in data else None
    sizes = data["sizes"].astype(np.int64) if "sizes" in data else None
    return files, vectors, mtimes, sizes


def build_or_update_index(
    client: ObsidianClient,
    embedder: Embedder,
    index_file: str,
    vault_path: str = "",
    force: bool = False,
):
    """Build or incrementally update the vector index."""
    started = time.time()
    if vault_path:
        entries = scan_vault(vault_path)
    else:
        entries = [
            {"path": name, "mtime_ns": None, "size": None}
            for name in client.list_files_recursive()
        ]

    if not entries:
        sys.stderr.write("[ERROR] No markdown files found.\n")
        return None

    cached: dict[str, tuple[np.ndarray, int | None, int | None]] = {}
    if not force and os.path.exists(index_file):
        try:
            old_files, old_vectors, old_mtimes, old_sizes = load_index(index_file)
            if old_vectors.shape[1] == embedder.embedding_dim:
                for i, name in enumerate(old_files):
                    cached[name] = (
                        old_vectors[i],
                        int(old_mtimes[i]) if old_mtimes is not None else None,
                        int(old_sizes[i]) if old_sizes is not None else None,
                    )
                sys.stderr.write(f"[INFO] Loaded {len(cached)} cached embeddings.\n")
        except Exception as exc:
            sys.stderr.write(f"[INFO] Index cache ignored: {exc}\n")

    files: list[str] = []
    vectors: list[np.ndarray] = []
    mtimes: list[int] = []
    sizes: list[int] = []
    embedded = 0

    for number, entry in enumerate(entries, start=1):
        rel_path = entry["path"]
        mtime = entry["mtime_ns"]
        size = entry["size"]

        cached_item = cached.get(rel_path)
        unchanged = False
        if cached_item is not None:
            cached_vector, cached_mtime, cached_size = cached_item
            # If either side has no mtime (old index / REST mode), keep the vector.
            if cached_mtime is None or mtime is None:
                unchanged = True
            elif cached_mtime == mtime and cached_size == size:
                unchanged = True

        if unchanged:
            vector = cached_item[0]
        else:
            raw = read_note(client, vault_path, rel_path)
            title = os.path.basename(rel_path)[:-3] if rel_path.lower().endswith(".md") else os.path.basename(rel_path)
            content = clean_markdown(raw)
            snippet = content[:SNIPPET_CHARS] if content else title
            vector = embedder.embed_text(f"passage: {title}\n{snippet}")
            embedded += 1
            if embedded == 1 or embedded % 25 == 0:
                sys.stderr.write(
                    f"[INFO] Embedded {embedded} note(s) on {embedder.device} "
                    f"({number}/{len(entries)})\n"
                )

        files.append(rel_path)
        vectors.append(vector)
        mtimes.append(int(mtime) if mtime is not None else -1)
        sizes.append(int(size) if size is not None else -1)

    vectors_arr = np.array(vectors, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(index_file)), exist_ok=True)
    np.savez_compressed(
        index_file,
        files=np.array(files),
        vectors=vectors_arr,
        mtimes=np.array(mtimes, dtype=np.int64),
        sizes=np.array(sizes, dtype=np.int64),
    )
    elapsed = time.time() - started
    sys.stderr.write(
        f"[INFO] Index saved: {len(files)} notes, {embedded} embedded, "
        f"shape {vectors_arr.shape}, {elapsed:.1f}s -> {index_file}\n"
    )
    return files, vectors_arr, np.array(mtimes, dtype=np.int64), np.array(sizes, dtype=np.int64)


def index_needs_update(
    client: ObsidianClient,
    index_file: str,
    vault_path: str = "",
    files: list[str] | None = None,
    mtimes=None,
    sizes=None,
) -> bool:
    if not files:
        return True
    if vault_path:
        live = scan_vault(vault_path)
        if set(files) != {entry["path"] for entry in live}:
            return True
        if mtimes is None:
            return False
        stored = {
            name: (int(mtimes[i]), int(sizes[i]) if sizes is not None else -1)
            for i, name in enumerate(files)
        }
        for entry in live:
            if stored.get(entry["path"]) != (entry["mtime_ns"], entry["size"]):
                return True
        return False

    live_files = set(client.list_files_recursive())
    return live_files != set(files)


def get_moc_referenced_notes(client: ObsidianClient, vault_path: str = "") -> set[str]:
    content = read_note(client, vault_path, "📚 Карта знаний (MOC).md")
    if not content:
        return set()
    return set(match.split("|")[0].strip() for match in re.findall(r"\[\[(.*?)\]\]", content))


def search_vault(
    query: str,
    top_k: int = 5,
    device: str = "auto",
    auto_sync: bool = True,
    filter_type: str = "all",
    api_key: str = "",
    rest_url: str = DEFAULT_REST_URL,
    model_dir: str = "",
    index_file: str = "",
    vault_path: str = "",
) -> list[dict]:
    started = time.time()
    vault_path = resolve_vault_path(vault_path)
    api_key = resolve_api_key(api_key, vault_path)
    client = ObsidianClient(base_url=rest_url, api_key=api_key)
    embedder = Embedder(device=device, model_dir=model_dir)
    index_path = resolve_index_file(index_file, vault_path)

    expected_dim = embedder.embedding_dim
    rebuild_needed = not os.path.exists(index_path)

    if not rebuild_needed:
        try:
            files, vectors, mtimes, sizes = load_index(index_path)
            if vectors.shape[1] != expected_dim:
                rebuild_needed = True
        except Exception:
            rebuild_needed = True

    if rebuild_needed:
        sys.stderr.write(f"[INFO] Building index for dimension {expected_dim}...\n")
        index_data = build_or_update_index(
            client, embedder, index_file=index_path, vault_path=vault_path, force=False
        )
        if not index_data:
            return []
        files, vectors, mtimes, sizes = index_data
    else:
        try:
            if auto_sync and index_needs_update(
                client, index_path, vault_path=vault_path,
                files=files, mtimes=mtimes, sizes=sizes,
            ):
                sys.stderr.write("[INFO] Vault changes detected, updating index...\n")
                index_data = build_or_update_index(
                    client, embedder, index_file=index_path, vault_path=vault_path, force=False
                )
                if index_data:
                    files, vectors, mtimes, sizes = index_data
        except Exception as exc:
            sys.stderr.write(f"[WARNING] Index auto-sync failed: {exc}\n")

    filter_norm = (filter_type or "all").lower().strip()
    if filter_norm in ("moc", "карта", "карта знаний", "map"):
        moc_links = get_moc_referenced_notes(client, vault_path)
        candidate_indices = []
        for idx, name in enumerate(files):
            base = name.split("/")[-1].replace(".md", "")
            if (
                "Карта знаний" in name
                or "MOC" in name
                or "Карта проектов" in name
                or base in moc_links
                or name == "📚 Карта знаний (MOC).md"
            ):
                candidate_indices.append(idx)
    elif filter_norm in ("projects", "project", "проекты", "проект"):
        candidate_indices = [
            idx for idx, name in enumerate(files)
            if "Projects" in name or "Карта проектов" in name
        ]
    elif filter_norm in ("dev", "linux", "programming"):
        candidate_indices = [idx for idx, name in enumerate(files) if name.startswith("dev/")]
    elif filter_norm in ("deutsch", "german", "немецкий", "словарь"):
        candidate_indices = [idx for idx, name in enumerate(files) if name.startswith("Deutsch/")]
    else:
        candidate_indices = list(range(len(files)))

    if not candidate_indices:
        return []

    query_started = time.time()
    query_vec = embedder.embed_text(f"query: {query.strip()}")
    infer_time = time.time() - query_started

    candidate_vectors = vectors[candidate_indices]
    candidate_scores = np.dot(candidate_vectors, query_vec)

    top_k_sub = min(top_k, len(candidate_indices))
    top_sub_indices = np.argsort(candidate_scores)[::-1][:top_k_sub]

    total_ms = round((time.time() - started) * 1000, 1)
    results = []
    for rank, sub_idx in enumerate(top_sub_indices):
        orig_idx = candidate_indices[sub_idx]
        results.append({
            "rank": rank + 1,
            "filename": str(files[orig_idx]),
            "score": float(round(float(candidate_scores[sub_idx]), 4)),
            "device": embedder.device,
            "infer_time_ms": round(infer_time * 1000, 2),
            "load_time_ms": embedder.load_time_ms,
            "total_ms": total_ms,
            "filter": filter_norm,
        })
    return results


def check_environment(args) -> dict:
    info = {
        "python": sys.version.split()[0],
        "script": os.path.realpath(__file__),
        "plugin_dir": PLUGIN_DIR,
        "model_dir": resolve_model_dir(args.model_dir),
        "index_file": resolve_index_file(args.index_file, args.vault_path),
        "vault_path": resolve_vault_path(args.vault_path),
        "openvino": None,
        "devices": [],
        "selected_device": None,
        "model_ok": False,
        "index_notes": 0,
        "index_dim": 0,
    }
    model_dir = info["model_dir"]
    info["model_ok"] = (
        os.path.isfile(resolve_model_xml(model_dir))
        and os.path.isfile(os.path.join(model_dir, "tokenizer.json"))
    )

    try:
        import openvino as ov
        info["openvino"] = getattr(ov, "__version__", "unknown")
        info["devices"] = list(ov.Core().get_available_devices())
    except Exception as exc:
        info["openvino_error"] = str(exc)

    try:
        import tokenizers
        info["tokenizers"] = getattr(tokenizers, "__version__", "unknown")
    except Exception as exc:
        info["tokenizers_error"] = str(exc)

    try:
        files, vectors, _, _ = load_index(info["index_file"])
        info["index_notes"] = len(files)
        info["index_dim"] = int(vectors.shape[1])
    except Exception:
        pass

    if info["model_ok"] and info["openvino"]:
        try:
            embedder = Embedder(device=args.device, model_dir=args.model_dir)
            info["selected_device"] = embedder.device
            info["load_time_ms"] = embedder.load_time_ms
        except Exception as exc:
            info["compile_error"] = str(exc)

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone semantic search for Obsidian")
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    parser.add_argument("--filter", "-f", type=str, default="all",
                        help="Category filter: all, moc, projects, dev, deutsch")
    parser.add_argument("--api-key", type=str, default="", help="Obsidian REST API key")
    parser.add_argument("--rest-url", type=str, default=DEFAULT_REST_URL, help="Obsidian REST API URL")
    parser.add_argument("--model-dir", type=str, default="", help="Path to Multilingual-E5-Base")
    parser.add_argument("--index-file", type=str, default="", help="Path to the npz index")
    parser.add_argument("--vault-path", type=str, default="", help="Vault root for direct file access")
    parser.add_argument("--build-index", "--reindex", action="store_true", help="Rebuild the index")
    parser.add_argument("--check", action="store_true", help="Print environment diagnostics as JSON")
    parser.add_argument("--device", "-d", type=str, default="auto",
                        choices=["auto", "NPU", "GPU", "CPU"], help="Inference device")
    parser.add_argument("--json", action="store_true", default=True, help="Output JSON (default)")

    args = parser.parse_args()

    if args.check:
        print(json.dumps(check_environment(args), ensure_ascii=False, indent=2))
        return

    vault_path = resolve_vault_path(args.vault_path)
    api_key = resolve_api_key(args.api_key, vault_path)

    if args.build_index:
        client = ObsidianClient(base_url=args.rest_url, api_key=api_key)
        embedder = Embedder(device=args.device, model_dir=args.model_dir)
        index_path = resolve_index_file(args.index_file, vault_path)
        build_or_update_index(
            client, embedder, index_file=index_path, vault_path=vault_path, force=True
        )
        if not args.query:
            return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    results = search_vault(
        args.query,
        top_k=args.top_k,
        device=args.device,
        filter_type=args.filter,
        api_key=api_key,
        rest_url=args.rest_url,
        model_dir=args.model_dir,
        index_file=args.index_file,
        vault_path=vault_path,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
