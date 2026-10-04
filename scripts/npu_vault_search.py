#!/usr/bin/env python3
"""
NPU-powered Ephemeral Semantic Search for Obsidian Vault.
Uses Intel AI Boost NPU via OpenVINO and Multilingual-E5-Base (768-dim).
Zero idle memory: starts on demand, performs vector similarity search, and exits immediately.
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import urllib.parse
import numpy as np

# Default connection settings
DEFAULT_REST_URL = os.environ.get("OBSIDIAN_REST_URL", "http://127.0.0.1:27123")


def resolve_api_key(args_key: str = "", vault_dir: str = "") -> str:
    """Resolve API key from args, env, or Obsidian Local REST API config."""
    if args_key:
        return args_key
    env_key = os.environ.get("OBSIDIAN_API_KEY", "")
    if env_key:
        return env_key

    # Auto-discover from local-rest-api data.json if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        vault_dir,
        os.environ.get("OBSIDIAN_VAULT_PATH", ""),
        # If script is in <vault>/.obsidian/plugins/npu-search/scripts:
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))),
    ]
    for v in candidates:
        if not v:
            continue
        cfg_path = os.path.join(v, ".obsidian", "plugins", "obsidian-local-rest-api", "data.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    key = cfg.get("apiKey", "")
                    if key:
                        return key
            except Exception:
                pass
    return ""


def resolve_model_dir(custom_path: str = "") -> str:
    """Locate OpenVINO Multilingual-E5-Base model directory."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    env_path = os.environ.get("NPU_MODEL_DIR", "")
    if env_path and os.path.exists(env_path):
        return env_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "models", "Multilingual-E5-Base"),
        os.path.join(os.path.dirname(script_dir), "models", "Multilingual-E5-Base"),
        "/home/direkt/Seafile/sync/projekts/Puthon/local_ai/models/Multilingual-E5-Base",
        "/home/direkt/Seafile/sync/projekts/Puthon/npu_micro_nn/models/Multilingual-E5-Base"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


class ObsidianClient:
    """Interacts with Obsidian Local REST API without touching the filesystem."""
    def __init__(self, base_url: str = DEFAULT_REST_URL, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(self, path: str, method: str = "GET", headers: dict = None) -> bytes:
        url = f"{self.base_url}/{path.lstrip('/')}"
        req_headers = {"Authorization": f"Bearer {self.api_key}"}
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(url, headers=req_headers, method=method)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read()

    def list_files_recursive(self, directory: str = "") -> list[str]:
        """Recursively list all markdown files in the vault via REST API."""
        encoded_dir = urllib.parse.quote(directory, safe="/")
        path = f"vault/{encoded_dir}" if encoded_dir else "vault/"
        try:
            data = json.loads(self._request(path).decode("utf-8"))
        except Exception:
            return []

        results = []
        for item in data.get("files", []):
            if item.endswith("/"):
                sub_folder_name = item.rstrip("/")
                if sub_folder_name in ("cache", "cashe", ".obsidian", ".trash", "templates", "Templates"):
                    continue
                full_sub_dir = f"{directory}/{sub_folder_name}" if directory else sub_folder_name
                results.extend(self.list_files_recursive(full_sub_dir))
            elif item.endswith(".md"):
                full_file_path = f"{directory}/{item}" if directory else item
                results.append(full_file_path)
        return results

    def get_file_content(self, filename: str) -> str:
        """Fetch file markdown content via REST API."""
        encoded = urllib.parse.quote(filename, safe="/")
        try:
            content = self._request(f"vault/{encoded}").decode("utf-8", errors="replace")
            return content
        except Exception:
            return ""


class NPUEmbedder:
    """Manages Intel NPU / OpenVINO inference for Multilingual-E5-Base."""
    def __init__(self, device: str = "NPU", model_dir: str = ""):
        self.device = device
        self.model_dir = resolve_model_dir(model_dir)
        self.cache_dir = os.path.join(os.path.dirname(self.model_dir), "model_cache")
        self._load_model()

    def _load_model(self):
        import openvino as ov
        from tokenizers import Tokenizer

        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_padding(length=512)
        self.tokenizer.enable_truncation(max_length=512)

        core = ov.Core()
        os.makedirs(self.cache_dir, exist_ok=True)
        core.set_property({"CACHE_DIR": self.cache_dir})

        xml_path = os.path.join(self.model_dir, "openvino", "openvino_model.xml")
        model = core.read_model(xml_path)
        self.input_names = [inp.get_any_name() for inp in model.inputs]
        reshape_dict = {name: [1, 512] for name in self.input_names}
        model.reshape(reshape_dict)

        try:
            self.compiled = core.compile_model(model, self.device)
        except Exception as e:
            sys.stderr.write(f"[WARNING] NPU compile failed ({e}), falling back to CPU\n")
            self.device = "CPU"
            self.compiled = core.compile_model(model, "CPU")

    @property
    def embedding_dim(self) -> int:
        return self.compiled.output(0).shape[-1]

    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text string into a normalized vector."""
        encoded = self.tokenizer.encode(text)
        inputs = {}
        for name in self.input_names:
            if name == "input_ids":
                inputs[name] = np.array([encoded.ids], dtype=np.int64)
            elif name == "attention_mask":
                inputs[name] = np.array([encoded.attention_mask], dtype=np.int64)
            elif name == "token_type_ids":
                inputs[name] = np.array([encoded.type_ids], dtype=np.int64)

        res = self.compiled(inputs)
        last_hidden_state = list(res.values())[0]

        mask = np.array([encoded.attention_mask], dtype=np.float32)[:, :, None]
        sum_embeddings = np.sum(last_hidden_state * mask, axis=1)
        sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
        embedding = sum_embeddings / sum_mask
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return (embedding / norm)[0]


def resolve_index_file(custom_path: str = "") -> str:
    if custom_path:
        return custom_path
    env_path = os.environ.get("NPU_INDEX_FILE", "")
    if env_path:
        return env_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "vault_embeddings.npz")


def build_or_update_index(client: ObsidianClient, embedder: NPUEmbedder, index_file: str, force: bool = False):
    """Scan vault markdown files and build/update vector index."""
    files = client.list_files_recursive()
    if not files:
        sys.stderr.write("[ERROR] No markdown files found via REST API.\n")
        return None

    cached_vectors = {}
    if not force and os.path.exists(index_file):
        try:
            data = np.load(index_file, allow_pickle=True)
            cached_files = data["files"]
            cached_vecs = data["vectors"]
            if cached_vecs.shape[1] == embedder.embedding_dim:
                for f, v in zip(cached_files, cached_vecs):
                    cached_vectors[f] = v
        except Exception:
            cached_vectors = {}

    to_embed = [f for f in files if f not in cached_vectors]
    if to_embed:
        sys.stderr.write(f"[INFO] Indexing {len(files)} vault notes with {embedder.device}...\n")
    else:
        sys.stderr.write(f"[INFO] All {len(files)} notes already indexed.\n")

    updated_files = []
    updated_vectors = []

    for f in files:
        if f in cached_vectors:
            updated_files.append(f)
            updated_vectors.append(cached_vectors[f])
        else:
            raw_content = client.get_file_content(f)
            lines = [l for l in raw_content.splitlines() if not l.startswith("---") and not l.startswith("tags:")]
            clean_text = "\n".join(lines).strip()
            content_snippet = clean_text[:1200] if clean_text else f.replace(".md", "")
            title = f.split("/")[-1].replace(".md", "")
            text_to_embed = f"passage: {title}\n{content_snippet}"

            vec = embedder.embed_text(text_to_embed)
            updated_files.append(f)
            updated_vectors.append(vec)

    vectors_arr = np.array(updated_vectors, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(index_file)), exist_ok=True)
    np.savez_compressed(index_file, files=np.array(updated_files), vectors=vectors_arr)
    sys.stderr.write(f"[INFO] Index saved ({len(updated_files)} notes, shape {vectors_arr.shape})\n")
    return updated_files, vectors_arr


def get_moc_referenced_notes(client: ObsidianClient) -> set[str]:
    import re
    content = ""
    try:
        content = client.get_file_content("📚 Карта знаний (MOC).md")
    except Exception:
        pass
    if not content:
        return set()
    return set(l.split("|")[0].strip() for l in re.findall(r"\[\[(.*?)\]\]", content))


def search_vault(
    query: str,
    top_k: int = 5,
    device: str = "NPU",
    auto_sync: bool = True,
    filter_type: str = "all",
    api_key: str = "",
    rest_url: str = DEFAULT_REST_URL,
    model_dir: str = "",
    index_file: str = ""
) -> list[dict]:
    api_key = resolve_api_key(api_key)
    client = ObsidianClient(base_url=rest_url, api_key=api_key)
    embedder = NPUEmbedder(device=device, model_dir=model_dir)
    idx_path = resolve_index_file(index_file)

    # Check if index exists and dimension matches active model
    expected_dim = embedder.embedding_dim
    rebuild_needed = not os.path.exists(idx_path)

    if not rebuild_needed:
        try:
            data = np.load(idx_path, allow_pickle=True)
            if data["vectors"].shape[1] != expected_dim:
                rebuild_needed = True
        except Exception:
            rebuild_needed = True

    if rebuild_needed:
        sys.stderr.write(f"[INFO] Rebuilding index for dimension {expected_dim}...\n")
        index_data = build_or_update_index(client, embedder, index_file=idx_path, force=True)
        if not index_data:
            return []
        files, vectors = index_data
    else:
        try:
            data = np.load(idx_path, allow_pickle=True)
            files = list(data["files"])
            vectors = data["vectors"]

            if auto_sync:
                live_files = client.list_files_recursive()
                if set(files) != set(live_files):
                    sys.stderr.write("[INFO] Vault changes detected, auto-syncing vector index...\n")
                    index_data = build_or_update_index(client, embedder, index_file=idx_path, force=False)
                    if index_data:
                        files, vectors = index_data
        except Exception:
            index_data = build_or_update_index(client, embedder, index_file=idx_path, force=True)
            files, vectors = index_data

    # Filter candidate indices
    filter_norm = (filter_type or "all").lower().strip()
    if filter_norm in ("moc", "карта", "карта знаний", "map"):
        moc_links = get_moc_referenced_notes(client)
        candidate_indices = []
        for idx, f in enumerate(files):
            base = f.split("/")[-1].replace(".md", "")
            if (
                "Карта знаний" in f or 
                "MOC" in f or 
                "Карта проектов" in f or 
                base in moc_links or
                f == "📚 Карта знаний (MOC).md"
            ):
                candidate_indices.append(idx)
    elif filter_norm in ("projects", "project", "проекты", "проект"):
        candidate_indices = [idx for idx, f in enumerate(files) if "Projects" in f or "Карта проектов" in f]
    elif filter_norm in ("dev", "linux", "programming"):
        candidate_indices = [idx for idx, f in enumerate(files) if f.startswith("dev/")]
    elif filter_norm in ("deutsch", "german", "немецкий", "словарь"):
        candidate_indices = [idx for idx, f in enumerate(files) if f.startswith("Deutsch/")]
    else:
        candidate_indices = list(range(len(files)))

    if not candidate_indices:
        return []

    # E5 model requires 'query: ' prefix for search queries
    query_text = f"query: {query.strip()}"
    t0 = time.time()
    query_vec = embedder.embed_text(query_text)
    infer_time = time.time() - t0

    # Cosine similarity for filtered candidates
    candidate_vectors = vectors[candidate_indices]
    candidate_scores = np.dot(candidate_vectors, query_vec)

    top_k_sub = min(top_k, len(candidate_indices))
    top_sub_indices = np.argsort(candidate_scores)[::-1][:top_k_sub]

    results = []
    for rank, sub_idx in enumerate(top_sub_indices):
        orig_idx = candidate_indices[sub_idx]
        results.append({
            "rank": rank + 1,
            "filename": str(files[orig_idx]),
            "score": float(round(float(candidate_scores[sub_idx]), 4)),
            "device": embedder.device,
            "infer_time_ms": round(infer_time * 1000, 2),
            "filter": filter_norm
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Ephemeral NPU Semantic Search for Obsidian")
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--filter", "-f", type=str, default="all", help="Category filter: all, moc, projects, dev, deutsch")
    parser.add_argument("--api-key", type=str, default="", help="Obsidian REST API key")
    parser.add_argument("--rest-url", type=str, default=DEFAULT_REST_URL, help="Obsidian REST API base URL")
    parser.add_argument("--model-dir", type=str, default="", help="Path to Multilingual-E5-Base directory")
    parser.add_argument("--index-file", type=str, default="", help="Path to npz index file")
    parser.add_argument("--build-index", "--reindex", action="store_true", help="Rebuild embedding index")
    parser.add_argument("--device", "-d", type=str, default="NPU", choices=["NPU", "GPU", "CPU"], help="Inference device")
    parser.add_argument("--json", action="store_true", default=True, help="Output results as JSON")

    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)

    if args.build_index:
        client = ObsidianClient(base_url=args.rest_url, api_key=api_key)
        embedder = NPUEmbedder(device=args.device, model_dir=args.model_dir)
        idx_path = resolve_index_file(args.index_file)
        build_or_update_index(client, embedder, index_file=idx_path, force=True)
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
        index_file=args.index_file
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
