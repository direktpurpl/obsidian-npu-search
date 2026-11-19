# Intel NPU Semantic Search

**English** | [Русский](README.ru.md)

Semantic (vector) search for Obsidian notes. The plugin runs
[`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)
through **OpenVINO** and picks the accelerator automatically:
**Intel NPU → Intel GPU → CPU**.

CPU-only machines work too: the plugin falls back to CPU, builds the index once
and keeps it up to date incrementally.

> **Not in the Obsidian Community directory.** The plugin needs a Python runtime
> and a local OpenVINO model, which cannot be shipped inside `main.js`. Install
> it from source as described below.

---

## Features

- Search by meaning, not by exact keyword matches.
- Hardware acceleration on Intel NPU (AI Boost), with automatic fallback to GPU/CPU.
- Category filters: all notes, maps of content (MOC), projects, `dev/`, German.
- Incremental index: only new and changed notes are re-embedded.
- Reads the vault directly from disk (`--vault-path`) — no Obsidian REST API required.
- Standalone CLI for diagnostics and full reindexing.
- Settings inside Obsidian: Python, model, device, top-K.

---

## Requirements

- Obsidian Desktop (the plugin uses Node.js `child_process`).
- Python **3.10+**.
- Python packages: `openvino`, `tokenizers`, `numpy` (installed by `scripts/setup.sh`).
- The `Multilingual-E5-Base` model in OpenVINO IR format. The INT8 build
  (≈288 MB, NNCF weight-only quantization) is preferred automatically; the FP
  build (≈1.1 GB) is used as a fallback. Quantization: `scripts/quantize_int8.py`.
- For NPU: Meteor Lake / Lunar Lake with the `intel_vpu` driver. Intel GPU is
  used when present. CPU always works.

---

## Installation

Obsidian loads plugins only from `<vault>/.obsidian/plugins/<id>`, so clone the
repository straight into that directory.

### 1. Clone the plugin

```bash
PLUGIN="<vault>/.obsidian/plugins/npu-search"
git clone https://github.com/direktpurpl/obsidian-npu-search.git "$PLUGIN"
```

### 2. Create the Python environment

```bash
"$PLUGIN/scripts/setup.sh"
```

The script creates a venv **outside the repository**:

```text
~/.local/share/npu-search/venv
```

### 3. Install the model

Option A — copy an existing model from another machine:

```bash
~/.local/share/npu-search/venv/bin/python \
  "$PLUGIN/scripts/fetch_model.py" --from /path/to/Multilingual-E5-Base
```

Option B — download and export from HuggingFace:

```bash
~/.local/share/npu-search/venv/bin/python -m pip install -r "$PLUGIN/requirements-model.txt"
~/.local/share/npu-search/venv/bin/python "$PLUGIN/scripts/fetch_model.py" --download
```

By default the model is looked up in:

```text
~/.local/share/npu-search/models/Multilingual-E5-Base
```

Override it with `NPU_MODEL_DIR` or the plugin's "Model" setting.

### 4. Enable the plugin

`Obsidian → Settings → Community plugins → Intel NPU Semantic Search`.

Search: the brain icon in the left ribbon, `Ctrl+Alt+F` (search), `Ctrl+Alt+D`
(search with the German filter applied). All hotkeys work with any keyboard
layout and can be reassigned in `Settings → Hotkeys`. Or run `Ctrl+P` →
"Semantic search (NPU)".

---

## Plugin settings

| Setting | Default | Description |
| :-- | :-- | :-- |
| Python | auto | plugin venv → `~/.local/share/npu-search/venv` → `python3` |
| Model | `~/.local/share/npu-search/models/Multilingual-E5-Base` | Folder with `tokenizer.json` and OpenVINO IR |
| Device | `auto` | `NPU` → `GPU` → `CPU`, or forced |
| Obsidian REST API | `http://127.0.0.1:27123` | Only needed when the vault cannot be read from disk |
| Results | 10 | Number of result cards |
| Check | — | Environment diagnostics |
| Rebuild | — | Full reindex |

"Check" reports the Python version, OpenVINO version, available devices, the
selected device, model presence and the number of indexed notes.

---

## CLI

```bash
PY=~/.local/share/npu-search/venv/bin/python
SCRIPT="<vault>/.obsidian/plugins/npu-search/scripts/npu_vault_search.py"

# Environment diagnostics
$PY $SCRIPT --check

# Search
$PY $SCRIPT --query "docker deployment" --top-k 5

# Search only inside dev/
$PY $SCRIPT --query "systemd timer" --filter dev

# Full reindex
$PY $SCRIPT --build-index --vault-path /path/to/your/vault

# Force CPU (useful for debugging)
$PY $SCRIPT --query "backup" --device CPU
```

Environment variables:

| Variable | Purpose |
| :-- | :-- |
| `NPU_MODEL_DIR` | Model directory |
| `NPU_INDEX_FILE` | Path to the `.npz` index |
| `NPU_CACHE_DIR` | OpenVINO compilation cache |
| `NPU_DATA_DIR` | Base data directory (default `~/.local/share/npu-search`) |
| `OBSIDIAN_VAULT_PATH` | Vault root |
| `OBSIDIAN_API_KEY` | Local REST API key |

---

## Repository layout

```text
obsidian-npu-search/                       # git repository (code only)
├── main.js                                # Obsidian UI and commands
├── manifest.json
├── styles.css
├── requirements.txt
├── requirements-model.txt
├── eval/
│   ├── queries.json                       # 40-query evaluation set
│   └── results/                           # metrics (JSON)
└── scripts/
    ├── npu_vault_search.py                # search engine
    ├── quantize_int8.py                   # NNCF quantization
    ├── eval_search.py                     # Recall@k / MRR evaluation
    ├── fetch_model.py                     # copy or export the model
    └── setup.sh                           # venv + dependencies

~/.local/share/npu-search/                 # runtime data, outside git
├── venv/
├── models/Multilingual-E5-Base/           # FP model (fallback)
├── models/Multilingual-E5-Base-INT8/      # INT8 (288 MB, default)
├── indexes/<vault-hash>.npz               # vector index
├── indexes/<vault-hash>.fts.sqlite        # BM25 index (FTS5)
└── cache/openvino/
```

The index is keyed by the vault path hash, so several vaults do not conflict.
`models/`, `venv/`, `.cache/`, `*.npz` and `*.fts.sqlite` are git-ignored.

Build the INT8 model:

```bash
~/.local/share/npu-search/venv/bin/python "$PLUGIN/scripts/quantize_int8.py" --mode weight-only
```

Run the evaluation (40 queries, Recall@1/3/5, MRR):

```bash
~/.local/share/npu-search/venv/bin/python "$PLUGIN/scripts/eval_search.py" \
  --queries "$PLUGIN/eval/queries.json" --out eval/results/my.json \
  --vault-path /path/to/your/vault --label my --device auto --runs 3
```

---

## Performance

Measured with the INT8 model, batched indexing, on an index of
**207 notes / ~1050 sections**.

| Operation | Intel NPU | CPU only |
| :-- | --: | --: |
| Full search (query → results) | ~0.8 s | ~1.2 s |
| Model load per query | ~0.6–0.8 s | ~1.1 s |
| Query embedding | ~130 ms | ~90 ms |
| Index search (dot product) | < 1 ms | < 1 ms |
| Full reindex (batched) | ~90 s | ~10 min |
| Incremental update | seconds | tens of seconds |

Reference hardware: NPU column — Intel Core Ultra 7 155H (Meteor Lake);
CPU column — Intel Core i5-6300U (Skylake, AVX2).

Quality on the 40-query evaluation set:

| Metric | Score |
| :-- | --: |
| Recall@1 | 0.875 |
| Recall@3 | 1.000 |
| Recall@5 | 1.000 |
| MRR | 0.929 |

Search runs in a separate Python process (zero idle RAM): the model is loaded
for the duration of a query and then released. Changed notes are re-embedded in
the background, so search never waits for a rebuild.

---

## How it works

1. The plugin calls `npu_vault_search.py` with `--query` and `--json`.
2. The script loads the vector index (`.npz`) and the keyword index (`.fts.sqlite`).
3. The query is encoded by `multilingual-e5-base` (`query: ...`).
4. Dense results (cosine similarity) and BM25 results (SQLite FTS5) are fused
   with RRF; matches in the heading, path or tags get a boost, large "hub" notes
   are penalized. Results are returned to the UI.

Each note is split into sections by headings and every section is indexed
separately (up to 40 sections per note). That way search finds the relevant part
of a long note instead of just the first 1200 characters.

The index stores the note path, section title, a 768-dimensional vector, `mtime`
and file size. Changed notes are re-embedded automatically during search, and the
matching section is shown in the results.

---

## Troubleshooting

**`No module named openvino`**
Run `scripts/setup.sh`, or point the plugin's Python setting at the right interpreter.

**`OpenVINO model not found`**
The model is missing. Run `fetch_model.py --from ...` or `--download`, or set the
"Model" setting explicitly.

**NPU unavailable, CPU selected**
Expected behavior. Run `--check`: `NPU` must appear in the device list. On
machines without an NPU, CPU is the correct outcome.

**REST API 401 / connection refused**
The plugin reads the vault from disk anyway. REST is only needed for external
scenarios. The key is read from
`.obsidian/plugins/obsidian-local-rest-api/data.json`.

**Index is stale**
Run "Rebuild the NPU vector index" or `--build-index`. During normal search,
changes are picked up automatically by `mtime`.

---

## Roadmap

Planned improvements, measurements and rejected options: [ROADMAP.md](ROADMAP.md).

---

## License

MIT
