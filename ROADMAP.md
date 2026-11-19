# Roadmap: fast and accurate search

**English** | [Русский](ROADMAP.ru.md)

Improvement plan for the NPU search plugin. Nothing here is guaranteed — these
are options and the order of work. Accepted items are checked off and moved into
development.

This file is the single source of the plan.

---

## Goals

1. **Response time.** Full search (keystroke → results):
   - CPU-only machines: **≤ 1 s** (currently ~1.9–2.2 s);
   - NPU machines: **≤ 0.5 s** (currently ~1.1 s).
2. **Result quality.** Exact terms (commands, IPs, MACs, model names), the right
   note in the top 3, the right section at rank 1.
3. **No waiting.** Rebuilding or updating the index never blocks search.

## Constraints (decisions)

- **The model is not kept in RAM permanently.** It is loaded only for the
  duration of a query and released afterwards. Resident daemon processes are
  **rejected**.
- Same behavior on NPU and CPU-only machines without manual reconfiguration.
- Runtime data (venv, model, index, cache) stays in `~/.local/share/npu-search`.

## Baseline measurements

| Metric | NPU | CPU only |
| :-- | --: | --: |
| Model load per query | ~0.86 s | ~1.2–1.6 s |
| Query embedding | ~130 ms (NPU) / ~75 ms (CPU) | ~90 ms |
| Index search (dot product) | < 1 ms | < 1 ms |
| Full response | ~1.1 s | ~1.9–2.2 s |
| Full reindex (207 notes / ~1050 sections) | ~63 s | ~4–5 min |

**Conclusion:** almost all of the time is model loading, not inference. The main
levers for speed are a smaller model (faster load) and fewer processed tokens.
Searching ~1000 sections is already instantaneous.

**Current result (hybrid + INT8):** NPU — R@1 **0.875** / R@3 **1.000** /
R@5 **1.000** / MRR **0.9292**, latency **~0.82 s** (down from ~1.0 s on FP).
Quality is well above the original baseline (R@1 0.825 / MRR 0.864). Remaining:
roll INT8 out to CPU-only machines and optional items (debounce, query cache).

---

## Phase 0 — Measurability (do this first)

Without it, any improvement is judged by feel.

- [x] **Evaluation set** — 40 queries with expected notes/sections (exact terms,
  semantic, mixed, natural phrasing, German).
- [x] **Evaluation script** `scripts/eval_search.py` — Recall@1/3/5, MRR, mean
  response time; run on both machine classes, results to JSON.
- [x] **Baseline** — recorded: R@1 0.825 / R@3 0.900 / R@5 0.950 / MRR 0.8642.

## Phase 1 — Speed (without a resident model)

Ordered by expected impact.

- [x] **INT8 quantization** — done: NNCF weight-only INT8 (`compress_weights`),
  model 1080 → 288 MB. Full static INT8 failed quality (MRR 0.898) and was
  discarded. Result: latency 1001 → 817 ms on NPU, quality R@1 0.875 /
  MRR 0.9292 (one fluctuating query out of 40). Auto-selection: the INT8 model is
  preferred over FP when present in `models/Multilingual-E5-Base-INT8`.
- [x] **Batched indexing** (batches of 16 sections) — done: full rebuild 63 → 32 s
  on NPU.
- [x] **Background index updates** — done: search does not wait for reindexing;
  one-shot background process, lock file, atomic npz write.
- [x] **Re-measure NPU / GPU / CPU** — done: CPU is slightly faster by latency
  (492 vs 817 ms), but NPU is kept by the owner's decision to avoid heating the
  CPU. CPU-only machines use CPU anyway.
- [ ] **Debounce 250 ms** in the UI for search-as-you-type (if live input is enabled).
- [ ] **LRU query cache** (optional) — repeated queries return instantly.
- [ ] **OS cache warm-up** (optional) — a one-shot run at Obsidian startup warms
  the model files in the page cache; the model is still released after a query.

> ANN/HNSW/FAISS are not needed: at ~1000 sections brute force is faster and simpler.

## Phase 2 — Quality

- [x] **Hybrid dense + BM25** — keyword index (SQLite FTS5) over sections plus
  the vector index, fused with RRF (k=20). Done: R@1 0.825 → 0.900,
  MRR 0.864 → 0.946.
- [x] **Title / heading / tag boost** — query tokens matching the section
  heading, note path or tags promote the result.
- [x] **Breadcrumbs** — the heading chain is prepended to the section text
  (Anthropic-style).
- [x] **Anti-hub penalty** — demotes large "hub" notes (MOCs etc.),
  beta = 0.1, tuned by sweeping 12 configs on the evaluation set.
- [x] **Section chunking by headings** — partial token-aware chunking is in place:
  sections are split by headings with breadcrumbs.
- [ ] **Paragraph chunking with overlap** — split long sections by paragraphs or
  tokens with 15–20% overlap instead of a 1000-character cut. Better coverage of
  long sections.
- [ ] **Exclude service notes** — `README.md`, templates and empty notes are
  demoted or excluded from results.
- [ ] **MMR / diversity** — avoid returning several near-duplicate notes.
- [ ] **Reranker top-10/20** (optional, behind a flag) — a cross-encoder reorders
  candidates. +0.3–1 s on CPU, which contradicts the speed goal, so only as a
  separate "refine results" mode or on NPU.
- [ ] **Model change** (optional):
  - `multilingual-e5-small` — 2–3× faster than base, lower quality; a compromise
    for CPU-only machines if INT8 is not fast enough;
  - `multilingual-e5-large` / `bge-m3` — higher quality, but 2.2 GB and slow;
    only if the evaluation shows a quality gap.
- [ ] **Query expansion / HyDE** (optional) — rewriting the query with a local
  LLM. +0.5–3 s, complex, not always helpful — the last item.

## Rejected

- **Resident daemon with the model in RAM** — rejected for memory reasons; the
  model is loaded only for the duration of a query.
- **ANN/HNSW/FAISS** — overkill at the current index size.
- **Max length 256 tokens** — rejected by measurement: saves ~10 ms per query
  (model loading dominates) but costs −5 pp R@1 (Russian sections are ~300–350
  tokens, truncation cuts content). 512 is kept.

---

## Build presets

- **Minimal** — Phase 0 + BM25 hybrid + heading boost. Speed as today, noticeably
  better quality.
- **Balanced (recommended)** — Phase 0 + INT8 + batched indexing + background
  updates + BM25 hybrid + boost + breadcrumbs.
- **Maximum quality** — balanced + paragraph chunking + MMR + optional reranker.

## Implementation order

1. Phase 0 (measurability) — mandatory first.
2. INT8 quantization (speed).
3. Hybrid dense + BM25 + heading boost (quality).
4. Batched indexing + background updates (convenience, removes waiting).
5. Everything else — based on evaluation results and the chosen preset.

Every step is accepted based on before/after measurements on the evaluation set
(speed + Recall/MRR).
