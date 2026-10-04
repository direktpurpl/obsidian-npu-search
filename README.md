# 🧠 Obsidian NPU Semantic Search

An ultra-fast, zero-RAM, hardware-accelerated semantic vector search plugin for [Obsidian](https://obsidian.md), powered by **Intel AI Boost (NPU 3720)** and OpenVINO with `intfloat/multilingual-e5-base` (768-dimensional embeddings).

---

## ✨ Features

- **Semantic & Associative Search:** Find notes by concepts and ideas, even without exact keyword matches.
- **Hardware NPU Acceleration:** Computes vector embeddings in **~66 ms** on the dedicated Intel AI Boost NPU (Meteor Lake Core Ultra), consuming 0 MB of discrete GPU VRAM and keeping your laptop cool and quiet.
- **Interactive Category Filters:**
  - 🌐 **Все (All Notes):** Search across the entire knowledge base.
  - 📚 **Карта знаний (MOC):** Focused search across curated Map of Content notes and indices.
  - 🚀 **Проекты (Projects):** Search across dev project cards and roadmaps.
  - 💻 **Dev & Linux:** Technical programming and system administration notes.
  - 🇩🇪 **Deutsch:** German vocabulary and grammar.
- **Native Obsidian UI:**
  - Crisp high-DPI typography (optimized for 2.8K displays without scaling blur).
  - Full keyboard navigation (`↑` / `↓` for selection, `Enter` to search or open note, `Esc` to close).
  - Clear match score badges (`≥ 85%` High, `80–84%` Mid, `< 80%` Low) and inference latency indicators.
  - Dynamic wrapping for long file names without ugly truncations.

---

## 🛠 Architecture & Prerequisites

1. **Hardware & OS:**
   - Intel Core Ultra (Meteor Lake / Lunar Lake) with Intel AI Boost NPU (e.g. NPU 3720).
   - Linux kernel with `intel_vpu` driver, Level-Zero loader, and OpenVINO NPU plugin.
2. **Python Environment:**
   ```bash
   pip install openvino tokenizers numpy
   ```
3. **Model:**
   - `intfloat/multilingual-e5-base` OpenVINO IR format (768-dim embeddings).

---

## 🚀 Installation

1. Copy or clone this folder into your Obsidian vault's plugins directory:
   ```bash
   cd <your-vault>/.obsidian/plugins/
   git clone https://github.com/direktpurpl/obsidian-npu-search npu-search
   ```
2. Enable **Intel NPU Semantic Search** in Obsidian: `Settings -> Community plugins`.
3. Open search via `Ctrl + P` -> **Семантический поиск по смыслу (NPU)** or click the ribbon brain icon on the left.

---

## 📄 License
MIT License
