# 🧠 Obsidian NPU Semantic Search

Семантический (векторный) поиск по заметкам Obsidian. Плагин запускает
[`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)
через **OpenVINO** и сам выбирает ускоритель: **Intel NPU → Intel GPU → CPU**.

Работает и на ноутбуках без NPU: на ThinkPad X270 с i5-6300U поиск идёт на CPU,
индекс строится один раз и дальше обновляется инкрементально.

---

## ✨ Возможности

- Поиск по смыслу, а не по точному совпадению слов.
- Аппаратное ускорение на Intel NPU (AI Boost), с автоматическим fallback на GPU/CPU.
- Категорийные фильтры: все заметки, карты знаний (MOC), проекты, `dev/`, немецкий.
- Инкрементальный индекс: пересчитываются только новые и изменённые заметки.
- Прямое чтение vault с диска (`--vault-path`) — быстро и без Obsidian REST API.
- Отдельный CLI для проверки и пересборки индекса.
- Настройки прямо в Obsidian: Python, модель, устройство, REST URL, top-K.

---

## 🛠 Требования

- Obsidian Desktop (плагин использует Node.js `child_process`).
- Python **3.10+**.
- Python-пакеты: `openvino`, `tokenizers`, `numpy` (ставятся через `scripts/setup.sh`).
- Модель `Multilingual-E5-Base` в формате OpenVINO IR (≈1.1 ГБ).
- Для NPU: Meteor Lake / Lunar Lake и драйвер `intel_vpu`; для GPU — Intel GPU;
  CPU работает всегда.

---

## 📦 Установка

Репозиторий — обычный git-проект. В vault он подключается **симлинком**, потому что
Obsidian умеет загружать плагины только из `<vault>/.obsidian/plugins/<id>`.

### 1. Клонировать и подключить

```bash
git clone git@github.com:direktpurpl/obsidian-npu-search.git \
  ~/projekts/Puthon/obsidian-npu-search

ln -s ~/projekts/Puthon/obsidian-npu-search \
  "$VAULT/.obsidian/plugins/npu-search"
```

`$VAULT` — путь к хранилищу, например `~/Документы/learn`.

### 2. Создать Python-окружение

```bash
~/projekts/Puthon/obsidian-npu-search/scripts/setup.sh
```

Скрипт создаёт venv **вне репозитория**:

```text
~/.local/share/npu-search/venv
```

### 3. Положить модель

Вариант A — скопировать уже готовую модель с другого компьютера:

```bash
~/.local/share/npu-search/venv/bin/python \
  ~/projekts/Puthon/obsidian-npu-search/scripts/fetch_model.py \
  --from /path/to/Multilingual-E5-Base
```

Например, с main по Tailscale:

```bash
rsync -a --info=progress2 \
  /home/direkt/.local/share/npu-search/models/Multilingual-E5-Base/ \
  direkt@100.64.0.25:~/.local/share/npu-search/models/Multilingual-E5-Base/
```

Вариант B — скачать и экспортировать из HuggingFace:

```bash
~/.local/share/npu-search/venv/bin/python -m pip install -r requirements-model.txt
~/.local/share/npu-search/venv/bin/python scripts/fetch_model.py --download
```

По умолчанию модель ищется здесь:

```text
~/.local/share/npu-search/models/Multilingual-E5-Base
```

Путь можно переопределить через `NPU_MODEL_DIR` или настройку плагина «Модель».

### 4. Включить плагин

`Obsidian → Settings → Community plugins → Intel NPU Semantic Search`.

Поиск: иконка 🧠 слева или `Ctrl+P` → «Семантический поиск по смыслу (NPU)».

---

## ⚙️ Настройки плагина

| Настройка | По умолчанию | Описание |
| :-- | :-- | :-- |
| Python | авто | venv плагина → `~/.local/share/npu-search/venv` → `python3` |
| Модель | `~/.local/share/npu-search/models/Multilingual-E5-Base` | Папка с `tokenizer.json` и OpenVINO IR |
| Устройство | `auto` | `NPU` → `GPU` → `CPU`, либо принудительно |
| Obsidian REST API | `http://127.0.0.1:27123` | Только если vault не читается напрямую |
| Результатов | 10 | Сколько карточек показывать |
| Проверить | — | Диагностика окружения |
| Пересобрать | — | Полная переиндексация |

Кнопка «Проверить» показывает: версию Python, OpenVINO, доступные устройства,
выбранное устройство, наличие модели и число заметок в индексе.

---

## 💻 CLI

```bash
PY=~/.local/share/npu-search/venv/bin/python
SCRIPT=~/projekts/Puthon/obsidian-npu-search/scripts/npu_vault_search.py

# Проверка окружения
$PY $SCRIPT --check

# Поиск
$PY $SCRIPT --query "docker deployment" --top-k 5

# Поиск только по dev/
$PY $SCRIPT --query "systemd timer" --filter dev

# Полная пересборка индекса
$PY $SCRIPT --build-index --vault-path ~/Документы/learn

# Принудительно CPU (например, для отладки)
$PY $SCRIPT --query "backup" --device CPU
```

Полезные переменные окружения:

| Переменная | Назначение |
| :-- | :-- |
| `NPU_MODEL_DIR` | Путь к модели |
| `NPU_INDEX_FILE` | Путь к `.npz`-индексу |
| `NPU_CACHE_DIR` | Кэш компиляции OpenVINO |
| `NPU_DATA_DIR` | Базовый каталог данных (по умолчанию `~/.local/share/npu-search`) |
| `OBSIDIAN_VAULT_PATH` | Корень vault |
| `OBSIDIAN_API_KEY` | Ключ Local REST API |

---

## 🗂 Где что лежит

```text
~/projekts/Puthon/obsidian-npu-search/     # git-репозиторий (только код)
├── main.js                                # UI и команды Obsidian
├── manifest.json
├── styles.css
├── requirements.txt
├── requirements-model.txt
└── scripts/
    ├── npu_vault_search.py                # движок поиска
    ├── fetch_model.py                     # копирование/экспорт модели
    └── setup.sh                           # venv + зависимости

~/.local/share/npu-search/                 # runtime, вне git
├── venv/
├── models/Multilingual-E5-Base/
├── indexes/<hash-vault>.npz
└── cache/openvino/
```

Индекс привязан к пути vault (хэш), поэтому несколько хранилищ не конфликтуют.
Файлы `models/`, `venv/`, `.cache/` и `*.npz` не попадают в git.

---

## 🚀 Производительность

Замеры на main (Core Ultra, NPU) и X270 (i5-6300U, CPU, AVX2).
Индекс — 207 заметок / ~1050 секций:

| Операция | NPU (main) | CPU (X270) |
| :-- | --: | --: |
| Эмбеддинг запроса | ~130 мс | ~90 мс |
| Загрузка модели на запрос | ~0.9 с | ~1.3–1.6 с |
| Полный поиск (запрос + выдача) | ~1.1 с | ~1.5–2.2 с |
| Полная индексация | ~63 с | ~4–5 мин |
| Инкрементальное обновление | секунды | десятки секунд |

Поиск запускает отдельный процесс Python (zero idle RAM): модель грузится на
время запроса и выгружается. Если нужен мгновенный отклик, следующий шаг —
фоновый демон (см. roadmap проекта).

---

## 🧩 Как это работает

1. Плагин вызывает `npu_vault_search.py` с `--query` и `--json`.
2. Скрипт загружает эмбеддинги заметок из `.npz` (или строит индекс).
3. Запрос кодируется моделью `multilingual-e5-base` (`query: ...`).
4. Считается косинусное сходство, результаты возвращаются в UI.

Заметка режется на секции по заголовкам; каждая секция индексируется отдельно
(до 40 секций на заметку). Поэтому поиск находит нужный раздел даже в длинной
заметке, а не только первые 1200 символов.

Индекс хранит: путь заметки, название секции, вектор 768 измерений, `mtime` и
размер файла. При поиске изменённые заметки переиндексируются автоматически,
а в результатах показывается найденная секция.

---

## 🐧 ThinkPad X270

X270 (i5-6300U, 8 ГБ RAM) не имеет NPU и Intel Arc. Плагин это учитывает:

- `device=auto` выбирает **CPU**;
- для CPU используется динамическая длина последовательности (без паддинга до 512),
  поэтому короткие запросы считаются заметно быстрее;
- модель и venv лежат в `~/.local/share/npu-search`, а не в vault;
- индекс строится локально и не синхронизируется через Syncthing;
- поиск работает без запущенного Obsidian REST API, если передан `--vault-path`.

Модель на X270 можно залить с main по Tailscale (см. выше) — 1.1 ГБ при прямой
Tailscale-связи занимает около 20–30 секунд.

---

## 🧯 Troubleshooting

**`No module named openvino`**
Запустите `scripts/setup.sh` или укажите путь к Python в настройках плагина.

**`OpenVINO model not found`**
Модель не найдена. Запустите `fetch_model.py --from ...` или `--download`,
либо пропишите путь в настройке «Модель».

**NPU недоступен, выбрался CPU**
Это нормально. Проверьте `--check`: в списке устройств должен быть `NPU`.
На X270 NPU нет — CPU является ожидаемым поведением.

**REST API 401 / connection refused**
Плагин и так читает vault с диска. REST нужен только для внешних сценариев.
Ключ берётся из `.obsidian/plugins/obsidian-local-rest-api/data.json`.

**Индекс устарел**
Команда «Пересобрать векторный NPU-индекс заметок» или `--build-index`.
При обычном поиске изменения подхватываются автоматически по `mtime`.

---

## 📄 Лицензия

MIT
