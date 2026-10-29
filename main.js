const { Plugin, Modal, Notice, PluginSettingTab, Setting } = require('obsidian');
const { execFile } = require('child_process');
const os = require('os');
const path = require('path');
const fs = require('fs');

const DEFAULT_SETTINGS = {
    pythonPath: '',
    modelDir: '',
    device: 'auto',
    restUrl: 'http://127.0.0.1:27123',
    indexFile: '',
    topK: 10
};

function expandHome(value) {
    if (!value) return value;
    if (value === '~') return os.homedir();
    if (value.startsWith('~/') || value.startsWith('~\\')) {
        return path.join(os.homedir(), value.slice(2));
    }
    return value;
}

function getPluginDir(app, plugin) {
    try {
        const vaultPath = app.vault.adapter.getBasePath ? app.vault.adapter.getBasePath() : '';
        const manifestDir = plugin && plugin.manifest ? plugin.manifest.dir : '';
        if (manifestDir) {
            if (path.isAbsolute(manifestDir)) return manifestDir;
            if (vaultPath) return path.join(vaultPath, manifestDir);
        }
        if (vaultPath) {
            const configDir = app.vault.configDir || '.obsidian';
            return path.join(vaultPath, configDir, 'plugins', 'npu-search');
        }
    } catch (e) {}
    return __dirname;
}

function getScriptPath(app, plugin) {
    return path.join(getPluginDir(app, plugin), 'scripts', 'npu_vault_search.py');
}

function getDataDir() {
    const custom = (process.env.NPU_DATA_DIR || '').trim();
    if (custom) return expandHome(custom);
    const xdg = (process.env.XDG_DATA_HOME || '').trim();
    const base = xdg || path.join(os.homedir(), '.local', 'share');
    return path.join(base, 'npu-search');
}

function getPythonPath(app, plugin) {
    const settings = (plugin && plugin.settings) || DEFAULT_SETTINGS;
    if (settings.pythonPath) return expandHome(settings.pythonPath);

    const pluginDir = getPluginDir(app, plugin);
    const dataDir = getDataDir();
    const candidates = process.platform === 'win32'
        ? [
            path.join(pluginDir, '.venv', 'Scripts', 'python.exe'),
            path.join(dataDir, 'venv', 'Scripts', 'python.exe')
        ]
        : [
            path.join(pluginDir, '.venv', 'bin', 'python'),
            path.join(dataDir, 'venv', 'bin', 'python')
        ];
    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) return candidate;
    }
    return process.platform === 'win32' ? 'python' : 'python3';
}

function getRuntimeArgs(app, plugin) {
    const settings = (plugin && plugin.settings) || DEFAULT_SETTINGS;
    const args = [
        '--device', settings.device || 'auto',
        '--rest-url', settings.restUrl || DEFAULT_SETTINGS.restUrl
    ];
    if (settings.modelDir) args.push('--model-dir', expandHome(settings.modelDir));
    if (settings.indexFile) args.push('--index-file', expandHome(settings.indexFile));

    try {
        const vaultPath = app.vault.adapter.getBasePath ? app.vault.adapter.getBasePath() : '';
        if (vaultPath) args.push('--vault-path', vaultPath);
    } catch (e) {}
    return args;
}

function runPython(app, plugin, extraArgs, callback) {
    const python = getPythonPath(app, plugin);
    const script = getScriptPath(app, plugin);
    if (!fs.existsSync(script)) {
        callback(new Error(`Не найден скрипт: ${script}`), '', '');
        return;
    }
    execFile(
        python,
        [script, ...extraArgs],
        { maxBuffer: 8 * 1024 * 1024, env: { ...process.env, PYTHONUNBUFFERED: '1' } },
        callback
    );
}

function pythonErrorText(error, stderr) {
    const tail = (stderr || '').trim().split('\n').slice(-6).join('\n');
    return tail ? `${error.message}\n${tail}` : error.message;
}

class NPUSearchModal extends Modal {
    constructor(app, plugin, initialFilter = 'all') {
        super(app);
        this.plugin = plugin;
        this.results = [];
        this.selectedIndex = -1;
        this.lastSearchedQuery = '';
        this.currentFilter = initialFilter;
        this.isSearching = false;
        this.filterButtons = {};
    }

    onOpen() {
        // Установка четких размеров и нативного класса модального окна
        this.modalEl.addClass('npu-modal-window');
        this.modalEl.style.width = '760px';
        this.modalEl.style.maxWidth = '92vw';

        const { contentEl } = this;
        contentEl.empty();
        contentEl.addClass('npu-search-modal');

        // Заголовок окна
        const header = contentEl.createDiv({ cls: 'npu-search-header' });
        const titleWrap = header.createDiv({ cls: 'npu-header-title-wrap' });
        titleWrap.createEl('span', { text: '🧠', cls: 'npu-header-icon' });
        titleWrap.createEl('h2', { text: 'Семантический NPU-поиск' });

        const badge = header.createDiv({ cls: 'npu-header-badge' });
        badge.setText('NPU 3720 • E5-Base');

        // Строка поиска (Flex-контейнер без наложения иконок на текст)
        const box = contentEl.createDiv({ cls: 'npu-search-box' });
        const inputWrap = box.createDiv({ cls: 'npu-input-wrapper' });
        
        // Векторная четкая иконка поиска
        const iconSpan = inputWrap.createSpan({ cls: 'npu-search-icon-wrap' });
        iconSpan.innerHTML = `
            <svg class="npu-search-svg" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
        `;
        
        const input = inputWrap.createEl('input', {
            type: 'text',
            placeholder: 'Введите мысль, тему или концепцию...',
            cls: 'npu-search-input'
        });

        // Векторная четкая кнопка очистки
        const clearBtn = inputWrap.createEl('button', {
            cls: 'npu-clear-btn',
            attr: { 'aria-label': 'Очистить' }
        });
        clearBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
        `;
        clearBtn.style.display = 'none';

        const searchBtn = box.createEl('button', {
            text: 'Найти',
            cls: 'npu-search-btn'
        });

        // Панель фильтров категорий
        const filterBar = contentEl.createDiv({ cls: 'npu-filter-bar' });
        const filters = [
            { id: 'all', label: '🌐 Все' },
            { id: 'moc', label: '📚 Карта знаний' },
            { id: 'projects', label: '🚀 Проекты' },
            { id: 'dev', label: '💻 Dev & Linux' },
            { id: 'deutsch', label: '🇩🇪 Deutsch' }
        ];

        filters.forEach(f => {
            const btn = filterBar.createEl('button', {
                cls: `npu-filter-chip ${f.id === this.currentFilter ? 'active' : ''}`,
                text: f.label
            });
            btn.dataset.filter = f.id;
            this.filterButtons[f.id] = btn;

            btn.addEventListener('click', () => {
                if (this.currentFilter === f.id) return;
                this.setFilter(f.id, resultsEl, input);
            });
        });

        // Контейнер результатов
        const resultsEl = contentEl.createDiv({ cls: 'npu-search-results' });
        
        // Начальное состояние
        this.renderInitialView(resultsEl, input);

        // Футер с подсказками клавиш
        const footer = contentEl.createDiv({ cls: 'npu-search-footer' });
        footer.innerHTML = `
            <div class="npu-shortcut-hint"><kbd>↵ Enter</kbd> <span>Поиск / Открыть</span></div>
            <div class="npu-shortcut-hint"><kbd>↑ ↓</kbd> <span>Навигация</span></div>
            <div class="npu-shortcut-hint"><kbd>Esc</kbd> <span>Закрыть</span></div>
        `;

        const updateClearButton = () => {
            clearBtn.style.display = input.value.trim() ? 'flex' : 'none';
        };

        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            input.value = '';
            updateClearButton();
            this.results = [];
            this.selectedIndex = -1;
            this.lastSearchedQuery = '';
            this.renderInitialView(resultsEl, input);
            input.focus();
        });

        const runSearch = () => {
            const query = input.value.trim();
            if (!query) {
                if (this.currentFilter === 'moc') {
                    this.renderMocOverview(resultsEl);
                } else if (this.currentFilter === 'projects') {
                    this.renderProjectsOverview(resultsEl);
                } else {
                    new Notice('Введите поисковый запрос');
                }
                return;
            }

            if (this.isSearching) return;
            this.isSearching = true;
            searchBtn.disabled = true;
            searchBtn.addClass('loading');
            searchBtn.setText('Поиск...');

            resultsEl.empty();
            const loading = resultsEl.createDiv({ cls: 'npu-loading-state' });
            loading.createDiv({ cls: 'npu-spinner' });
            loading.createDiv({ 
                text: `Вычисление семантики на NPU [${this.getFilterLabel(this.currentFilter)}]...`, 
                cls: 'npu-loading-text' 
            });

            const args = [
                '--query', query,
                '--json',
                '--top-k', String(this.plugin.settings.topK || 10),
                '--filter', this.currentFilter,
                ...getRuntimeArgs(this.app, this.plugin)
            ];

            runPython(this.app, this.plugin, args, (error, stdout, stderr) => {
                this.isSearching = false;
                searchBtn.disabled = false;
                searchBtn.removeClass('loading');
                searchBtn.setText('Найти');
                loading.remove();

                if (error) {
                    const detail = pythonErrorText(error, stderr);
                    new Notice('❌ Ошибка NPU: ' + detail, 10000);
                    const errEl = resultsEl.createDiv({ cls: 'npu-empty-state error' });
                    errEl.setText('Ошибка выполнения поиска: ' + detail);
                    return;
                }

                try {
                    const data = JSON.parse(stdout);
                    this.results = data || [];
                    this.selectedIndex = -1;
                    this.lastSearchedQuery = query;

                    if (!data || data.length === 0) {
                        const empty = resultsEl.createDiv({ cls: 'npu-empty-state' });
                        empty.setText(`Ничего не найдено в фильтре "${this.getFilterLabel(this.currentFilter)}". Попробуйте выбрать фильтр "Все".`);
                        return;
                    }

                    this.renderCards(resultsEl, data);

                    // Автофокус на первом результате
                    if (data.length > 0) {
                        this.highlightIndex(resultsEl, 0);
                    }

                } catch (e) {
                    const errEl = resultsEl.createDiv({ cls: 'npu-empty-state error' });
                    errEl.setText('Ошибка парсинга вывода NPU: ' + e.message);
                }
            });
        };

        this.runSearchAction = runSearch;
        searchBtn.addEventListener('click', runSearch);

        input.addEventListener('input', () => {
            updateClearButton();
            if (input.value.trim() !== this.lastSearchedQuery) {
                this.selectedIndex = -1;
            }
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const currentQuery = input.value.trim();
                
                if (!this.lastSearchedQuery || currentQuery !== this.lastSearchedQuery) {
                    runSearch();
                } else if (this.selectedIndex >= 0 && this.results[this.selectedIndex]) {
                    this.openNote(this.results[this.selectedIndex].filename);
                } else if (this.results.length > 0) {
                    this.openNote(this.results[0].filename);
                } else {
                    runSearch();
                }
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                this.navigateResults(resultsEl, 1);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                this.navigateResults(resultsEl, -1);
            }
        });

        setTimeout(() => input.focus(), 50);
    }

    getFilterLabel(filterId) {
        const labels = {
            'all': 'Все',
            'moc': 'Карта знаний',
            'projects': 'Проекты',
            'dev': 'Dev & Linux',
            'deutsch': 'Deutsch'
        };
        return labels[filterId] || filterId;
    }

    setFilter(filterId, resultsEl, input) {
        this.currentFilter = filterId;
        Object.values(this.filterButtons).forEach(b => b.removeClass('active'));
        if (this.filterButtons[filterId]) {
            this.filterButtons[filterId].addClass('active');
        }

        const query = input.value.trim();
        if (query) {
            this.runSearchAction();
        } else {
            this.renderInitialView(resultsEl, input);
        }
    }

    renderInitialView(resultsEl, input) {
        resultsEl.empty();
        if (this.currentFilter === 'moc') {
            this.renderMocOverview(resultsEl);
        } else if (this.currentFilter === 'projects') {
            this.renderProjectsOverview(resultsEl);
        } else {
            const initialHint = resultsEl.createDiv({ cls: 'npu-empty-state' });
            initialHint.createEl('div', { 
                text: '💡 Введите запрос и нажмите Enter для семантического поиска', 
                cls: 'npu-empty-title' 
            });
            initialHint.createEl('div', { 
                text: 'Поиск работает по смыслу и ассоциациям на нейрочипе NPU, даже если слова не совпадают буквально.', 
                cls: 'npu-empty-desc' 
            });
        }
    }

    renderMocOverview(container) {
        container.empty();
        const header = container.createDiv({ cls: 'npu-section-header' });
        header.setText('📚 Карты знаний (MOC) в хранилище:');

        const mocNotes = [
            {
                filename: '📚 Карта знаний (MOC).md',
                title: '📚 Карта знаний (MOC)',
                desc: 'Главная навигационная карта по всей базе знаний и конспектам'
            },
            {
                filename: 'dev/🗺️ Карта проектов (MOC).md',
                title: '🗺️ Карта проектов (MOC)',
                desc: 'Реестр и карта всех 18 активных и завершенных проектов'
            }
        ];

        this.results = mocNotes;
        this.selectedIndex = -1;

        mocNotes.forEach((item, index) => {
            const card = container.createDiv({ cls: 'npu-result-card moc-featured' });
            const info = card.createDiv({ cls: 'npu-result-info' });
            info.createDiv({ text: item.title, cls: 'npu-result-title' });
            info.createDiv({ text: item.desc, cls: 'npu-result-path' });

            const meta = card.createDiv({ cls: 'npu-result-meta' });
            meta.createDiv({ text: 'MOC', cls: 'npu-badge-score score-high' });

            card.addEventListener('click', () => {
                this.openNote(item.filename);
            });
            card.addEventListener('mouseenter', () => {
                this.highlightIndex(container, index);
            });
        });

        const tip = container.createDiv({ cls: 'npu-empty-desc', attr: { style: 'margin-top: 15px; text-align: center;' } });
        tip.setText('💡 Введите любой запрос в строке поиска выше для аппаратного NPU-поиска по Карте знаний.');

        if (mocNotes.length > 0) {
            this.highlightIndex(container, 0);
        }
    }

    renderProjectsOverview(container) {
        container.empty();
        const header = container.createDiv({ cls: 'npu-section-header' });
        header.setText('🚀 Быстрый доступ к проектам:');

        const projNotes = [
            {
                filename: 'dev/🗺️ Карта проектов (MOC).md',
                title: '🗺️ Карта проектов (MOC)',
                desc: 'Общая карта всех 18 проектов'
            },
            {
                filename: 'dev/Projects/AI/Obsidian NPU Search/Obsidian NPU Search.md',
                title: 'Obsidian NPU Search Plugin',
                desc: 'Плагин семантического поиска на NPU'
            },
            {
                filename: 'dev/Projects/AI/Local AI/Локальный ИИ (Qwen 14B OpenVINO).md',
                title: 'Локальный ИИ (Qwen 14B)',
                desc: 'Автономный ассистент на Intel Arc GPU'
            },
            {
                filename: 'dev/Projects/AI/Intel NPU OpenVINO/Intel NPU OpenVINO.md',
                title: 'Intel NPU OpenVINO',
                desc: 'Аппаратный инференс на NPU 3720'
            }
        ];

        this.results = projNotes;
        this.selectedIndex = -1;

        projNotes.forEach((item, index) => {
            const card = container.createDiv({ cls: 'npu-result-card' });
            const info = card.createDiv({ cls: 'npu-result-info' });
            info.createDiv({ text: item.title, cls: 'npu-result-title' });
            info.createDiv({ text: item.desc, cls: 'npu-result-path' });

            const meta = card.createDiv({ cls: 'npu-result-meta' });
            meta.createDiv({ text: 'PROJECT', cls: 'npu-badge-score score-mid' });

            card.addEventListener('click', () => {
                this.openNote(item.filename);
            });
            card.addEventListener('mouseenter', () => {
                this.highlightIndex(container, index);
            });
        });

        if (projNotes.length > 0) {
            this.highlightIndex(container, 0);
        }
    }

    renderCards(container, data) {
        data.forEach((item, index) => {
            const card = container.createDiv({ cls: 'npu-result-card' });
            
            const info = card.createDiv({ cls: 'npu-result-info' });
            const titleText = item.filename.split('/').pop().replace(/\.md$/, '');
            // Полное отображение названия без принудительного многоточия
            info.createDiv({ text: titleText, cls: 'npu-result-title' });
            
            const pathParts = item.filename.split('/');
            const dir = pathParts.length > 1 ? pathParts.slice(0, -1).join(' / ') : '';
            const pathEl = info.createDiv({ cls: 'npu-result-path' });
            const section = item.section ? ` · ${item.section}` : '';
            if (dir) {
                pathEl.innerHTML = '<span class="npu-folder-icon">📁</span> ';
                pathEl.createSpan({ text: `${dir}${section}` });
            } else {
                pathEl.innerHTML = '<span class="npu-folder-icon">📄</span> ';
                pathEl.createSpan({ text: `${item.filename}${section}` });
            }

            const meta = card.createDiv({ cls: 'npu-result-meta' });
            const scoreVal = Number(item.score);
            const percent = (scoreVal * 100).toFixed(1);
            
            let scoreCls = 'score-mid';
            if (scoreVal >= 0.85) scoreCls = 'score-high';
            else if (scoreVal < 0.80) scoreCls = 'score-low';

            meta.createDiv({ text: `${percent}%`, cls: `npu-badge-score ${scoreCls}` });
            meta.createDiv({ 
                text: `${Math.round(item.infer_time_ms)} мс • ${item.device}`, 
                cls: 'npu-badge-latency' 
            });

            card.addEventListener('click', () => {
                this.openNote(item.filename);
            });

            card.addEventListener('mouseenter', () => {
                this.highlightIndex(container, index);
            });
        });
    }

    navigateResults(container, direction) {
        if (!this.results || this.results.length === 0) return;
        let newIndex = this.selectedIndex + direction;
        if (this.selectedIndex === -1) {
            newIndex = direction > 0 ? 0 : this.results.length - 1;
        } else if (newIndex >= this.results.length) {
            newIndex = 0;
        } else if (newIndex < 0) {
            newIndex = this.results.length - 1;
        }
        this.highlightIndex(container, newIndex);
    }

    highlightIndex(container, index) {
        this.selectedIndex = index;
        const cards = container.querySelectorAll('.npu-result-card');
        cards.forEach((c, idx) => {
            if (idx === index) {
                c.addClass('selected');
                c.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            } else {
                c.removeClass('selected');
            }
        });
    }

    openNote(filename) {
        this.app.workspace.openLinkText(filename, '', false);
        this.close();
    }

    onClose() {
        const { contentEl } = this;
        contentEl.empty();
    }
}

class NPUSearchSettingTab extends PluginSettingTab {
    constructor(app, plugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display() {
        const { containerEl } = this;
        containerEl.empty();
        containerEl.createEl('h2', { text: 'Семантический NPU-поиск' });

        new Setting(containerEl)
            .setName('Python')
            .setDesc('Путь к интерпретатору. Пусто — автоопределение: venv в папке плагина, ~/.local/share/npu-search/venv, затем python3.')
            .addText(text => text
                .setPlaceholder('/usr/bin/python3')
                .setValue(this.plugin.settings.pythonPath)
                .onChange(async value => {
                    this.plugin.settings.pythonPath = value.trim();
                    await this.plugin.saveSettings();
                }));

        new Setting(containerEl)
            .setName('Модель')
            .setDesc('Папка Multilingual-E5-Base с tokenizer.json и openvino/openvino_model.xml. Пусто — ~/.local/share/npu-search/models.')
            .addText(text => text
                .setPlaceholder('~/.local/share/npu-search/models/Multilingual-E5-Base')
                .setValue(this.plugin.settings.modelDir)
                .onChange(async value => {
                    this.plugin.settings.modelDir = value.trim();
                    await this.plugin.saveSettings();
                }));

        new Setting(containerEl)
            .setName('Устройство')
            .setDesc('Авто: NPU → GPU → CPU. На X270 без NPU автоматически будет CPU.')
            .addDropdown(dropdown => dropdown
                .addOption('auto', 'Авто (NPU → GPU → CPU)')
                .addOption('NPU', 'Intel NPU')
                .addOption('GPU', 'Intel GPU')
                .addOption('CPU', 'CPU')
                .setValue(this.plugin.settings.device)
                .onChange(async value => {
                    this.plugin.settings.device = value;
                    await this.plugin.saveSettings();
                }));

        new Setting(containerEl)
            .setName('Obsidian REST API')
            .setDesc('Используется только если заметки не читаются напрямую с диска.')
            .addText(text => text
                .setPlaceholder('http://127.0.0.1:27123')
                .setValue(this.plugin.settings.restUrl)
                .onChange(async value => {
                    this.plugin.settings.restUrl = value.trim() || DEFAULT_SETTINGS.restUrl;
                    await this.plugin.saveSettings();
                }));

        new Setting(containerEl)
            .setName('Результатов в выдаче')
            .addSlider(slider => slider
                .setLimits(3, 30, 1)
                .setValue(this.plugin.settings.topK)
                .setDynamicTooltip()
                .onChange(async value => {
                    this.plugin.settings.topK = value;
                    await this.plugin.saveSettings();
                }));

        new Setting(containerEl)
            .setName('Окружение')
            .setDesc('Проверить Python, OpenVINO, модель, устройство и индекс.')
            .addButton(button => button
                .setButtonText('Проверить')
                .onClick(() => this.plugin.checkEnvironment()));

        new Setting(containerEl)
            .setName('Индекс')
            .setDesc('Полностью пересобрать векторный индекс заметок.')
            .addButton(button => button
                .setButtonText('Пересобрать')
                .onClick(() => this.plugin.rebuildIndex()));
    }
}

module.exports = class NPUSearchPlugin extends Plugin {
    async onload() {
        console.log('[NPU Search] Загрузка плагина NPU Semantic Search');
        await this.loadSettings();
        this.addSettingTab(new NPUSearchSettingTab(this.app, this));

        // Иконка в ленте слева (Ribbon)
        this.addRibbonIcon('brain', 'Семантический NPU-поиск', () => {
            this.openSearch();
        });

        // Команда быстрого поиска (Ctrl+P)
        this.addCommand({
            id: 'npu-semantic-search',
            name: 'Семантический поиск по смыслу (NPU)',
            hotkeys: [{ modifiers: ['Mod', 'Alt'], key: 'f' }],
            callback: () => {
                this.openSearch();
            }
        });

        // Хоткей Ctrl+Alt+F по физической клавише (event.code) —
        // работает при любой раскладке (клавиша F даёт 'а' в русской раскладке,
        // поэтому сверка по key: 'f' не срабатывает).
        // Ctrl+Alt+D и ★-клавиша ThinkPad (KEY_BOOKMARKS) — фильтр Deutsch.
        const starKeyNames = ['Favorites', 'BrowserFavorites', 'XF86Favorites', 'Bookmarks',
            'LaunchAssistant'];
        this.registerDomEvent(document, 'keydown', (event) => {
            if ((event.ctrlKey || event.metaKey) && event.altKey && !event.shiftKey) {
                if (event.code === 'KeyF') {
                    event.preventDefault();
                    this.openSearch();
                } else if (event.code === 'KeyD') {
                    event.preventDefault();
                    this.openSearch('deutsch');
                }
                return;
            }
            const isStarKey = starKeyNames.includes(event.code) || starKeyNames.includes(event.key);
            if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey
                && isStarKey) {
                event.preventDefault();
                this.openSearch('deutsch');
                return;
            }
            // Диагностика: спецклавиши с неизвестными именами пишем в keys.log,
            // чтобы по логу найти код ★-клавиши, если она не в списке выше.
            if (event.key && event.key.length > 1) {
                this.logUnknownKey(event);
            }
        });

        // Команда поиска по немецкому (фильтр Deutsch)
        this.addCommand({
            id: 'npu-german-search',
            name: 'Семантический поиск по немецкому (Deutsch)',
            hotkeys: [
                { modifiers: ['Mod', 'Alt'], key: 'd' },
            ],
            callback: () => {
                this.openSearch('deutsch');
            }
        });

        // Команда пересборки индекса хранилища
        this.addCommand({
            id: 'npu-rebuild-index',
            name: 'Пересобрать векторный NPU-индекс заметок',
            callback: () => this.rebuildIndex()
        });
    }

    logUnknownKey(event) {
        // Игнорируем обычные спецклавиши, чтобы не засорять лог.
        const common = ['Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'Tab', 'Enter',
            'Escape', 'Backspace', 'Delete', 'ArrowUp', 'ArrowDown', 'ArrowLeft',
            'ArrowRight', 'Home', 'End', 'PageUp', 'PageDown', 'ContextMenu'];
        if (common.includes(event.key)) return;
        if (/^F\d{1,2}$/.test(event.key)) return;
        try {
            const logFile = path.join(getDataDir(), 'keys.log');
            const stamp = new Date().toISOString().slice(11, 19);
            fs.appendFileSync(logFile,
                `${stamp} key=${event.key} code=${event.code} keyCode=${event.keyCode}\n`);
        } catch (e) { /* диагностика не должна ломать работу */ }
    }

    openSearch(initialFilter = 'all') {
        if (this.activeSearchModal) return;
        const modal = new NPUSearchModal(this.app, this, initialFilter);
        this.activeSearchModal = modal;
        const originalOnClose = modal.onClose.bind(modal);
        modal.onClose = () => {
            this.activeSearchModal = null;
            originalOnClose();
        };
        modal.open();
    }

    async loadSettings() {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    }

    async saveSettings() {
        await this.saveData(this.settings);
    }

    runPython(args, callback) {
        return runPython(this.app, this, args, callback);
    }

    checkEnvironment() {
        new Notice('🔎 Проверка окружения NPU...');
        const args = ['--check', ...getRuntimeArgs(this.app, this)];
        this.runPython(args, (error, stdout, stderr) => {
            if (error) {
                new Notice('❌ Проверка не удалась: ' + pythonErrorText(error, stderr), 10000);
                return;
            }
            try {
                const info = JSON.parse(stdout);
                const parts = [
                    `Python: ${info.python}`,
                    `OpenVINO: ${info.openvino || 'нет'}`,
                    `Устройства: ${(info.devices || []).join(', ') || 'нет'}`,
                    `Выбрано: ${info.selected_device || 'не проверено'}`,
                    `Модель: ${info.model_ok ? 'OK' : 'НЕ НАЙДЕНА'}`,
                    `Индекс: ${info.index_notes} заметок`
                ];
                new Notice('🧠 ' + parts.join('\n'), 12000);
                console.log('[NPU Search] environment:', info);
            } catch (e) {
                new Notice('⚠️ Не удалось разобрать ответ проверки: ' + stdout, 10000);
            }
        });
    }

    rebuildIndex() {
        new Notice('⚡ Пересборка векторного индекса...');
        const args = ['--build-index', ...getRuntimeArgs(this.app, this)];
        this.runPython(args, (error, stdout, stderr) => {
            if (error) {
                new Notice('❌ Ошибка пересборки: ' + pythonErrorText(error, stderr), 10000);
                return;
            }
            console.log('[NPU Search] rebuild:', stderr);
            new Notice('✅ Векторный индекс обновлён!', 5000);
        });
    }

    onunload() {
        console.log('[NPU Search] Выгрузка плагина NPU Semantic Search');
    }
};
