/**
 * PresetSelector — System Prompt 预设选择器组件
 *
 * 从 /api/presets 加载按模式分组的 YAML 预设，渲染为按钮组。
 * 选中预设后回调通知页面更新 system prompt。
 *
 * API 返回格式: { "turnbased": [...], "audio_duplex": [...], "omni": [...] }
 */
class PresetSelector {
    constructor({ container, page, detailsEl, onSelect, storageKey }) {
        this._container = container;
        this._page = page;
        this._detailsEl = detailsEl;
        this._onSelect = onSelect;
        this._storageKey = storageKey || `${page}_preset`;
        this._presets = [];
        this._selectedId = null;
        this._btnRow = null;
        this._advBtn = null;
    }

    async init() {
        try {
            const resp = await fetch('/api/presets');
            if (!resp.ok) return;
            const data = await resp.json();
            this._presets = data[this._page] || [];
        } catch (e) {
            console.warn('Failed to load presets:', e);
            return;
        }

        if (this._presets.length === 0) {
            this._container.style.display = 'none';
            return;
        }

        this._render();

        const saved = localStorage.getItem(this._storageKey);
        if (saved && this._presets.some(p => p.id === saved)) {
            this.select(saved, false);
        } else {
            this.select(this._presets[0].id, false);
        }
    }

    _render() {
        const wrap = document.createElement('div');
        wrap.className = 'preset-selector-wrap';

        const header = document.createElement('div');
        header.className = 'preset-header';
        header.innerHTML = `
            <span class="preset-title">Preset System Prompt</span>
            <span class="preset-subtitle">Controls response language, voice style, rhythm and timbre. Customizable via Advanced. You can customize the reference audio and system prompt in advanced settings. More presets are coming soon.</span>
        `;
        wrap.appendChild(header);

        const row = document.createElement('div');
        row.className = 'preset-row';

        const btnRow = document.createElement('div');
        btnRow.className = 'preset-btn-row';
        for (const preset of this._presets) {
            const btn = document.createElement('button');
            btn.className = 'preset-btn';
            btn.dataset.presetId = preset.id;
            btn.textContent = preset.name;
            btn.title = preset.description || preset.name;
            btn.addEventListener('click', () => this.select(preset.id, true));
            btnRow.appendChild(btn);
        }
        row.appendChild(btnRow);

        const advBtn = document.createElement('button');
        advBtn.className = 'preset-adv-btn';
        advBtn.textContent = 'Advanced ▾';
        advBtn.title = 'Show/hide system prompt details for customization';
        advBtn.addEventListener('click', () => this._toggleAdvanced());
        this._advBtn = advBtn;
        row.appendChild(advBtn);

        wrap.appendChild(row);
        this._container.appendChild(wrap);
        this._btnRow = btnRow;
    }

    select(presetId, isUserAction) {
        if (this._selectedId === presetId && !isUserAction) return;

        const preset = this._presets.find(p => p.id === presetId);
        if (!preset) return;

        this._selectedId = presetId;
        localStorage.setItem(this._storageKey, presetId);

        for (const btn of this._btnRow.querySelectorAll('.preset-btn')) {
            btn.classList.toggle('active', btn.dataset.presetId === presetId);
        }

        if (this._detailsEl) {
            this._detailsEl.removeAttribute('open');
        }
        if (this._advBtn) {
            this._advBtn.textContent = 'Advanced ▾';
        }

        if (this._onSelect) {
            this._onSelect(preset);
        }
    }

    getSelectedId() {
        return this._selectedId;
    }

    _toggleAdvanced() {
        if (!this._detailsEl) return;
        if (this._detailsEl.hasAttribute('open')) {
            this._detailsEl.removeAttribute('open');
            this._advBtn.textContent = 'Advanced ▾';
        } else {
            this._detailsEl.setAttribute('open', '');
            this._advBtn.textContent = 'Advanced ▴';
        }
    }
}

(function injectPresetCSS() {
    if (document.getElementById('preset-selector-css')) return;
    const style = document.createElement('style');
    style.id = 'preset-selector-css';
    style.textContent = `
.preset-selector-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.preset-header {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.preset-title {
    font-size: 13px;
    font-weight: 600;
    color: #444;
}
.preset-subtitle {
    font-size: 11px;
    color: #aaa;
    line-height: 1.4;
}
.preset-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.preset-btn-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}
.preset-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border: 1.5px solid #e0ddd8;
    border-radius: 20px;
    background: #fff;
    font-size: 13px;
    font-weight: 500;
    color: #555;
    cursor: pointer;
    transition: all 0.15s ease;
    font-family: inherit;
    line-height: 1.3;
}
.preset-btn:hover {
    border-color: #bbb;
    background: #fafaf8;
}
.preset-btn.active {
    border-color: #2d2d2d;
    background: #2d2d2d;
    color: #fff;
}
.preset-adv-btn {
    padding: 4px 10px;
    border: none;
    background: transparent;
    font-size: 11px;
    color: #999;
    cursor: pointer;
    font-family: inherit;
    transition: color 0.15s;
    white-space: nowrap;
}
.preset-adv-btn:hover {
    color: #666;
}
`;
    document.head.appendChild(style);
})();

window.PresetSelector = PresetSelector;
