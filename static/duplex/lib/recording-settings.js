/**
 * lib/recording-settings.js — Self-contained recording settings panel
 *
 * Provides a draggable floating panel for configuring video recording parameters:
 *   - Format selection (auto-detected via MediaRecorder.isTypeSupported)
 *   - Video quality / bitrate
 *
 * Usage:
 *   import { RecordingSettings } from './recording-settings.js';
 *   const recSettings = new RecordingSettings(document.getElementById('btnRecSettings'));
 *   // When starting recording:
 *   await recorder.start(recSettings.getSettings());
 */

/** 所有候选 MIME 类型，按优先级排列 */
const FORMAT_CANDIDATES = [
    { mime: 'video/webm;codecs=vp9,opus',  label: 'WebM (VP9 + Opus)',   category: 'video' },
    { mime: 'video/webm;codecs=vp8,opus',  label: 'WebM (VP8 + Opus)',   category: 'video' },
    { mime: 'video/webm;codecs=h264,opus', label: 'WebM (H.264 + Opus)', category: 'video' },
    { mime: 'video/webm',                  label: 'WebM',                 category: 'video' },
    { mime: 'video/mp4',                   label: 'MP4 (H.264 + AAC)',    category: 'video' },
    { mime: 'audio/webm;codecs=opus',      label: 'Audio WebM (Opus)',    category: 'audio' },
    { mime: 'audio/mp4',                   label: 'Audio MP4 (AAC)',      category: 'audio' },
];

const QUALITY_OPTIONS = [
    { value: 1_000_000, label: 'Low (1 Mbps)' },
    { value: 2_000_000, label: 'Medium (2 Mbps)' },
    { value: 4_000_000, label: 'High (4 Mbps)' },
    { value: 8_000_000, label: 'Ultra (8 Mbps)' },
];

const DEFAULT_QUALITY = 2_000_000;

export class RecordingSettings {
    /**
     * @param {HTMLElement} triggerBtn - 齿轮按钮，点击 toggle 面板
     * @param {string} [storageKey='omni_rec_settings'] - localStorage key
     */
    constructor(triggerBtn, storageKey = 'omni_rec_settings') {
        this._triggerBtn = triggerBtn;
        this._storageKey = storageKey;
        this._visible = false;
        this._selectedMime = 'auto';
        this._videoBps = DEFAULT_QUALITY;
        this._subtitleEnabled = true;
        this._subtitleHeight = 25;         // percent of video height from bottom
        this._subtitleOpacityBottom = 90;  // percent
        this._subtitleOpacityTop = 15;     // percent

        // 从 localStorage 恢复（在构建 DOM 之前，这样 DOM 使用恢复后的值）
        this._restoreFromStorage();

        // 检测格式
        this._formats = this._detectFormats();

        // 构建 DOM
        this._panel = this._buildPanel();
        document.body.appendChild(this._panel);

        // 绑定事件
        this._triggerBtn.addEventListener('click', () => this._togglePanel());

        // 点击面板外关闭
        this._outsideClickHandler = (e) => {
            if (this._visible
                && !this._panel.contains(e.target)
                && !this._triggerBtn.contains(e.target)) {
                this._hidePanel();
            }
        };
        document.addEventListener('mousedown', this._outsideClickHandler);
    }

    /**
     * 获取当前录制设置
     * @returns {{ mimeType: string, videoBitsPerSecond: number }}
     */
    getSettings() {
        return {
            mimeType: this._selectedMime,
            videoBitsPerSecond: this._videoBps,
            subtitle: this._subtitleEnabled,
            subtitleHeight: this._subtitleHeight,
            subtitleOpacityBottom: this._subtitleOpacityBottom / 100,
            subtitleOpacityTop: this._subtitleOpacityTop / 100,
        };
    }

    /** 销毁组件，移除 DOM 和事件 */
    destroy() {
        document.removeEventListener('mousedown', this._outsideClickHandler);
        this._panel.remove();
    }

    /** 保存当前设置到 localStorage */
    _saveToStorage() {
        try {
            localStorage.setItem(this._storageKey, JSON.stringify({
                mime: this._selectedMime,
                bps: this._videoBps,
                sub: this._subtitleEnabled,
                subH: this._subtitleHeight,
                subOB: this._subtitleOpacityBottom,
                subOT: this._subtitleOpacityTop,
            }));
        } catch (_) {}
    }

    /** 从 localStorage 恢复设置（在 _buildPanel 之前调用） */
    _restoreFromStorage() {
        try {
            const raw = localStorage.getItem(this._storageKey);
            if (!raw) return;
            const d = JSON.parse(raw);
            if (d.mime != null) this._selectedMime = d.mime;
            if (d.bps != null) this._videoBps = Number(d.bps);
            if (d.sub != null) this._subtitleEnabled = !!d.sub;
            if (d.subH != null) this._subtitleHeight = Number(d.subH);
            if (d.subOB != null) this._subtitleOpacityBottom = Number(d.subOB);
            if (d.subOT != null) this._subtitleOpacityTop = Number(d.subOT);
            console.log(`[RecordingSettings] restored from "${this._storageKey}"`);
        } catch (_) {}
    }

    /** 清除 localStorage 中的录制设置 */
    clearStorage() {
        try { localStorage.removeItem(this._storageKey); } catch (_) {}
    }

    // ==================== Format Detection ====================

    _detectFormats() {
        const hasMediaRecorder = typeof MediaRecorder !== 'undefined'
            && typeof MediaRecorder.isTypeSupported === 'function';

        // Chrome's MP4 muxer has a known bug: audio from Web Audio API
        // (AudioWorklet → MediaStreamDestination) is not written into
        // the MP4 container. WebM works fine. Safari MP4 is unaffected.
        const isChromium = /Chrome\//.test(navigator.userAgent);

        return FORMAT_CANDIDATES.map((fmt) => {
            const supported = hasMediaRecorder && MediaRecorder.isTypeSupported(fmt.mime);
            const chromeMp4NoAudio = isChromium && fmt.mime.includes('mp4') && supported;
            return {
                ...fmt,
                supported,
                warning: chromeMp4NoAudio ? 'No audio on Chrome' : null,
            };
        });
    }

    // ==================== DOM Construction ====================

    _buildPanel() {
        const panel = document.createElement('div');
        panel.className = 'rec-settings-panel';

        // Header (draggable)
        const header = document.createElement('div');
        header.className = 'rec-settings-header';
        header.innerHTML = `
            <span class="rec-settings-title">Recording Settings</span>
            <button class="rec-settings-close">✕</button>
        `;
        panel.appendChild(header);

        header.querySelector('.rec-settings-close')
            .addEventListener('click', () => this._hidePanel());
        this._makeDraggable(panel, header);

        // Body
        const body = document.createElement('div');
        body.className = 'rec-settings-body';

        // Format section
        body.appendChild(this._buildFormatSection());

        // Quality section
        body.appendChild(this._buildQualitySection());

        // Subtitle section
        body.appendChild(this._buildSubtitleSection());

        // Detection info footer
        const info = document.createElement('div');
        info.className = 'rec-settings-info';
        const supportedCount = this._formats.filter(f => f.supported).length;
        const browser = this._detectBrowser();
        info.textContent = `${browser} · ${supportedCount} format${supportedCount !== 1 ? 's' : ''} available`;
        body.appendChild(info);

        panel.appendChild(body);
        return panel;
    }

    _buildFormatSection() {
        const section = document.createElement('div');
        section.className = 'rec-settings-section';

        const label = document.createElement('div');
        label.className = 'rec-settings-label';
        label.textContent = 'Format';
        section.appendChild(label);

        const options = document.createElement('div');
        options.className = 'rec-fmt-options';

        // Auto option
        options.appendChild(this._createFormatRadio('auto', 'Auto (recommended)', true, this._selectedMime === 'auto'));

        // Candidate formats
        for (const fmt of this._formats) {
            options.appendChild(
                this._createFormatRadio(fmt.mime, fmt.label, fmt.supported, this._selectedMime === fmt.mime, fmt.warning)
            );
        }

        section.appendChild(options);
        return section;
    }

    /**
     * @param {string} value
     * @param {string} labelText
     * @param {boolean} supported
     * @param {boolean} checked
     * @param {string|null} [warning=null] - 警告文案（如 Chrome MP4 无音频）
     */
    _createFormatRadio(value, labelText, supported, checked, warning = null) {
        const row = document.createElement('label');
        row.className = 'rec-fmt-option'
            + (supported ? '' : ' disabled')
            + (warning ? ' has-warning' : '');

        const radio = document.createElement('input');
        radio.type = 'radio';
        radio.name = 'recFormat';
        radio.value = value;
        radio.checked = checked;
        radio.disabled = !supported;
        radio.addEventListener('change', () => {
            if (radio.checked) { this._selectedMime = value; this._saveToStorage(); }
        });

        const dot = document.createElement('span');
        dot.className = 'rec-fmt-dot' + (supported ? (warning ? ' warn' : ' on') : '');

        const text = document.createElement('span');
        text.className = 'rec-fmt-text';
        text.textContent = labelText;

        row.appendChild(radio);
        row.appendChild(dot);
        row.appendChild(text);

        if (warning) {
            const warn = document.createElement('span');
            warn.className = 'rec-fmt-warn';
            warn.textContent = warning;
            row.appendChild(warn);
        }

        return row;
    }

    _buildQualitySection() {
        const section = document.createElement('div');
        section.className = 'rec-settings-section';

        const label = document.createElement('div');
        label.className = 'rec-settings-label';
        label.textContent = 'Video Quality';
        section.appendChild(label);

        const select = document.createElement('select');
        select.className = 'rec-settings-select';
        for (const opt of QUALITY_OPTIONS) {
            const option = document.createElement('option');
            option.value = String(opt.value);
            option.textContent = opt.label;
            if (opt.value === this._videoBps) option.selected = true;
            select.appendChild(option);
        }
        select.addEventListener('change', () => {
            this._videoBps = Number(select.value);
            this._saveToStorage();
        });

        section.appendChild(select);
        return section;
    }

    _buildSubtitleSection() {
        const section = document.createElement('div');
        section.className = 'rec-settings-section';

        // Toggle row
        const row = document.createElement('label');
        row.className = 'rec-subtitle-toggle';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = this._subtitleEnabled;
        checkbox.addEventListener('change', () => {
            this._subtitleEnabled = checkbox.checked;
            detailsBox.style.display = checkbox.checked ? 'block' : 'none';
            this._saveToStorage();
        });

        const text = document.createElement('span');
        text.className = 'rec-subtitle-text';
        text.textContent = 'Burn subtitles into video';

        row.appendChild(checkbox);
        row.appendChild(text);
        section.appendChild(row);

        // Details container (collapsible with checkbox)
        const detailsBox = document.createElement('div');
        detailsBox.className = 'rec-subtitle-details';
        if (!this._subtitleEnabled) detailsBox.style.display = 'none';

        // Height row
        detailsBox.appendChild(this._buildSubtitleRow(
            'Height',
            '% from bottom',
            this._subtitleHeight, 5, 60, 5,
            (v) => { this._subtitleHeight = v; this._saveToStorage(); },
        ));

        // Opacity row (bottom ↗ top)
        const opacityRow = document.createElement('div');
        opacityRow.className = 'rec-subtitle-param-row';

        const oLabel = document.createElement('span');
        oLabel.className = 'rec-subtitle-param-label';
        oLabel.textContent = 'Opacity %';

        const botInput = this._buildNumInput(this._subtitleOpacityBottom, 10, 100, 10, (v) => {
            this._subtitleOpacityBottom = v;
            this._saveToStorage();
        });
        const arrow = document.createElement('span');
        arrow.textContent = '↗';
        arrow.style.cssText = 'font-size:12px;color:#666;';
        const topInput = this._buildNumInput(this._subtitleOpacityTop, 0, 100, 5, (v) => {
            this._subtitleOpacityTop = v;
            this._saveToStorage();
        });

        opacityRow.appendChild(oLabel);
        opacityRow.appendChild(botInput);
        opacityRow.appendChild(arrow);
        opacityRow.appendChild(topInput);
        detailsBox.appendChild(opacityRow);

        section.appendChild(detailsBox);
        return section;
    }

    /**
     * Build a labeled number input row for subtitle settings.
     * @param {string} label
     * @param {string} suffix
     * @param {number} value
     * @param {number} min
     * @param {number} max
     * @param {number} step
     * @param {function(number): void} onChange
     * @returns {HTMLElement}
     */
    _buildSubtitleRow(label, suffix, value, min, max, step, onChange) {
        const row = document.createElement('div');
        row.className = 'rec-subtitle-param-row';

        const lbl = document.createElement('span');
        lbl.className = 'rec-subtitle-param-label';
        lbl.textContent = label;

        const input = this._buildNumInput(value, min, max, step, onChange);

        const sfx = document.createElement('span');
        sfx.className = 'rec-subtitle-param-suffix';
        sfx.textContent = suffix;

        row.appendChild(lbl);
        row.appendChild(input);
        row.appendChild(sfx);
        return row;
    }

    /**
     * Build a compact number input element.
     * @param {number} value
     * @param {number} min
     * @param {number} max
     * @param {number} step
     * @param {function(number): void} onChange
     * @returns {HTMLInputElement}
     */
    _buildNumInput(value, min, max, step, onChange) {
        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'rec-settings-num';
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        input.value = String(value);
        input.addEventListener('change', () => {
            const v = Math.max(min, Math.min(max, parseInt(input.value, 10) || value));
            input.value = String(v);
            onChange(v);
        });
        return input;
    }

    // ==================== Panel Visibility ====================

    _togglePanel() {
        if (this._visible) {
            this._hidePanel();
        } else {
            this._showPanel();
        }
    }

    _showPanel() {
        // Position near the trigger button
        const rect = this._triggerBtn.getBoundingClientRect();
        this._panel.style.display = 'block';
        this._panel.style.left = rect.left + 'px';
        this._panel.style.top = (rect.bottom + 8) + 'px';

        // Clamp to viewport
        requestAnimationFrame(() => {
            const pr = this._panel.getBoundingClientRect();
            if (pr.right > window.innerWidth - 8) {
                this._panel.style.left = Math.max(8, window.innerWidth - pr.width - 8) + 'px';
            }
            if (pr.bottom > window.innerHeight - 8) {
                this._panel.style.top = Math.max(8, rect.top - pr.height - 8) + 'px';
            }
        });

        this._visible = true;
    }

    _hidePanel() {
        this._panel.style.display = 'none';
        this._visible = false;
    }

    // ==================== Draggable ====================

    _makeDraggable(panel, handle) {
        let dragging = false, startX = 0, startY = 0, origX = 0, origY = 0;

        const onStart = (clientX, clientY, e) => {
            if (e.target.closest('.rec-settings-close')) return;
            dragging = true;
            startX = clientX; startY = clientY;
            const rect = panel.getBoundingClientRect();
            origX = rect.left; origY = rect.top;
            panel.style.right = 'auto';
        };

        handle.addEventListener('mousedown', (e) => {
            onStart(e.clientX, e.clientY, e);
            e.preventDefault();
        });
        document.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            panel.style.left = (origX + e.clientX - startX) + 'px';
            panel.style.top = (origY + e.clientY - startY) + 'px';
        });
        document.addEventListener('mouseup', () => { dragging = false; });

        handle.addEventListener('touchstart', (e) => {
            const t = e.touches[0];
            onStart(t.clientX, t.clientY, e);
        }, { passive: true });
        document.addEventListener('touchmove', (e) => {
            if (!dragging) return;
            const t = e.touches[0];
            panel.style.left = (origX + t.clientX - startX) + 'px';
            panel.style.top = (origY + t.clientY - startY) + 'px';
        }, { passive: true });
        document.addEventListener('touchend', () => { dragging = false; });
    }

    // ==================== Utils ====================

    _detectBrowser() {
        const ua = navigator.userAgent;
        if (/Firefox\/(\d+)/.test(ua)) return `Firefox ${RegExp.$1}`;
        if (/Edg\/(\d+)/.test(ua)) return `Edge ${RegExp.$1}`;
        if (/Chrome\/(\d+)/.test(ua)) return `Chrome ${RegExp.$1}`;
        if (/Safari\/(\d+)/.test(ua) && /Version\/(\d+)/.test(ua)) return `Safari ${RegExp.$1}`;
        return navigator.userAgent.split(/\s+/).pop() || 'Unknown';
    }
}
