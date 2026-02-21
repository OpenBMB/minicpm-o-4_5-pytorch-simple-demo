/**
 * UserContentEditor — User Message Content List 编辑器
 *
 * 将 User 消息建模为一个 content list，每个 item 可以是 text、audio 或 image。
 * 用户可以任意添加、删除、重排项目，支持录音和上传。
 *
 * 特性：
 *   - 空列表零摩擦：直接打字 → 创建 text item；🎤 / Space → 录音
 *   - T/A/I badge 标识类型（text / audio / image）
 *   - 拖拽 + ↑↓ 按钮排序
 *   - Space 键录音（tap toggle / hold push-to-talk）+ ESC 取消
 *   - 全局录音锁（同时只允许一个录音）
 *   - 图片上传 + 缩略图预览 + 删除
 *
 * 用法：
 *   const editor = new UserContentEditor(container, {
 *       onChange(items) { ... },
 *   });
 *   editor.setItems([...]);
 *   const items = editor.getItems();
 */

/* ── CSS 注入 ── */
(function injectCSS() {
    if (document.getElementById('uce-css')) return;
    const style = document.createElement('style');
    style.id = 'uce-css';
    style.textContent = `
/* ── Audio shared ── */
.uce-audio-action-zone {
    display: flex; gap: 6px; align-items: stretch;
}
.uce-audio-action-btn {
    display: flex; align-items: center; gap: 6px;
    padding: 8px 14px;
    border: 1px dashed #d5d5d0;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
    font-size: 12px;
    color: #999;
    background: transparent;
    flex: 1;
    justify-content: center;
}
.uce-audio-action-btn:hover { border-color: #999; color: #666; background: #fafaf8; }
.uce-audio-action-btn.uce-record-btn:hover { border-color: #cf222e; color: #cf222e; background: #fff5f5; }
.uce-audio-info {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: #888;
}
.uce-audio-info .name { color: #2d2d2d; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.uce-audio-play-btn {
    background: none; border: none; color: #d4a574; cursor: pointer;
    font-size: 14px; padding: 2px;
}
.uce-audio-remove-btn {
    background: none; border: none; color: #ccc; cursor: pointer;
    font-size: 11px; padding: 2px; transition: color 0.12s;
}
.uce-audio-remove-btn:hover { color: #cf222e; }

/* ── Image UI ── */
.uce-image-action-zone {
    display: flex; gap: 6px; align-items: stretch;
}
.uce-image-action-btn {
    display: flex; align-items: center; gap: 6px;
    padding: 8px 14px;
    border: 1px dashed #d5d5d0;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
    font-size: 12px;
    color: #999;
    background: transparent;
    flex: 1;
    justify-content: center;
}
.uce-image-action-btn:hover { border-color: #4070a0; color: #4070a0; background: #f5f8fb; }
.uce-image-info {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: #888;
}
.uce-image-thumb {
    width: 48px; height: 48px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #e5e5e0;
    flex-shrink: 0;
}
.uce-image-info .name { color: #2d2d2d; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.uce-image-remove-btn {
    background: none; border: none; color: #ccc; cursor: pointer;
    font-size: 11px; padding: 2px; transition: color 0.12s;
}
.uce-image-remove-btn:hover { color: #cf222e; }

/* ── Recording UI ── */
.uce-recording-inline {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px;
    background: #fff5f5;
    border: 1px solid #cf222e;
    border-radius: 6px;
    animation: uce-rec-pulse 1.5s ease-in-out infinite;
}
@keyframes uce-rec-pulse {
    0%, 100% { border-color: #cf222e; box-shadow: 0 0 0 0 rgba(207,34,46,0); }
    50% { border-color: rgba(207,34,46,0.5); box-shadow: 0 0 8px 0 rgba(207,34,46,0.15); }
}
.uce-rec-dot {
    width: 10px; height: 10px;
    background: #cf222e;
    border-radius: 50%;
    flex-shrink: 0;
    animation: uce-rec-dot-pulse 1s ease-in-out infinite;
}
@keyframes uce-rec-dot-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}
.uce-rec-label { color: #cf222e; font-size: 12px; font-weight: 500; }
.uce-rec-timer {
    color: #cf222e; font-size: 14px;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    min-width: 36px;
}
.uce-rec-keys-hint { font-size: 10px; color: #ccc; margin-left: auto; white-space: nowrap; }
.uce-rec-stop-btn {
    padding: 4px 14px;
    border: 1px solid #cf222e;
    border-radius: 5px;
    background: rgba(207,34,46,0.08);
    color: #cf222e;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.12s;
}
.uce-rec-stop-btn:hover { background: rgba(207,34,46,0.15); }
.uce-rec-cancel-btn {
    padding: 4px 10px;
    border: 1px solid #d5d5d0;
    border-radius: 5px;
    background: transparent;
    color: #999;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.12s;
}
.uce-rec-cancel-btn:hover { border-color: #999; color: #666; }

/* ═══ Editor container ═══ */
.uce-wrap { min-height: 40px; outline: none; }

/* ── Compose 模式：融入外层 input-wrap，无自身边框 ── */
.uce-wrap.uce-compose { min-height: 0; }
.uce-compose .uce-empty { padding: 0; gap: 4px; }
.uce-compose .uce-empty-ta {
    border: none !important;
    background: transparent !important;
    padding: 8px 10px 4px !important;
    border-radius: 0 !important;
}
.uce-compose .uce-empty-ta:focus { border: none !important; }
.uce-compose .uce-empty-bottom { padding: 0 6px 2px; }
.uce-compose .uce-items { padding: 4px 6px; }

/* ── Empty state ── */
.uce-empty {
    display: flex; flex-direction: column; gap: 8px;
    padding: 4px 0;
}
.uce-empty-ta {
    width: 100%;
    background: #fff;
    border: 1px solid #e5e5e0;
    border-radius: 6px;
    color: #2d2d2d;
    padding: 10px 12px;
    font-size: 14px;
    font-family: inherit;
    line-height: 1.5;
    resize: none;
    min-height: 40px;
    outline: none;
    transition: border-color 0.15s;
}
.uce-empty-ta:focus { border-color: #999; }
.uce-empty-ta::placeholder { color: #ccc; }
.uce-empty-bottom {
    display: flex; align-items: center; justify-content: space-between;
}
.uce-empty-adds { display: flex; gap: 6px; }
.uce-empty-add-btn {
    font-size: 11px; padding: 3px 10px;
    border: 1px solid #e5e5e0; border-radius: 4px;
    background: transparent; color: #bbb;
    cursor: pointer; transition: all 0.12s;
}
.uce-empty-add-btn:hover { border-color: #999; color: #888; }
.uce-empty-rec-zone {
    display: flex; flex-direction: column; align-items: center; gap: 2px;
}
.uce-empty-rec-btn {
    width: 40px; height: 40px;
    border-radius: 50%;
    border: 1px solid #d5d5d0;
    background: #fff;
    cursor: pointer;
    font-size: 18px;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s;
}
.uce-empty-rec-btn:hover { border-color: #cf222e; background: #fff5f5; }
.uce-empty-rec-hint { font-size: 9px; color: #ccc; letter-spacing: 0.5px; }

/* ── Full state items ── */
.uce-items { display: flex; flex-direction: column; gap: 2px; }
.uce-item {
    display: flex; align-items: flex-start; gap: 6px;
    padding: 5px 6px;
    border-radius: 6px;
    position: relative;
    transition: background 0.1s;
    border: 1px solid transparent;
}
.uce-item:hover { background: #fafaf8; border-color: #e8e6e0; }
.uce-item.uce-dragging { opacity: 0.4; border-color: #d4a574; }
.uce-drag {
    width: 14px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    color: #e5e5e0; font-size: 11px;
    cursor: grab; padding-top: 5px;
    opacity: 0; transition: opacity 0.1s;
    user-select: none;
}
.uce-item:hover .uce-drag { opacity: 1; color: #bbb; }
.uce-badge {
    font-size: 9px; font-weight: 700;
    padding: 2px 6px; border-radius: 3px;
    flex-shrink: 0; margin-top: 4px;
    user-select: none; letter-spacing: 0.3px;
}
.uce-badge-text { background: #e8e4de; color: #888; }
.uce-badge-audio { background: #f0e6d8; color: #a07040; }
.uce-badge-image { background: #dde8f0; color: #4070a0; }
.uce-content { flex: 1; min-width: 0; }
.uce-textarea {
    width: 100%;
    background: transparent;
    border: none;
    color: #2d2d2d;
    padding: 3px 6px;
    font-size: 13px;
    font-family: inherit;
    line-height: 1.5;
    resize: none;
    min-height: 26px;
    outline: none;
}
.uce-textarea:focus { background: #f5f5f0; border-radius: 3px; }
.uce-textarea::placeholder { color: #ccc; }

/* ── Item actions (hover) ── */
.uce-item-actions {
    display: flex; align-items: center; gap: 1px;
    flex-shrink: 0; margin-top: 2px;
    opacity: 0; transition: opacity 0.1s;
}
.uce-item:hover .uce-item-actions { opacity: 1; }
.uce-act {
    width: 20px; height: 20px;
    border: none; border-radius: 3px;
    background: transparent; color: #bbb;
    cursor: pointer; font-size: 11px;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.1s; padding: 0;
}
.uce-act:hover { background: #f0f0eb; color: #666; }
.uce-act:disabled { opacity: 0.2; cursor: default; }
.uce-act:disabled:hover { background: transparent; }
.uce-act.uce-act-remove:hover { background: #ffe3e6; color: #cf222e; }

/* ── Add row ── */
.uce-add-row {
    display: flex; align-items: center; justify-content: center; flex-wrap: wrap;
    gap: 6px; margin-top: 6px; padding-top: 6px;
    border-top: 1px solid #e8e6e0;
}
.uce-add-group { display: flex; align-items: center; gap: 6px; }
.uce-add-btn {
    font-size: 11px; padding: 4px 10px;
    border: 1px dashed #e5e5e0; border-radius: 4px;
    background: transparent; color: #bbb;
    cursor: pointer; transition: all 0.12s;
}
.uce-add-btn:hover { border-color: #999; color: #666; background: #fafaf8; }
.uce-act-group { display: flex; align-items: center; gap: 6px; }
.uce-act-btn {
    font-size: 12px; padding: 3px 12px;
    border: 1px solid #d5d5d0; border-radius: 6px;
    background: #fff; color: #555;
    cursor: pointer; transition: all 0.12s;
}
.uce-act-btn:hover { opacity: 0.8; }
.uce-act-primary { background: #2d2d2d; color: #fff; border-color: #2d2d2d; }
.uce-rec-btn-add {
    border-color: rgba(207,34,46,0.2) !important; color: rgba(207,34,46,0.5) !important;
}
.uce-rec-btn-add:hover {
    border-color: #cf222e !important; color: #cf222e !important;
    background: #fff5f5 !important;
}
.uce-rec-hint { font-size: 8px; color: #ccc; letter-spacing: 0.5px; }
`;
    document.head.appendChild(style);
})();


// ═══════════════════════════════════════════════════════════════
// Global recording state
// ═══════════════════════════════════════════════════════════════

var _uceRecordingCount = 0;
/** Global recording lock — only one recording at a time (one microphone) */
var _uceGlobalRecordingActive = false;
/** Reference to the active _UCERecordingSession instance */
var _uceActiveRecordingSession = null;
/** The UserContentEditor that last received user interaction */
var _uceLastActiveEditor = null;

// ── Space key: tap = toggle, hold = push-to-talk; ESC = cancel ──
var _uceSpaceHeld = false;
var _uceSpaceDownTime = 0;
const _UCE_HOLD_THRESHOLD_MS = 300;

(function _uceInitRecordingShortcuts() {

    function isInputFocused() {
        const tag = document.activeElement?.tagName;
        return tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT';
    }

    document.addEventListener('keydown', (e) => {
        // ESC: cancel recording (always, even when input focused)
        if (e.key === 'Escape' && _uceGlobalRecordingActive && _uceActiveRecordingSession) {
            e.preventDefault();
            _uceActiveRecordingSession.cancel();
            _uceSpaceHeld = false;
            return;
        }

        // Space: only when not in any text input, not repeat
        // (empty UCE textarea is handled directly by the textarea's own keydown)
        if (e.code !== 'Space' || isInputFocused() || e.repeat) return;
        if (!_uceLastActiveEditor) return;
        if (!_uceLastActiveEditor.wrap.offsetParent) return;
        e.preventDefault();

        if (!_uceGlobalRecordingActive) {
            _uceSpaceHeld = true;
            _uceSpaceDownTime = Date.now();
            _uceLastActiveEditor._startRecording();
        } else if (!_uceSpaceHeld && _uceActiveRecordingSession) {
            _uceActiveRecordingSession.stop();
        }
    });

    document.addEventListener('keyup', (e) => {
        if (e.code !== 'Space' || !_uceSpaceHeld) return;
        e.preventDefault();
        _uceSpaceHeld = false;

        if (!_uceGlobalRecordingActive || !_uceActiveRecordingSession) return;

        const holdMs = Date.now() - _uceSpaceDownTime;
        if (holdMs >= _UCE_HOLD_THRESHOLD_MS) {
            _uceActiveRecordingSession.stop();
        }
    });
})();


// ═══════════════════════════════════════════════════════════════
// _UCERecordingSession (internal)
// ═══════════════════════════════════════════════════════════════

class _UCERecordingSession {
    constructor(container, callbacks) {
        this.container = container;
        this.callbacks = callbacks;
        this.mediaRecorder = null;
        this.chunks = [];
        this.startTime = 0;
        this.timerInterval = null;
        this.stream = null;
        this._cancelled = false;
        this._start();
    }

    async _start() {
        if (_uceGlobalRecordingActive) {
            this.callbacks.onCancel();
            return;
        }
        _uceGlobalRecordingActive = true;
        _uceActiveRecordingSession = this;

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (e) {
            _uceGlobalRecordingActive = false;
            _uceActiveRecordingSession = null;
            alert('Microphone access denied. Please allow microphone access.');
            this.callbacks.onCancel();
            return;
        }

        this.chunks = [];
        this.mediaRecorder = new MediaRecorder(this.stream);

        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this.chunks.push(e.data);
        };

        this.mediaRecorder.onstop = () => {
            this._stopTimer();
            this._stopStream();
            _uceGlobalRecordingActive = false;
            _uceActiveRecordingSession = null;
            if (this._cancelled || this.chunks.length === 0) {
                this.callbacks.onCancel();
                return;
            }
            const blob = new Blob(this.chunks, { type: 'audio/webm' });
            const objectUrl = URL.createObjectURL(blob);
            const duration = (Date.now() - this.startTime) / 1000;
            _uceRecordingCount++;
            const name = `recording_${String(_uceRecordingCount).padStart(3, '0')}.webm`;
            this.callbacks.onDone(objectUrl, name, duration, blob);
        };

        this.mediaRecorder.start(100);
        this.startTime = Date.now();
        this._renderUI();
        this._startTimer();
    }

    _renderUI() {
        this.container.innerHTML = '';
        const row = document.createElement('div');
        row.className = 'uce-recording-inline';
        row.innerHTML = `
            <div class="uce-rec-dot"></div>
            <span class="uce-rec-label">Recording</span>
            <span class="uce-rec-timer">0:00</span>
            <span class="uce-rec-keys-hint">Space stop · ESC cancel</span>
            <button class="uce-rec-cancel-btn">Cancel</button>
            <button class="uce-rec-stop-btn">⏹ Stop</button>
        `;
        this._timerEl = row.querySelector('.uce-rec-timer');
        row.querySelector('.uce-rec-stop-btn').addEventListener('click', () => this.stop());
        row.querySelector('.uce-rec-cancel-btn').addEventListener('click', () => this.cancel());
        this.container.appendChild(row);
    }

    _startTimer() {
        this.timerInterval = setInterval(() => {
            const elapsed = (Date.now() - this.startTime) / 1000;
            const mins = Math.floor(elapsed / 60);
            const secs = Math.floor(elapsed % 60);
            if (this._timerEl) {
                this._timerEl.textContent = `${mins}:${String(secs).padStart(2, '0')}`;
            }
        }, 200);
    }

    _stopTimer() {
        if (this.timerInterval) { clearInterval(this.timerInterval); this.timerInterval = null; }
    }

    _stopStream() {
        if (this.stream) { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
    }

    stop() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
    }

    cancel() {
        this._cancelled = true;
        this._stopTimer();
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        } else {
            _uceGlobalRecordingActive = false;
            _uceActiveRecordingSession = null;
            this._stopStream();
            this.callbacks.onCancel();
        }
    }
}


// ═══════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════

function _uceDeepCopy(items) {
    // 浅拷贝每个 item，保留 _blob / file 等不可序列化的引用
    return items.map(it => ({ ...it }));
}

function _uceAutoResize(textarea) {
    textarea.style.height = '0';
    textarea.style.height = Math.max(28, textarea.scrollHeight) + 'px';
}

function _uceEscapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

/**
 * Create audio UI for an item.
 * States: recording → has-audio (info + play) → empty (upload/record)
 */
function _uceCreateAudioUI(item, idx, onChange) {
    const wrap = document.createElement('div');

    if (item._recording) {
        new _UCERecordingSession(wrap, {
            onDone: (objectUrl, name, duration, blob) => {
                item._recording = false;
                item.objectUrl = objectUrl;
                item.name = name;
                item.duration = duration;
                item._blob = blob; // 保留 blob 供后续转换为 base64
                onChange();
            },
            onCancel: () => {
                item._recording = false;
                onChange();
            }
        });
    } else if (item.name || item.data) {
        // Has audio — show info + play + remove
        wrap.className = 'uce-audio-info';

        const noteIcon = document.createElement('span');
        noteIcon.style.color = '#d4a574';
        noteIcon.innerHTML = '&#9835;';
        wrap.appendChild(noteIcon);

        const nameSpan = document.createElement('span');
        nameSpan.className = 'name';
        nameSpan.textContent = item.name || 'audio';
        wrap.appendChild(nameSpan);

        const durSpan = document.createElement('span');
        durSpan.style.color = '#bbb';
        durSpan.textContent = item.duration ? item.duration.toFixed(1) + 's' : '';
        wrap.appendChild(durSpan);

        const playBtn = document.createElement('button');
        playBtn.className = 'uce-audio-play-btn';
        playBtn.innerHTML = '&#9654;';
        playBtn.title = 'Play';
        let audioEl = null;
        playBtn.addEventListener('click', () => {
            if (audioEl && !audioEl.paused) {
                audioEl.pause(); audioEl.currentTime = 0;
                playBtn.innerHTML = '&#9654;';
                return;
            }
            const src = item.objectUrl || (item.file ? URL.createObjectURL(item.file) : null);
            if (!src) return;
            audioEl = new Audio(src);
            playBtn.innerHTML = '⏸';
            audioEl.play();
            audioEl.addEventListener('ended', () => { playBtn.innerHTML = '&#9654;'; });
            audioEl.addEventListener('pause', () => { playBtn.innerHTML = '&#9654;'; });
        });
        wrap.appendChild(playBtn);

        const removeBtn = document.createElement('button');
        removeBtn.className = 'uce-audio-remove-btn';
        removeBtn.innerHTML = '&#10005;';
        removeBtn.title = 'Remove audio';
        removeBtn.addEventListener('click', () => {
            item.file = null; item.name = ''; item.duration = 0; item.objectUrl = null; item.data = null; item._blob = null;
            onChange();
        });
        wrap.appendChild(removeBtn);
    } else {
        // Empty — Upload + Record
        wrap.className = 'uce-audio-action-zone';

        const uploadBtn = document.createElement('button');
        uploadBtn.className = 'uce-audio-action-btn';
        uploadBtn.innerHTML = '<span>📎</span> Upload';
        const fileInput = document.createElement('input');
        fileInput.type = 'file'; fileInput.accept = 'audio/*';
        fileInput.style.display = 'none';
        fileInput.addEventListener('change', (e) => {
            const f = e.target.files[0];
            if (!f) return;
            item.file = f;
            item.name = f.name;
            const url = URL.createObjectURL(f);
            item.objectUrl = url;
            const audio = new Audio(url);
            audio.addEventListener('loadedmetadata', () => { item.duration = audio.duration; onChange(); });
            audio.addEventListener('error', () => { item.duration = 0; onChange(); });
        });
        uploadBtn.appendChild(fileInput);
        uploadBtn.addEventListener('click', () => fileInput.click());
        wrap.appendChild(uploadBtn);

        const recordBtn = document.createElement('button');
        recordBtn.className = 'uce-audio-action-btn uce-record-btn';
        recordBtn.innerHTML = '<span>🎤</span> Record';
        recordBtn.addEventListener('click', () => {
            item._recording = true;
            onChange();
        });
        wrap.appendChild(recordBtn);
    }
    return wrap;
}


/**
 * Create image UI for an item.
 * States: has-image (thumbnail + info) → empty (upload)
 */
function _uceCreateImageUI(item, idx, onChange) {
    const wrap = document.createElement('div');

    if (item.name || item.data || item.objectUrl) {
        // Has image — show thumbnail + info + remove
        wrap.className = 'uce-image-info';

        const thumb = document.createElement('img');
        thumb.className = 'uce-image-thumb';
        thumb.alt = item.name || 'image';
        if (item.objectUrl) {
            thumb.src = item.objectUrl;
        } else if (item.data) {
            thumb.src = 'data:image/png;base64,' + item.data;
        }
        wrap.appendChild(thumb);

        const nameSpan = document.createElement('span');
        nameSpan.className = 'name';
        nameSpan.textContent = item.name || 'image';
        wrap.appendChild(nameSpan);

        const removeBtn = document.createElement('button');
        removeBtn.className = 'uce-image-remove-btn';
        removeBtn.innerHTML = '&#10005;';
        removeBtn.title = 'Remove image';
        removeBtn.addEventListener('click', () => {
            item.file = null; item.name = ''; item.objectUrl = null; item.data = null;
            onChange();
        });
        wrap.appendChild(removeBtn);
    } else {
        // Empty — Upload
        wrap.className = 'uce-image-action-zone';

        const uploadBtn = document.createElement('button');
        uploadBtn.className = 'uce-image-action-btn';
        uploadBtn.innerHTML = '<span>🖼</span> Upload Image';
        const fileInput = document.createElement('input');
        fileInput.type = 'file'; fileInput.accept = 'image/*';
        fileInput.style.display = 'none';
        fileInput.addEventListener('change', (e) => {
            const f = e.target.files[0];
            if (!f) return;
            item.file = f;
            item.name = f.name;
            item.objectUrl = URL.createObjectURL(f);
            onChange();
        });
        uploadBtn.appendChild(fileInput);
        uploadBtn.addEventListener('click', () => fileInput.click());
        wrap.appendChild(uploadBtn);
    }
    return wrap;
}


// ═══════════════════════════════════════════════════════════════
// UserContentEditor (public API)
// ═══════════════════════════════════════════════════════════════

class UserContentEditor {
    /**
     * @param {HTMLElement} container
     * @param {Object} [options]
     * @param {Function} [options.onChange] (items) => void
     * @param {Function} [options.onSubmit] () => void  — Enter 键提交（shift+enter 换行）
     * @param {string}   [options.placeholder] — 空状态 textarea 占位文本
     */
    constructor(container, options = {}) {
        this.container = container;
        this.onChange = options.onChange || (() => {});
        this.onSubmit = options.onSubmit || null;
        this.onCancel = options.onCancel || null;
        this.placeholder = options.placeholder || 'Type a message...';
        this.isCompose = !!options.compose;
        this.showImageBtn = options.showImageBtn !== false;
        this.items = [];
        this.wrap = document.createElement('div');
        this.wrap.className = 'uce-wrap' + (this.isCompose ? ' uce-compose' : '');
        this.wrap.tabIndex = -1; // 可聚焦但不在 tab 顺序中（用于接收 Enter 键）
        container.innerHTML = '';
        container.appendChild(this.wrap);
        this._dragIdx = -1;

        // Track focus for Space key routing
        this.wrap.addEventListener('focusin', () => { _uceLastActiveEditor = this; });
        this.wrap.addEventListener('click', () => { _uceLastActiveEditor = this; });

        // 全局键盘：Enter=submit, ESC=cancel（焦点不在 textarea 时）
        this.wrap.addEventListener('keydown', (e) => {
            // ESC → cancel（任何时候）
            if (e.key === 'Escape' && this.onCancel) {
                e.preventDefault();
                this.onCancel();
                return;
            }
            // Enter → submit（焦点不在 textarea/input 时，如只有 audio items）
            if (e.key === 'Enter' && !e.shiftKey && this.onSubmit && this.items.length > 0) {
                const tag = e.target?.tagName;
                if (tag !== 'TEXTAREA' && tag !== 'INPUT') {
                    e.preventDefault();
                    this.onSubmit();
                }
            }
        });

        // 初始渲染（空状态）
        this.render();
    }

    /** Set content list and re-render */
    setItems(items) {
        this.items = items.map(it => ({ ...it }));
        this.render();
    }

    /** Get a deep copy of current content list */
    getItems() { return _uceDeepCopy(this.items); }

    /** Programmatically append an item and re-render */
    addItem(item) {
        this.items.push({ ...item });
        this.render();
        this.onChange(this.getItems());
    }

    /** Clean up (call when removing editor from DOM) */
    destroy() {
        if (_uceLastActiveEditor === this) _uceLastActiveEditor = null;
    }

    render() {
        this.wrap.innerHTML = '';
        if (this.items.length === 0) {
            this._renderEmpty();
        } else {
            this._renderFull();
            // 如果没有 textarea（如只有 audio items），聚焦 wrap 使 Enter 可用
            if (!this.wrap.querySelector('textarea')) {
                requestAnimationFrame(() => this.wrap.focus());
            }
        }
    }

    // ── Empty state ──
    _renderEmpty() {
        const empty = document.createElement('div');
        empty.className = 'uce-empty';

        const ta = document.createElement('textarea');
        ta.className = 'uce-empty-ta';
        ta.placeholder = this.placeholder;
        ta.rows = 1;
        let composing = false;
        ta.addEventListener('compositionstart', () => { composing = true; });
        ta.addEventListener('compositionend', () => {
            composing = false;
            if (ta.value) this._promoteText(ta);
        });
        ta.addEventListener('input', () => {
            _uceAutoResize(ta);
            if (!composing && ta.value.length > 0) {
                this._promoteText(ta);
            }
        });
        ta.addEventListener('keydown', (e) => {
            // Space on empty textarea → start recording
            if (e.code === 'Space' && ta.value === '' && !composing && !e.repeat) {
                e.preventDefault();
                _uceLastActiveEditor = this;
                if (!_uceGlobalRecordingActive) {
                    _uceSpaceHeld = true;
                    _uceSpaceDownTime = Date.now();
                    this._startRecording();
                } else if (!_uceSpaceHeld && _uceActiveRecordingSession) {
                    _uceActiveRecordingSession.stop();
                }
                return;
            }
            // Enter 提交
            if (e.key === 'Enter' && !e.shiftKey && !composing && this.onSubmit) {
                e.preventDefault();
                if (ta.value.trim()) {
                    this._promoteText(ta);
                }
                this.onSubmit();
            }
        });
        empty.appendChild(ta);

        const bottom = document.createElement('div');
        bottom.className = 'uce-empty-bottom';

        const adds = document.createElement('div');
        adds.className = 'uce-empty-adds';
        const addAudioBtn = document.createElement('button');
        addAudioBtn.className = 'uce-empty-add-btn';
        addAudioBtn.textContent = '+ Audio';
        addAudioBtn.addEventListener('click', () => {
            this.items.push({ type: 'audio', file: null, name: '', duration: 0, objectUrl: null });
            this.render();
            this.onChange(this.getItems());
        });
        adds.appendChild(addAudioBtn);

        if (this.showImageBtn) {
            const addImageBtn = document.createElement('button');
            addImageBtn.className = 'uce-empty-add-btn';
            addImageBtn.textContent = '+ Image';
            addImageBtn.addEventListener('click', () => {
                this.items.push({ type: 'image', file: null, name: '', objectUrl: null, data: null });
                this.render();
                this.onChange(this.getItems());
            });
            adds.appendChild(addImageBtn);
        }
        bottom.appendChild(adds);

        const recZone = document.createElement('div');
        recZone.className = 'uce-empty-rec-zone';
        const recBtn = document.createElement('button');
        recBtn.className = 'uce-empty-rec-btn';
        recBtn.textContent = '🎤';
        recBtn.title = 'Tap Space: toggle record\nHold Space: push-to-talk\nESC: cancel';
        recBtn.addEventListener('click', () => this._startRecording());
        recZone.appendChild(recBtn);
        const recHint = document.createElement('span');
        recHint.className = 'uce-empty-rec-hint';
        recHint.textContent = 'Space';
        recZone.appendChild(recHint);
        bottom.appendChild(recZone);

        // compose 模式不显示 UCE 底栏（外部有 input-bottom）
        if (!this.isCompose) {
            empty.appendChild(bottom);
        }
        this.wrap.appendChild(empty);
        requestAnimationFrame(() => _uceAutoResize(ta));
    }

    _promoteText(ta) {
        const text = ta.value;
        this.items.push({ type: 'text', text });
        this.render();
        this.onChange(this.getItems());
        const newTa = this.wrap.querySelector('.uce-textarea');
        if (newTa) {
            newTa.focus();
            newTa.selectionStart = newTa.selectionEnd = text.length;
        }
    }

    _startRecording() {
        if (_uceGlobalRecordingActive) return;
        this.items.push({ type: 'audio', file: null, name: '', duration: 0, objectUrl: null, _recording: true });
        this.render();
        this.onChange(this.getItems());
    }

    // ── Full state ──
    _renderFull() {
        const list = document.createElement('div');
        list.className = 'uce-items';

        this.items.forEach((item, idx) => {
            const row = document.createElement('div');
            row.className = 'uce-item';
            row.draggable = true;
            row.dataset.idx = idx;

            // Drag handle
            const drag = document.createElement('div');
            drag.className = 'uce-drag';
            drag.textContent = '⠿';
            row.appendChild(drag);

            // Badge
            const badge = document.createElement('span');
            badge.className = `uce-badge uce-badge-${item.type}`;
            badge.textContent = item.type === 'text' ? 'T' : item.type === 'audio' ? 'A' : 'I';
            row.appendChild(badge);

            // Content
            const content = document.createElement('div');
            content.className = 'uce-content';

            if (item.type === 'text') {
                const ta = document.createElement('textarea');
                ta.className = 'uce-textarea';
                ta.value = item.text || '';
                ta.placeholder = 'Type here...';
                ta.rows = 1;
                let composing = false;
                ta.addEventListener('compositionstart', () => { composing = true; });
                ta.addEventListener('compositionend', () => {
                    composing = false;
                    item.text = ta.value;
                    _uceAutoResize(ta);
                    this.onChange(this.getItems());
                });
                ta.addEventListener('input', () => {
                    item.text = ta.value;
                    _uceAutoResize(ta);
                    if (!composing) this.onChange(this.getItems());
                });
                ta.addEventListener('keydown', (e) => {
                    if (e.key === 'Backspace' && ta.value === '' && this.items.length > 1) {
                        e.preventDefault();
                        this._remove(idx);
                    }
                    // Enter 提交（shift+enter 换行）
                    if (e.key === 'Enter' && !e.shiftKey && !composing && this.onSubmit) {
                        e.preventDefault();
                        item.text = ta.value;
                        this.onSubmit();
                    }
                });
                content.appendChild(ta);
                requestAnimationFrame(() => _uceAutoResize(ta));
            } else if (item.type === 'image') {
                content.appendChild(_uceCreateImageUI(item, idx, () => {
                    this.render();
                    this.onChange(this.getItems());
                }));
            } else {
                content.appendChild(_uceCreateAudioUI(item, idx, () => {
                    this.render();
                    this.onChange(this.getItems());
                }));
            }
            row.appendChild(content);

            // Action buttons: ↑ ↓ ✕
            const actions = document.createElement('div');
            actions.className = 'uce-item-actions';

            const upBtn = document.createElement('button');
            upBtn.className = 'uce-act';
            upBtn.textContent = '↑'; upBtn.title = 'Move up';
            upBtn.disabled = idx === 0;
            upBtn.addEventListener('click', () => this._move(idx, -1));
            actions.appendChild(upBtn);

            const downBtn = document.createElement('button');
            downBtn.className = 'uce-act';
            downBtn.textContent = '↓'; downBtn.title = 'Move down';
            downBtn.disabled = idx === this.items.length - 1;
            downBtn.addEventListener('click', () => this._move(idx, 1));
            actions.appendChild(downBtn);

            const delBtn = document.createElement('button');
            delBtn.className = 'uce-act uce-act-remove';
            delBtn.textContent = '✕'; delBtn.title = 'Remove';
            delBtn.addEventListener('click', () => this._remove(idx));
            actions.appendChild(delBtn);

            row.appendChild(actions);

            // Drag & drop
            row.addEventListener('dragstart', (e) => {
                this._dragIdx = idx;
                row.classList.add('uce-dragging');
                e.dataTransfer.effectAllowed = 'move';
            });
            row.addEventListener('dragend', () => {
                row.classList.remove('uce-dragging');
                this._dragIdx = -1;
            });
            row.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
            });
            row.addEventListener('drop', (e) => {
                e.preventDefault();
                const from = this._dragIdx;
                const to = idx;
                if (from === -1 || from === to) return;
                const [moved] = this.items.splice(from, 1);
                this.items.splice(to, 0, moved);
                this.render();
                this.onChange(this.getItems());
            });

            list.appendChild(row);
        });

        this.wrap.appendChild(list);

        // 底部操作栏（add + save/cancel 合并）
        const addRow = document.createElement('div');
        addRow.className = 'uce-add-row';

        // 左侧：添加按钮
        const addGroup = document.createElement('div');
        addGroup.className = 'uce-add-group';

        const addText = document.createElement('button');
        addText.className = 'uce-add-btn';
        addText.textContent = '+ Text';
        addText.addEventListener('click', () => this._add('text'));
        addGroup.appendChild(addText);

        const addAudio = document.createElement('button');
        addAudio.className = 'uce-add-btn';
        addAudio.textContent = '+ Audio';
        addAudio.addEventListener('click', () => this._add('audio'));
        addGroup.appendChild(addAudio);

        if (this.showImageBtn) {
            const addImage = document.createElement('button');
            addImage.className = 'uce-add-btn';
            addImage.textContent = '+ Image';
            addImage.addEventListener('click', () => this._add('image'));
            addGroup.appendChild(addImage);
        }

        const recBtn = document.createElement('button');
        recBtn.className = 'uce-add-btn uce-rec-btn-add';
        recBtn.textContent = '🎤';
        recBtn.title = 'Space: record';
        recBtn.addEventListener('click', () => this._startRecording());
        addGroup.appendChild(recBtn);

        addRow.appendChild(addGroup);

        // 右侧：Save/Cancel（非 compose 模式，如 edit inline）
        if (this.onSubmit && !this.isCompose) {
            const actGroup = document.createElement('div');
            actGroup.className = 'uce-act-group';

            if (this.onCancel) {
                const cancelBtn = document.createElement('button');
                cancelBtn.className = 'uce-act-btn';
                cancelBtn.textContent = 'Cancel';
                cancelBtn.addEventListener('click', () => this.onCancel());
                actGroup.appendChild(cancelBtn);
            }

            const saveBtn = document.createElement('button');
            saveBtn.className = 'uce-act-btn uce-act-primary';
            saveBtn.textContent = 'Save';
            saveBtn.addEventListener('click', () => this.onSubmit());
            actGroup.appendChild(saveBtn);

            addRow.appendChild(actGroup);
        }

        this.wrap.appendChild(addRow);
    }

    _add(type) {
        if (type === 'text') this.items.push({ type: 'text', text: '' });
        else if (type === 'image') this.items.push({ type: 'image', file: null, name: '', objectUrl: null, data: null });
        else this.items.push({ type: 'audio', file: null, name: '', duration: 0, objectUrl: null });
        this.render();
        this.onChange(this.getItems());
        if (type === 'text') {
            const tas = this.wrap.querySelectorAll('.uce-textarea');
            const last = tas[tas.length - 1];
            if (last) last.focus();
        }
    }

    _move(idx, dir) {
        const ni = idx + dir;
        if (ni < 0 || ni >= this.items.length) return;
        [this.items[idx], this.items[ni]] = [this.items[ni], this.items[idx]];
        this.render();
        this.onChange(this.getItems());
    }

    _remove(idx) {
        this.items.splice(idx, 1);
        this.render();
        this.onChange(this.getItems());
    }
}
