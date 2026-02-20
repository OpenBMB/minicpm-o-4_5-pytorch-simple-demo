/**
 * Session Save & Share 共享组件
 *
 * 聊天页面：Upload & Share 按钮 → 有前端录制 blob 时先上传再复制链接，无 blob 直接复制链接 + 5s toast。
 * 首页：读取 localStorage 展示 Recent Sessions 列表。
 *
 * 使用方式（聊天页面）：
 *   const ui = new SaveShareUI({ containerId: 'save-share-container', appType: 'omni_duplex' });
 *   ui.setSessionId('omni_xxx');
 *   // 前端录制完成后：
 *   ui.setRecordingBlob(blob, 'webm');
 */

const RECENT_SESSIONS_KEY = 'minicpmo45_recent_sessions';
const MAX_RECENT = 20;

class SaveShareUI {
    /**
     * @param {Object} opts
     * @param {string} opts.containerId - 挂载容器的 DOM id
     * @param {string} [opts.appType] - 应用类型标识
     */
    constructor(opts) {
        this.appType = opts.appType || 'unknown';
        this._sessionId = null;
        this._recordingBlob = null;
        this._recordingExt = null;
        this._uploading = false;
        this._container = document.getElementById(opts.containerId);
        if (!this._container) return;
        this._render();
    }

    setSessionId(sessionId) {
        this._sessionId = sessionId;
        this._updateBtn();
    }

    /**
     * 设置前端录制的 Blob，Upload & Share 时会先上传此文件
     * @param {Blob} blob - 录制的音频/视频 blob
     * @param {string} ext - 文件扩展名 ('webm', 'wav', 'mp4')
     */
    setRecordingBlob(blob, ext) {
        this._recordingBlob = blob;
        this._recordingExt = ext || 'webm';
        this._updateBtn();
    }

    _updateBtn() {
        const btn = this._container?.querySelector('.ss-btn');
        if (btn) {
            btn.disabled = !this._sessionId || this._uploading;
            btn.textContent = this._uploading ? 'Uploading...' : 'Upload & Share';
        }
    }

    _render() {
        this._container.innerHTML = `
            <button class="ss-btn" disabled>Upload & Share</button>
            <div class="ss-toast" style="display:none;"></div>
        `;
        this._container.querySelector('.ss-btn').addEventListener('click', () => this._onSave());
    }

    async _onSave() {
        if (!this._sessionId || this._uploading) return;
        const url = `${window.location.origin}/s/${this._sessionId}`;

        if (this._recordingBlob && this._recordingBlob.size > 0) {
            this._uploading = true;
            this._updateBtn();
            this._showToast('上传前端录制中...');
            try {
                const form = new FormData();
                form.append('file', this._recordingBlob, `recording.${this._recordingExt}`);
                const resp = await fetch(`/api/sessions/${this._sessionId}/upload-recording`, {
                    method: 'POST', body: form,
                });
                if (!resp.ok) {
                    const detail = await resp.text();
                    throw new Error(`Upload failed: ${resp.status} ${detail}`);
                }
            } catch (e) {
                console.error('[SaveShare] upload error:', e);
                this._showToast(`上传失败: ${e.message}\n链接仍可用（无前端录制）: ${url}`, true);
                this._uploading = false;
                this._updateBtn();
                this._addToRecent(this._sessionId);
                return;
            }
            this._uploading = false;
            this._updateBtn();
        }

        this._addToRecent(this._sessionId);
        navigator.clipboard.writeText(url).then(() => {
            this._showToast(`已复制到剪贴板\n${url}`);
        }).catch(() => {
            this._showToast(`分享链接: ${url}`, true);
        });
    }

    _showToast(text, isManual) {
        const toast = this._container?.querySelector('.ss-toast');
        if (!toast) return;
        toast.textContent = text;
        toast.style.display = 'block';
        toast.classList.toggle('manual', !!isManual);
        clearTimeout(this._toastTimer);
        this._toastTimer = setTimeout(() => { toast.style.display = 'none'; }, 5000);
    }

    _addToRecent(sessionId) {
        const list = SaveShareUI.getRecentSessions();
        const idx = list.findIndex(s => s.id === sessionId);
        if (idx !== -1) list.splice(idx, 1);
        list.unshift({ id: sessionId, appType: this.appType, savedAt: new Date().toISOString() });
        if (list.length > MAX_RECENT) list.length = MAX_RECENT;
        localStorage.setItem(RECENT_SESSIONS_KEY, JSON.stringify(list));
    }

    static getRecentSessions() {
        try { return JSON.parse(localStorage.getItem(RECENT_SESSIONS_KEY) || '[]'); }
        catch { return []; }
    }

    static clearRecentSessions() {
        localStorage.removeItem(RECENT_SESSIONS_KEY);
    }
}

/* Inject minimal styles */
(function() {
    if (document.getElementById('ss-styles')) return;
    const s = document.createElement('style');
    s.id = 'ss-styles';
    s.textContent = `
        .ss-btn {
            padding: 7px 16px; border-radius: 8px; font-size: 13px; font-weight: 500;
            background: #2d2d2d; color: #fff; border: none; cursor: pointer;
            transition: opacity 0.15s;
        }
        .ss-btn:disabled { opacity: 0.35; cursor: not-allowed; }
        .ss-btn:not(:disabled):hover { opacity: 0.8; }
        .ss-toast {
            position: fixed; bottom: 56px; right: 16px;
            background: rgba(30,30,30,0.92); color: #fff;
            padding: 10px 16px; border-radius: 10px; font-size: 13px;
            line-height: 1.5; white-space: pre-line; max-width: 360px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.18);
            animation: ss-fade-in 0.2s ease;
            z-index: 10000;
        }
        .ss-toast.manual { user-select: all; }
        @keyframes ss-fade-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    `;
    document.head.appendChild(s);
})();
