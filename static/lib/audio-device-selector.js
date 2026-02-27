/**
 * AudioDeviceSelector — Shared audio input/output device picker
 *
 * Enumerates microphones and speakers, renders into <select> elements,
 * persists selection to localStorage, auto-refreshes on device changes.
 *
 * Usage:
 *   import { AudioDeviceSelector } from '../lib/audio-device-selector.js';
 *
 *   const devices = new AudioDeviceSelector({
 *       micSelectEl:     document.getElementById('micDevice'),
 *       speakerSelectEl: document.getElementById('speakerDevice'),
 *       refreshBtnEl:    document.getElementById('btnRefreshDevices'),
 *       storagePrefix:   'half_duplex',
 *   });
 *   await devices.init();
 *
 *   // When creating getUserMedia constraints:
 *   const micId = devices.getSelectedMicId();
 *   const constraints = { audio: { ...(micId ? { deviceId: { exact: micId } } : {}) } };
 *
 *   // After creating an AudioContext for playback:
 *   devices.applySinkId(audioContext);
 */

export class AudioDeviceSelector {
    /**
     * @param {object} opts
     * @param {HTMLSelectElement} opts.micSelectEl
     * @param {HTMLSelectElement} opts.speakerSelectEl
     * @param {HTMLButtonElement} [opts.refreshBtnEl]
     * @param {string} opts.storagePrefix - localStorage key prefix
     * @param {function} [opts.onMicChange]    - callback(deviceId)
     * @param {function} [opts.onSpeakerChange] - callback(deviceId)
     */
    constructor({ micSelectEl, speakerSelectEl, refreshBtnEl, storagePrefix, onMicChange, onSpeakerChange }) {
        this._micEl = micSelectEl;
        this._spkEl = speakerSelectEl;
        this._refreshBtn = refreshBtnEl;
        this._prefix = storagePrefix;
        this._onMicChange = onMicChange || null;
        this._onSpeakerChange = onSpeakerChange || null;

        this._micKey = `${storagePrefix}_mic`;
        this._spkKey = `${storagePrefix}_speaker`;

        this._micEl.addEventListener('change', () => {
            localStorage.setItem(this._micKey, this._micEl.value);
            if (this._onMicChange) this._onMicChange(this._micEl.value);
        });
        this._spkEl.addEventListener('change', () => {
            localStorage.setItem(this._spkKey, this._spkEl.value);
            if (this._onSpeakerChange) this._onSpeakerChange(this._spkEl.value);
        });
        if (this._refreshBtn) {
            this._refreshBtn.addEventListener('click', () => this.enumerate());
        }

        this._onDeviceChange = () => this.enumerate();
        navigator.mediaDevices.addEventListener('devicechange', this._onDeviceChange);
    }

    async init() {
        await this.enumerate();
    }

    async enumerate() {
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (_) { /* need permission to get labels */ }

        const devices = await navigator.mediaDevices.enumerateDevices();
        const mics = devices.filter(d => d.kind === 'audioinput');
        const speakers = devices.filter(d => d.kind === 'audiooutput');

        const savedMic = localStorage.getItem(this._micKey);
        const savedSpk = localStorage.getItem(this._spkKey);

        this._micEl.innerHTML = '';
        mics.forEach((d, i) => {
            const opt = document.createElement('option');
            opt.value = d.deviceId;
            opt.textContent = d.label || `Microphone ${i + 1}`;
            if (d.deviceId === savedMic) opt.selected = true;
            this._micEl.appendChild(opt);
        });

        this._spkEl.innerHTML = '';
        if (speakers.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'Default (browser-managed)';
            this._spkEl.appendChild(opt);
        } else {
            speakers.forEach((d, i) => {
                const opt = document.createElement('option');
                opt.value = d.deviceId;
                opt.textContent = d.label || `Speaker ${i + 1}`;
                if (d.deviceId === savedSpk) opt.selected = true;
                this._spkEl.appendChild(opt);
            });
        }
    }

    getSelectedMicId() {
        return this._micEl.value || undefined;
    }

    getSelectedSpeakerId() {
        return this._spkEl.value || undefined;
    }

    /**
     * Apply the selected speaker to an AudioContext via setSinkId.
     * @param {AudioContext} audioContext
     */
    applySinkId(audioContext) {
        if (!audioContext) return;
        const id = this._spkEl.value;
        if (id && typeof audioContext.setSinkId === 'function') {
            audioContext.setSinkId(id).catch(e =>
                console.warn('[AudioDeviceSelector] setSinkId failed:', e.message)
            );
        }
    }

    clearSaved() {
        localStorage.removeItem(this._micKey);
        localStorage.removeItem(this._spkKey);
    }
}
