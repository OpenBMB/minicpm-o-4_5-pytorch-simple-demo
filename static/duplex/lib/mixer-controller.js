/**
 * mixer-controller.js — Shared Mixer + Preview + Calibration controller (Layer 0)
 *
 * Extracted from omni-app.js and audio-duplex-app.js to eliminate ~500 lines
 * of duplicated logic. Page-specific differences are handled via a config
 * object passed during construction.
 *
 * Public API:
 *   - init()                — Wire all DOM event listeners (call once on DOMReady)
 *   - startMixerMic()       — Start standalone metering mic
 *   - stopMixerMic()        — Stop standalone metering mic
 *   - startMixerMeters()    — Start RAF meter loop
 *   - stopMixerMeters()     — Stop RAF meter loop
 *   - openMixer() / closeMixer() / toggleMixer()
 *   - getActiveNodes()      — Get current audio nodes for gain control
 *   - applyFileGain() / applyMicGain()
 *   - stopPreview()         — Stop preview recording
 *   - downloadRecording()   — Download session recording blob
 *   - micMeasuredLUFS       — Last calibrated mic LUFS (read by session code)
 */

import { measureLUFS } from './lufs.js';

// ============================================================================
// Pure utility: WAV encoder (mono, 16-bit)
// ============================================================================

/**
 * Encode Float32Array PCM to WAV Blob (mono, 16-bit)
 *
 * Args:
 *   pcm: Float32Array of PCM samples
 *   sampleRate: sample rate of the PCM data
 *
 * Returns:
 *   Blob of type audio/wav
 */
export function encodeWav(pcm, sampleRate) {
    const numSamples = pcm.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);
    const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, numSamples * 2, true);
    for (let i = 0; i < numSamples; i++) {
        const s = Math.max(-1, Math.min(1, pcm[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
}

// ============================================================================
// MixerController class
// ============================================================================

/**
 * 封装 Mixer 面板、Preview 录音、Mic 校准的全部逻辑
 *
 * Config:
 *   sampleRate          : number              — capture sample rate (e.g. 16000)
 *   addLog              : (text) => void       — page logging function
 *   getMedia            : () => object|null    — current media provider reference
 *   monitorElId         : string               — DOM id for monitor volume element
 *   isFileMode          : () => boolean        — whether file mode is active
 *   getFileChunks       : () => Float32Array[]|null — decoded file audio chunks from media
 *   getSelectedFile     : () => File|null      — user-selected File object
 *   getLastRecordingBlob: () => Blob|null      — session recording blob for download
 *   getFallbackNodes    : () => object         — extra fallback nodes (optional)
 *   getDownloadExt      : (blob) => string     — download file extension (optional)
 *   onInit              : (ctrl) => void       — post-init hook (optional)
 *   isMicActive         : () => boolean        — extra mic-active check (optional)
 */
export class MixerController {
    constructor(config) {
        this.sampleRate = config.sampleRate;
        this.addLog = config.addLog;
        this.getMedia = config.getMedia;
        this.monitorElId = config.monitorElId;
        this.isFileMode = config.isFileMode;
        this.getFileChunks = config.getFileChunks;
        this.getSelectedFile = config.getSelectedFile;
        this.getLastRecordingBlob = config.getLastRecordingBlob;
        this._getFallbackNodes = config.getFallbackNodes || (() => ({}));
        this._getDownloadExt = config.getDownloadExt || (() => 'wav');
        this._onInit = config.onInit || null;
        this._isMicActive = config.isMicActive || (() => false);

        // Mixer mic state (always-on metering when mixer panel is open)
        this._mixerMicCtx = null;
        this._mixerMicAnalyser = null;
        this._mixerMicGainNode = null;
        this._mixerMicStream = null;

        this._mixerAnimFrame = null;

        /** Last calibrated mic LUFS — read by session start code */
        this.micMeasuredLUFS = -23;

        // Preview recording state
        this._previewCtx = null;
        this._previewChunks = [];
        this._previewBlob = null;
        this._previewAudioEl = null;

        this._initialized = false;
    }

    // ---- Utility ------------------------------------------------------------

    static dbToLinear(db) { return Math.pow(10, db / 20); }

    // ---- Mixer Panel --------------------------------------------------------

    async startMixerMic() {
        if (this._mixerMicCtx) return;
        try {
            this._mixerMicStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false } });
            this._mixerMicCtx = new AudioContext({ sampleRate: this.sampleRate });
            if (this._mixerMicCtx.state === 'suspended') await this._mixerMicCtx.resume();
            const src = this._mixerMicCtx.createMediaStreamSource(this._mixerMicStream);

            this._mixerMicGainNode = this._mixerMicCtx.createGain();
            const targetLUFS = parseFloat(document.getElementById('mxMicTarget')?.value) || -23;
            const autoGainDb = targetLUFS - this.micMeasuredLUFS;
            const trimDb = parseInt(document.getElementById('mxMicTrim')?.value) || 0;
            this._mixerMicGainNode.gain.value = MixerController.dbToLinear(autoGainDb + trimDb);

            this._mixerMicAnalyser = this._mixerMicCtx.createAnalyser();
            this._mixerMicAnalyser.fftSize = 2048;
            src.connect(this._mixerMicGainNode);
            this._mixerMicGainNode.connect(this._mixerMicAnalyser);
        } catch (e) {
            console.warn('Mixer mic init failed:', e);
        }
    }

    stopMixerMic() {
        this._mixerMicStream?.getTracks().forEach(t => t.stop());
        this._mixerMicCtx?.close().catch(() => {});
        this._mixerMicCtx = null;
        this._mixerMicAnalyser = null;
        this._mixerMicGainNode = null;
        this._mixerMicStream = null;
    }

    getActiveNodes() {
        const media = this.getMedia();
        if (media && media.mixerNodes) return media.mixerNodes;
        if (this._previewCtx && this._previewCtx._nodes) return this._previewCtx._nodes;
        const fallback = this._getFallbackNodes();
        return {
            micGain: this._mixerMicGainNode,
            fileGain: null,
            monitorGain: null,
            monitorEl: null,
            micAnalyser: this._mixerMicAnalyser || fallback.micAnalyser || null,
            fileAnalyser: null,
            mixAnalyser: null,
            connected: fallback.connected || false,
        };
    }

    applyFileGain() {
        const trimDb = parseInt(document.getElementById('mxFileTrim')?.value) || 0;
        const nodes = this.getActiveNodes();
        if (nodes.fileGain) nodes.fileGain.gain.value = MixerController.dbToLinear(trimDb);
    }

    applyMicGain() {
        const targetLUFS = parseFloat(document.getElementById('mxMicTarget')?.value) || -23;
        const autoGainDb = targetLUFS - this.micMeasuredLUFS;
        const trimDb = parseInt(document.getElementById('mxMicTrim')?.value) || 0;
        const autoEl = document.getElementById('mxMicAuto');
        if (autoEl) autoEl.textContent = Math.round(autoGainDb);
        const nodes = this.getActiveNodes();
        if (nodes.micGain) nodes.micGain.gain.value = MixerController.dbToLinear(autoGainDb + trimDb);
    }

    openMixer() {
        const panel = document.getElementById('mixerPanel');
        if (!panel || panel.style.display === 'block') return;
        panel.style.display = 'block';
        if (!panel.dataset.positioned) {
            panel.style.right = '24px';
            panel.style.top = '80px';
            panel.dataset.positioned = '1';
        }
        const btn = document.getElementById('btnMixerToggle');
        if (btn) btn.classList.add('active');
        const media = this.getMedia();
        if (!media || !media.running) this.startMixerMic();
        this.startMixerMeters();
    }

    closeMixer() {
        const panel = document.getElementById('mixerPanel');
        if (panel) panel.style.display = 'none';
        const btn = document.getElementById('btnMixerToggle');
        if (btn) btn.classList.remove('active');
        this.stopMixerMeters();
        this.stopMixerMic();
    }

    toggleMixer() {
        const panel = document.getElementById('mixerPanel');
        if (panel && panel.style.display === 'block') this.closeMixer();
        else this.openMixer();
    }

    _initMixerDrag() {
        const panel = document.getElementById('mixerPanel');
        const handle = document.getElementById('mixerDragHandle');
        if (!panel || !handle) return;
        let dragging = false, startX = 0, startY = 0, origX = 0, origY = 0;
        handle.addEventListener('mousedown', (e) => {
            if (e.target.closest('.mixer-close') || e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
            dragging = true;
            startX = e.clientX; startY = e.clientY;
            const rect = panel.getBoundingClientRect();
            origX = rect.left; origY = rect.top;
            panel.style.right = 'auto';
            panel.style.left = origX + 'px';
            panel.style.top = origY + 'px';
            e.preventDefault();
        });
        document.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            panel.style.left = (origX + e.clientX - startX) + 'px';
            panel.style.top = (origY + e.clientY - startY) + 'px';
        });
        document.addEventListener('mouseup', () => { dragging = false; });
        handle.addEventListener('touchstart', (e) => {
            if (e.target.closest('.mixer-close') || e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
            const t = e.touches[0];
            dragging = true; startX = t.clientX; startY = t.clientY;
            const rect = panel.getBoundingClientRect();
            origX = rect.left; origY = rect.top;
            panel.style.right = 'auto';
            panel.style.left = origX + 'px';
            panel.style.top = origY + 'px';
        }, { passive: true });
        document.addEventListener('touchmove', (e) => {
            if (!dragging) return;
            const t = e.touches[0];
            panel.style.left = (origX + t.clientX - startX) + 'px';
            panel.style.top = (origY + t.clientY - startY) + 'px';
        }, { passive: true });
        document.addEventListener('touchend', () => { dragging = false; });
    }

    startMixerMeters() {
        if (this._mixerAnimFrame) return;
        this._updateMixerMeters();
    }

    stopMixerMeters() {
        if (this._mixerAnimFrame) { cancelAnimationFrame(this._mixerAnimFrame); this._mixerAnimFrame = null; }
    }

    _updateMixerMeters() {
        const panel = document.getElementById('mixerPanel');
        if (!panel || panel.style.display !== 'block') { this._mixerAnimFrame = null; return; }
        this._mixerAnimFrame = requestAnimationFrame(() => this._updateMixerMeters());

        const media = this.getMedia();
        const nodes = this.getActiveNodes();
        const dotMic = document.getElementById('mixerDotMic');
        const dotFile = document.getElementById('mixerDotFile');
        const dotWorklet = document.getElementById('mixerDotWorklet');
        const micOn = this._isMicActive() || (media && media.running) || !!this._previewCtx?._nodes || !!this._mixerMicAnalyser;
        const fileOn = (media && media.running && this.isFileMode()) || !!this._previewCtx?._nodes;
        if (dotMic) dotMic.className = 'mixer-dot ' + (micOn ? 'on' : '');
        if (dotFile) dotFile.className = 'mixer-dot ' + (fileOn ? 'on' : '');
        if (dotWorklet) dotWorklet.className = 'mixer-dot ' + (nodes.connected ? 'on' : '');

        const readMeter = (analyser, barId, valId) => {
            if (!analyser) {
                const el = document.getElementById(barId);
                const val = document.getElementById(valId);
                if (el) el.style.width = '0%';
                if (val) val.textContent = '-60.0 dB';
                return;
            }
            const buf = new Float32Array(analyser.fftSize);
            analyser.getFloatTimeDomainData(buf);
            let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
            const rms = Math.sqrt(s / buf.length);
            const db = rms > 0 ? 20 * Math.log10(rms) : -60;
            const pct = Math.max(0, Math.min(100, (db + 60) / 60 * 100));
            const el = document.getElementById(barId);
            const val = document.getElementById(valId);
            if (el) el.style.width = pct + '%';
            if (val) val.textContent = db.toFixed(1) + ' dB';
        };
        readMeter(nodes.fileAnalyser, 'mxMeterFile', 'mxMeterFileVal');
        readMeter(nodes.micAnalyser, 'mxMeterMic', 'mxMeterMicVal');
        readMeter(nodes.mixAnalyser, 'mxMeterMix', 'mxMeterMixVal');
    }

    _wireMixerSliders() {
        document.getElementById('mxFileTrim')?.addEventListener('input', (e) => {
            const db = parseInt(e.target.value);
            document.getElementById('mxFileTrimVal').textContent = db + ' dB';
            this.applyFileGain();
        });

        document.getElementById('mxMicTrim')?.addEventListener('input', (e) => {
            const db = parseInt(e.target.value);
            document.getElementById('mxMicTrimVal').textContent = db + ' dB';
            this.applyMicGain();
        });

        document.getElementById('mxMonitor')?.addEventListener('input', (e) => {
            const pct = parseInt(e.target.value);
            document.getElementById('mxMonitorVal').textContent = pct + '%';
            const nodes = this.getActiveNodes();
            if (nodes.monitorGain) {
                const comp = nodes.lufsCompensation || 1;
                nodes.monitorGain.gain.value = (pct / 100) * comp;
            }
            if (nodes.monitorEl) nodes.monitorEl.volume = pct / 100;
            const el = document.getElementById(this.monitorElId);
            if (el && !el.paused) el.volume = pct / 100;
        });

        document.getElementById('mxFileTarget')?.addEventListener('change', () => {
            const measEl = document.getElementById('mxFileMeasured');
            const meas = measEl ? parseFloat(measEl.textContent) : NaN;
            if (isFinite(meas)) {
                const target = parseFloat(document.getElementById('mxFileTarget').value) || -33;
                const autoEl = document.getElementById('mxFileAuto');
                if (autoEl) autoEl.textContent = (target - meas).toFixed(1);
            }
        });
        document.getElementById('mxMicTarget')?.addEventListener('change', () => {
            this.applyMicGain();
        });

        document.getElementById('mixerClose')?.addEventListener('click', () => this.closeMixer());
        document.getElementById('btnMixerToggle')?.addEventListener('click', () => this.toggleMixer());
        this._initMixerDrag();
    }

    // ---- Mic Calibration ----------------------------------------------------

    _wireMicCalibration() {
        document.getElementById('micCalBtn')?.addEventListener('click', async () => {
            const btn = document.getElementById('micCalBtn');
            const result = document.getElementById('micCalResult');
            if (btn.classList.contains('recording')) return;

            btn.classList.add('recording');
            result.textContent = 'Please speak normally...';
            btn.textContent = '3...';

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: false },
                });
                const ctx = new AudioContext({ sampleRate: 16000 });
                const source = ctx.createMediaStreamSource(stream);
                const processor = ctx.createScriptProcessor(4096, 1, 1);
                const samples = [];

                processor.onaudioprocess = (e) => {
                    samples.push(new Float32Array(e.inputBuffer.getChannelData(0)));
                };
                source.connect(processor);
                processor.connect(ctx.destination);

                for (let t = 2; t >= 1; t--) {
                    await new Promise(r => setTimeout(r, 1000));
                    btn.textContent = `${t}...`;
                }
                await new Promise(r => setTimeout(r, 1000));

                processor.disconnect();
                source.disconnect();
                stream.getTracks().forEach(t => t.stop());
                await ctx.close();

                const total = samples.reduce((a, c) => a + c.length, 0);
                const pcm = new Float32Array(total);
                let pos = 0;
                for (const chunk of samples) { pcm.set(chunk, pos); pos += chunk.length; }
                const lufs = measureLUFS(pcm, 16000);

                this.micMeasuredLUFS = isFinite(lufs) ? lufs : -23;
                const targetLUFS = parseFloat(document.getElementById('mxMicTarget')?.value) || -23;
                const autoGain = Math.round(targetLUFS - this.micMeasuredLUFS);

                const measEl = document.getElementById('mxMicMeasured');
                if (measEl) measEl.textContent = this.micMeasuredLUFS.toFixed(1);
                const autoEl = document.getElementById('mxMicAuto');
                if (autoEl) autoEl.textContent = autoGain;

                const trimSlider = document.getElementById('mxMicTrim');
                if (trimSlider) { trimSlider.value = 0; }
                const trimValEl = document.getElementById('mxMicTrimVal');
                if (trimValEl) trimValEl.textContent = '0 dB';
                const nodes = this.getActiveNodes();
                if (nodes.micGain) nodes.micGain.gain.value = MixerController.dbToLinear(autoGain);

                result.textContent = `${this.micMeasuredLUFS.toFixed(1)} LUFS → auto ${autoGain} dB`;
            } catch (err) {
                result.textContent = 'err';
                console.error('Mic cal failed:', err);
            } finally {
                btn.classList.remove('recording');
                btn.textContent = '\u{1F3A4} Measure';
            }
        });
    }

    // ---- Preview Recording --------------------------------------------------

    _wirePreviewButtons() {
        const btnRec = document.getElementById('mxPreviewRec');
        const btnStop = document.getElementById('mxPreviewStop');
        const btnPlay = document.getElementById('mxPreviewPlay');
        if (!btnRec || !btnStop || !btnPlay) return;

        btnRec.addEventListener('click', async () => {
            const selectedFile = this.getSelectedFile();
            if (!selectedFile) {
                this.addLog('Preview: no file selected');
                return;
            }
            try {
                btnRec.disabled = true;
                btnRec.textContent = 'Decoding...';

                let fileChunks = this.getFileChunks();
                if (!fileChunks || fileChunks.length === 0) {
                    this.addLog('Preview: decoding file audio...');
                    const arrayBuffer = await selectedFile.arrayBuffer();
                    const tmpCtx = new AudioContext();
                    const decoded = await tmpCtx.decodeAudioData(arrayBuffer.slice(0));
                    const targetFrames = Math.ceil(decoded.duration * this.sampleRate);
                    const offCtx = new OfflineAudioContext(1, targetFrames, this.sampleRate);
                    const src = offCtx.createBufferSource();
                    src.buffer = decoded;
                    src.connect(offCtx.destination);
                    src.start();
                    const resampled = await offCtx.startRendering();
                    const pcm = resampled.getChannelData(0);
                    await tmpCtx.close();

                    const targetLUFS = parseFloat(document.getElementById('mxFileTarget')?.value) || -33;
                    const srcLUFS = measureLUFS(pcm, this.sampleRate);
                    const measEl = document.getElementById('mxFileMeasured');
                    if (measEl) measEl.textContent = isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—';
                    const autoGainDb = isFinite(srcLUFS) ? targetLUFS - srcLUFS : 0;
                    const autoEl = document.getElementById('mxFileAuto');
                    if (autoEl) autoEl.textContent = isFinite(srcLUFS) ? autoGainDb.toFixed(1) : '—';
                    if (isFinite(srcLUFS)) {
                        const gain = Math.pow(10, autoGainDb / 20);
                        for (let i = 0; i < pcm.length; i++) pcm[i] *= gain;
                    }
                    fileChunks = [];
                    for (let i = 0; i < pcm.length; i += this.sampleRate) {
                        fileChunks.push(pcm.slice(i, Math.min(i + this.sampleRate, pcm.length)));
                    }
                }

                this.stopMixerMic();

                const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false } });
                this._previewCtx = new AudioContext({ sampleRate: this.sampleRate });
                if (this._previewCtx.state === 'suspended') await this._previewCtx.resume();
                await this._previewCtx.audioWorklet.addModule('/static/duplex/lib/capture-processor.js');

                const micSrc = this._previewCtx.createMediaStreamSource(stream);
                const micTarget = parseFloat(document.getElementById('mxMicTarget')?.value) || -23;
                const micAutoDb = micTarget - this.micMeasuredLUFS;
                const micTrimDb = parseInt(document.getElementById('mxMicTrim')?.value) || 0;
                const micGainNode = this._previewCtx.createGain();
                micGainNode.gain.value = MixerController.dbToLinear(micAutoDb + micTrimDb);

                const fileTrimDb = parseInt(document.getElementById('mxFileTrim')?.value) || 0;
                const fileGainNode = this._previewCtx.createGain();
                fileGainNode.gain.value = MixerController.dbToLinear(fileTrimDb);

                const micAnalyser = this._previewCtx.createAnalyser(); micAnalyser.fftSize = 2048;
                const fileAnalyser = this._previewCtx.createAnalyser(); fileAnalyser.fftSize = 2048;
                const mixAnalyser = this._previewCtx.createAnalyser(); mixAnalyser.fftSize = 2048;

                const monPct = parseInt(document.getElementById('mxMonitor')?.value) || 50;
                const fileAutoGainDb = parseFloat(document.getElementById('mxFileAuto')?.textContent) || 0;
                const monitorGain = this._previewCtx.createGain();
                const lufsCompensation = fileAutoGainDb < 0 ? Math.pow(10, -fileAutoGainDb / 20) : 1;
                monitorGain.gain.value = (monPct / 100) * lufsCompensation;

                const totalSamples = fileChunks.reduce((a, c) => a + c.length, 0);
                const audioBuf = this._previewCtx.createBuffer(1, totalSamples, this.sampleRate);
                const ch = audioBuf.getChannelData(0);
                let pos = 0;
                for (const c of fileChunks) { ch.set(c, pos); pos += c.length; }
                const fileSrc = this._previewCtx.createBufferSource();
                fileSrc.buffer = audioBuf;

                const captureNode = new AudioWorkletNode(this._previewCtx, 'capture-processor', {
                    processorOptions: { chunkSize: this.sampleRate },
                });

                // Audio graph routing (mirrors real session)
                micSrc.connect(micGainNode);
                micGainNode.connect(captureNode);
                micGainNode.connect(micAnalyser);

                fileSrc.connect(fileGainNode);
                fileGainNode.connect(captureNode);
                fileGainNode.connect(fileAnalyser);
                fileGainNode.connect(monitorGain);
                monitorGain.connect(this._previewCtx.destination);

                const mixBus = this._previewCtx.createGain();
                mixBus.gain.value = 1;
                micGainNode.connect(mixBus);
                fileGainNode.connect(mixBus);
                mixBus.connect(mixAnalyser);

                this._previewCtx._nodes = {
                    micGain: micGainNode, fileGain: fileGainNode, monitorGain: monitorGain,
                    monitorEl: null, lufsCompensation,
                    micAnalyser, fileAnalyser, mixAnalyser, connected: true,
                };

                this._previewChunks = [];
                captureNode.port.onmessage = (e) => {
                    if (e.data.type === 'chunk') this._previewChunks.push(new Float32Array(e.data.audio));
                };
                captureNode.port.postMessage({ command: 'start' });
                fileSrc.start();

                this._previewCtx._stream = stream;
                this._previewCtx._fileSrc = fileSrc;
                this._previewCtx._captureNode = captureNode;
                fileSrc.onended = () => this.stopPreview();

                btnRec.textContent = '● Rec';
                btnStop.disabled = false;
                btnPlay.disabled = true;
                btnRec.classList.add('recording');
                this.addLog('Preview: recording started');
            } catch (err) {
                this.addLog(`Preview error: ${err.message}`);
                btnRec.disabled = false;
                btnRec.textContent = '● Rec';
            }
        });

        btnStop.addEventListener('click', () => this.stopPreview());

        btnPlay.addEventListener('click', () => {
            if (!this._previewBlob) return;
            if (this._previewAudioEl) { this._previewAudioEl.pause(); }
            const url = URL.createObjectURL(this._previewBlob);
            this._previewAudioEl = new Audio(url);
            this._previewAudioEl.onended = () => URL.revokeObjectURL(url);
            this._previewAudioEl.play();
        });
    }

    async stopPreview() {
        const btnRec = document.getElementById('mxPreviewRec');
        const btnStop = document.getElementById('mxPreviewStop');
        const btnPlay = document.getElementById('mxPreviewPlay');
        const durEl = document.getElementById('mxPreviewDur');

        if (this._previewCtx) {
            try { this._previewCtx._captureNode?.port.postMessage({ command: 'stop' }); } catch (_) {}
            await new Promise(r => setTimeout(r, 100));
            try { this._previewCtx._fileSrc?.stop(); } catch (_) {}
            this._previewCtx._stream?.getTracks().forEach(t => t.stop());
            this._previewCtx._nodes = null;
            this._previewCtx.close().catch(() => {});
        }
        this._previewCtx = null;

        const panel = document.getElementById('mixerPanel');
        const media = this.getMedia();
        if (panel && panel.style.display === 'block' && (!media || !media.running)) {
            this.startMixerMic();
        }

        if (this._previewChunks.length > 0) {
            const total = this._previewChunks.reduce((a, c) => a + c.length, 0);
            const pcm = new Float32Array(total);
            let p = 0;
            for (const c of this._previewChunks) { pcm.set(c, p); p += c.length; }
            this._previewBlob = encodeWav(pcm, this.sampleRate);
            const dur = (total / this.sampleRate).toFixed(1);
            if (durEl) durEl.textContent = dur + 's';
            this.addLog(`Preview: recorded ${dur}s`);
        }
        this._previewChunks = [];

        if (btnRec) { btnRec.disabled = false; btnRec.classList.remove('recording'); }
        if (btnStop) btnStop.disabled = true;
        if (btnPlay) btnPlay.disabled = !this._previewBlob;
    }

    // ---- Recording Download -------------------------------------------------

    downloadRecording() {
        const blob = this.getLastRecordingBlob();
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const ext = this._getDownloadExt(blob);
        a.download = `session_${ts}.${ext}`;
        a.click();
        URL.revokeObjectURL(url);
        this.addLog(`Downloaded: ${a.download}`);
    }

    // ---- Init ---------------------------------------------------------------

    /** Wire all DOM event listeners. Call once on page load / DOMContentLoaded. */
    init() {
        if (this._initialized) return;
        this._initialized = true;
        this._wireMixerSliders();
        this._wirePreviewButtons();
        this._wireMicCalibration();
        const dlBtn = document.getElementById('btnDownloadRec');
        if (dlBtn) dlBtn.addEventListener('click', () => this.downloadRecording());
        if (this._onInit) this._onInit(this);
    }
}
