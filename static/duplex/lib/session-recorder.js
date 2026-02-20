/**
 * lib/session-recorder.js — Stereo session recorder (zero DOM dependency)
 *
 * Records a duplex session into a stereo WAV file:
 *   Left channel  = user input mix (mic + file, 16kHz from CaptureProcessor)
 *   Right channel = AI response audio (24kHz from AudioPlayer, resampled to 16kHz)
 *
 * Time alignment: both channels share the same wall-clock timeline starting
 * from recorder.start(). Left channel grows sequentially as chunks arrive.
 * Right channel is placed at the actual AudioPlayer scheduled playback time
 * (not arrival time), with forward-clamp to prevent overlap.
 *
 * @import { resampleAudio } from './duplex-utils.js'
 */

import { resampleAudio } from './duplex-utils.js';

export class SessionRecorder {
    /**
     * @param {number} [inputSR=16000] - Sample rate of left channel (CaptureProcessor output)
     * @param {number} [aiSR=24000] - Sample rate of AI audio (AudioPlayer raw output)
     */
    constructor(inputSR = 16000, aiSR = 24000) {
        this._inputSR = inputSR;
        this._aiSR = aiSR;
        this._recording = false;
        this._startTime = 0;

        /** @type {Float32Array[]} */
        this._leftChunks = [];

        /** @type {{offset: number, data: Float32Array}[]} */
        this._rightEntries = [];
        this._rightNextOffset = 0;
    }

    get recording() { return this._recording; }

    /**
     * Start recording. Call at session start.
     */
    start() {
        this._leftChunks = [];
        this._rightEntries = [];
        this._rightNextOffset = 0;
        this._recording = true;
        this._paused = false;
        this._startTime = performance.now();
        console.log('[SessionRecorder] recording started');
    }

    /**
     * Pause recording. Chunks pushed during pause are silently discarded.
     * Both channels stop growing simultaneously, keeping timeline aligned.
     */
    pause() {
        if (!this._recording || this._paused) return;
        this._paused = true;
        console.log('[SessionRecorder] paused');
    }

    /**
     * Resume recording after pause.
     */
    resume() {
        if (!this._recording || !this._paused) return;
        this._paused = false;
        console.log('[SessionRecorder] resumed');
    }

    /**
     * Push a left-channel chunk (user input mix from CaptureProcessor).
     * Chunks are appended sequentially — order must match real time.
     * @param {Float32Array} pcm16k - 16kHz mono PCM
     */
    pushLeft(pcm16k) {
        if (!this._recording || this._paused) return;
        this._leftChunks.push(pcm16k);
    }

    /**
     * Push a right-channel chunk (AI response from AudioPlayer.onRawAudio).
     * Placed at the actual playback-time offset (from AudioPlayer's schedule),
     * with forward-clamp to prevent overlap as safety net.
     * @param {Float32Array} samples - Raw PCM at aiSR
     * @param {number} sampleRate - Source sample rate (typically 24000)
     * @param {number} timestamp - performance.now()-based playback timestamp from AudioPlayer
     */
    pushRight(samples, sampleRate, timestamp) {
        if (!this._recording || this._paused) return;
        const resampled = resampleAudio(samples, sampleRate, this._inputSR);
        const elapsedMs = timestamp - this._startTime;
        const arrivalOffset = Math.max(0, Math.floor(elapsedMs / 1000 * this._inputSR));
        const offset = Math.max(arrivalOffset, this._rightNextOffset);
        this._rightEntries.push({ offset, data: resampled });
        this._rightNextOffset = offset + resampled.length;
    }

    /**
     * Stop recording and encode stereo WAV.
     * @returns {{ blob: Blob, durationSec: number, leftSamples: number, rightSamples: number }}
     */
    stop() {
        this._recording = false;

        const leftTotal = this._leftChunks.reduce((a, c) => a + c.length, 0);

        let rightMaxExtent = 0;
        for (const entry of this._rightEntries) {
            rightMaxExtent = Math.max(rightMaxExtent, entry.offset + entry.data.length);
        }

        const totalSamples = Math.max(leftTotal, rightMaxExtent);
        if (totalSamples === 0) {
            console.warn('[SessionRecorder] no audio recorded');
            return { blob: new Blob(), durationSec: 0, leftSamples: 0, rightSamples: 0 };
        }

        // Build left channel (sequential)
        const left = new Float32Array(totalSamples);
        let pos = 0;
        for (const chunk of this._leftChunks) {
            left.set(chunk, pos);
            pos += chunk.length;
        }

        // Build right channel (sparse placement)
        const right = new Float32Array(totalSamples);
        let rightSampleCount = 0;
        for (const entry of this._rightEntries) {
            for (let i = 0; i < entry.data.length && entry.offset + i < totalSamples; i++) {
                right[entry.offset + i] += entry.data[i];
            }
            rightSampleCount += entry.data.length;
        }

        const blob = this._encodeStereoWav(left, right, this._inputSR);
        const durationSec = totalSamples / this._inputSR;

        console.log(`[SessionRecorder] encoded: ${durationSec.toFixed(1)}s stereo WAV, ` +
            `${(blob.size / 1024).toFixed(0)} KB, left=${leftTotal} right=${rightSampleCount} samples`);

        // Free memory
        this._leftChunks = [];
        this._rightEntries = [];

        return { blob, durationSec, leftSamples: leftTotal, rightSamples: rightSampleCount };
    }

    /**
     * Encode two mono Float32 channels into a stereo 16-bit PCM WAV Blob.
     * @param {Float32Array} left
     * @param {Float32Array} right
     * @param {number} sampleRate
     * @returns {Blob}
     */
    _encodeStereoWav(left, right, sampleRate) {
        const numChannels = 2;
        const bitsPerSample = 16;
        const bytesPerSample = bitsPerSample / 8;
        const blockAlign = numChannels * bytesPerSample;
        const totalSamples = left.length; // both channels same length
        const dataSize = totalSamples * blockAlign;
        const bufferSize = 44 + dataSize;

        const buffer = new ArrayBuffer(bufferSize);
        const view = new DataView(buffer);

        const writeStr = (offset, str) => {
            for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
        };

        // RIFF header
        writeStr(0, 'RIFF');
        view.setUint32(4, bufferSize - 8, true);
        writeStr(8, 'WAVE');

        // fmt chunk
        writeStr(12, 'fmt ');
        view.setUint32(16, 16, true);                          // chunk size
        view.setUint16(20, 1, true);                           // PCM format
        view.setUint16(22, numChannels, true);                 // stereo
        view.setUint32(24, sampleRate, true);                  // sample rate
        view.setUint32(28, sampleRate * blockAlign, true);     // byte rate
        view.setUint16(32, blockAlign, true);                  // block align
        view.setUint16(34, bitsPerSample, true);               // bits per sample

        // data chunk
        writeStr(36, 'data');
        view.setUint32(40, dataSize, true);

        // Interleaved stereo PCM (L, R, L, R, ...)
        let offset = 44;
        for (let i = 0; i < totalSamples; i++) {
            const l = Math.max(-1, Math.min(1, left[i]));
            view.setInt16(offset, l < 0 ? l * 0x8000 : l * 0x7FFF, true);
            offset += 2;

            const r = Math.max(-1, Math.min(1, right[i]));
            view.setInt16(offset, r < 0 ? r * 0x8000 : r * 0x7FFF, true);
            offset += 2;
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }
}
