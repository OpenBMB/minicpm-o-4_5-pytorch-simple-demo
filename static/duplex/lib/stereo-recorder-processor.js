/**
 * stereo-recorder-processor.js — AudioWorklet for real-time stereo mixing
 *
 * Outputs stereo PCM for MediaRecorder (used by SessionVideoRecorder).
 * Node must be created with numberOfInputs: 2.
 *
 * Left channel  = user input mix (mic + file, 16kHz)
 *   - Preferred: audio graph input 0 (zero delay, connected via AudioNode)
 *   - Fallback: postMessage queue (legacy pushLeft path)
 *   - Mode selected by { command: 'useInputLeft' } message
 *
 * Right channel = AI response audio (resampled to 16kHz)
 *   - Preferred: audio graph input 1 (sample-accurate AudioBufferSourceNode scheduling)
 *   - Fallback: postMessage queue (legacy pushRight path, susceptible to queue-drain gaps)
 *   - Mode selected by { command: 'useInputRight' } message
 */

class StereoRecorderProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._useInputLeft = false;
        this._useInputRight = false;
        this._leftQueue = [];
        this._rightQueue = [];
        this._leftBuf = null;
        this._leftPos = 0;
        this._rightBuf = null;
        this._rightPos = 0;

        this.port.onmessage = (e) => {
            const { command, channel, audio } = e.data;
            if (command === 'useInputLeft') {
                this._useInputLeft = true;
            } else if (command === 'useInputRight') {
                this._useInputRight = true;
            } else if (channel === 'left') {
                this._leftQueue.push(new Float32Array(audio));
            } else if (channel === 'right') {
                this._rightQueue.push(new Float32Array(audio));
            }
        };
    }

    process(inputs, outputs) {
        const out = outputs[0];
        if (out.length < 2) return true;

        // Left: audio graph input 0 or postMessage queue
        if (this._useInputLeft) {
            const leftIn = inputs[0]?.[0];
            if (leftIn && leftIn.length > 0) {
                out[0].set(leftIn);
            } else {
                out[0].fill(0);
            }
        } else {
            this._fill(out[0], '_leftQueue', '_leftBuf', '_leftPos');
        }

        // Right: audio graph input 1 (scheduled AudioBufferSourceNodes) or postMessage queue
        if (this._useInputRight) {
            const rightIn = inputs[1]?.[0];
            if (rightIn && rightIn.length > 0) {
                out[1].set(rightIn);
            } else {
                out[1].fill(0);
            }
        } else {
            this._fill(out[1], '_rightQueue', '_rightBuf', '_rightPos');
        }

        return true;
    }

    _fill(dest, queueKey, bufKey, posKey) {
        let written = 0;
        while (written < dest.length) {
            if (!this[bufKey] || this[posKey] >= this[bufKey].length) {
                if (this[queueKey].length === 0) {
                    dest.fill(0, written);
                    return;
                }
                this[bufKey] = this[queueKey].shift();
                this[posKey] = 0;
            }
            const avail = this[bufKey].length - this[posKey];
            const need = dest.length - written;
            const n = Math.min(avail, need);
            dest.set(this[bufKey].subarray(this[posKey], this[posKey] + n), written);
            this[posKey] += n;
            written += n;
        }
    }
}

registerProcessor('stereo-recorder-processor', StereoRecorderProcessor);
