/**
 * capture-processor.js — AudioWorklet processor for mixing + capturing audio
 *
 * Runs on the audio rendering thread. Receives mixed audio from the graph
 * (browser auto-sums all connected inputs), passes it through to output
 * (for MediaStreamDestination + monitor), and posts 1-second PCM chunks
 * to the main thread via MessagePort.
 *
 * Commands via port.postMessage:
 *   { command: 'start' }  — begin accumulating and emitting chunks
 *   { command: 'stop' }   — stop accumulating, flush buffer
 *
 * Emits via port.postMessage:
 *   { type: 'chunk', audio: Float32Array }  — 1-second PCM chunk (Transferable)
 */
class CaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this._chunkSize = options.processorOptions?.chunkSize || 16000;
        this._buffer = new Float32Array(0);
        this._active = false;

        this.port.onmessage = (e) => {
            const { command } = e.data;
            if (command === 'start') {
                this._active = true;
                this._buffer = new Float32Array(0);
            } else if (command === 'stop') {
                if (this._buffer.length > 0) {
                    const remaining = this._buffer.slice(0);
                    this.port.postMessage(
                        { type: 'chunk', audio: remaining, final: true },
                        [remaining.buffer]
                    );
                }
                this._active = false;
                this._buffer = new Float32Array(0);
            }
        };
    }

    process(inputs, outputs) {
        const input = inputs[0]?.[0];
        const output = outputs[0]?.[0];

        // Always pass-through: enables MediaStreamDestination + monitor downstream
        if (input && output) {
            output.set(input);
        }

        if (!this._active || !input || input.length === 0) {
            return true;
        }

        // Accumulate input samples
        const newBuf = new Float32Array(this._buffer.length + input.length);
        newBuf.set(this._buffer);
        newBuf.set(input, this._buffer.length);
        this._buffer = newBuf;

        // Emit full chunks
        while (this._buffer.length >= this._chunkSize) {
            const chunk = this._buffer.slice(0, this._chunkSize);
            this._buffer = this._buffer.slice(this._chunkSize);
            this.port.postMessage({ type: 'chunk', audio: chunk }, [chunk.buffer]);
        }

        return true;
    }
}

registerProcessor('capture-processor', CaptureProcessor);
