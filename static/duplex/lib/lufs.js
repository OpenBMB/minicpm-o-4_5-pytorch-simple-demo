/**
 * lufs.js — ITU-R BS.1770 Integrated Loudness (LUFS/LKFS) measurement
 *
 * Implements K-weighting filter and gated loudness measurement.
 * Works offline on a mono Float32Array PCM buffer at any sample rate.
 *
 * Reference: ITU-R BS.1770-4, Annex 2
 * Filter coefficients derived via bilinear transform from analog prototypes.
 * Verified against pyloudnorm reference implementation.
 */

/**
 * Pre-filter (high-shelf) biquad coefficients for K-weighting.
 * Analog prototype: f0=1681.97 Hz, G=+4.0 dB, Q=0.7072
 */
function preFilterCoeffs(fs) {
    const db = 3.999843853973347;
    const f0 = 1681.974450955533;
    const Q  = 0.7071752369554196;
    const K  = Math.tan(Math.PI * f0 / fs);
    const Vh = Math.pow(10, db / 20);
    const Vb = Math.pow(Vh, 0.4996667741545416);
    const K2 = K * K;
    const a0 = 1 + K / Q + K2;
    return {
        b0: (Vh + Vb * K / Q + K2) / a0,
        b1: 2 * (K2 - Vh) / a0,
        b2: (Vh - Vb * K / Q + K2) / a0,
        a1: 2 * (K2 - 1) / a0,
        a2: (1 - K / Q + K2) / a0,
    };
}

/**
 * RLB weighting (high-pass) biquad coefficients for K-weighting.
 * Analog prototype: f0=38.14 Hz, Q=0.5003
 */
function rlbCoeffs(fs) {
    const f0 = 38.13547087602444;
    const Q  = 0.5003270373238773;
    const K  = Math.tan(Math.PI * f0 / fs);
    const K2 = K * K;
    const a0 = 1 + K / Q + K2;
    return {
        b0: 1 / a0,
        b1: -2 / a0,
        b2: 1 / a0,
        a1: 2 * (K2 - 1) / a0,
        a2: (1 - K / Q + K2) / a0,
    };
}

/** Apply a biquad filter (Direct Form II Transposed). Returns new Float32Array. */
function biquad(x, c) {
    const out = new Float32Array(x.length);
    let z1 = 0, z2 = 0;
    for (let i = 0; i < x.length; i++) {
        const xi = x[i];
        const yi = c.b0 * xi + z1;
        z1 = c.b1 * xi - c.a1 * yi + z2;
        z2 = c.b2 * xi - c.a2 * yi;
        out[i] = yi;
    }
    return out;
}

/**
 * Measure integrated loudness (LUFS) of a mono PCM buffer.
 * Implements ITU-R BS.1770-4 with K-weighting and two-stage gating.
 *
 * @param {Float32Array} pcm - Mono PCM samples
 * @param {number} sampleRate - Sample rate in Hz
 * @returns {number} Integrated loudness in LUFS (-Infinity if silent)
 */
export function measureLUFS(pcm, sampleRate) {
    // K-weighting: pre-filter (high shelf) → RLB weighting (high-pass)
    const filtered = biquad(biquad(pcm, preFilterCoeffs(sampleRate)), rlbCoeffs(sampleRate));

    // Split into 400ms blocks with 75% overlap, compute mean square per block
    const blockLen = Math.round(sampleRate * 0.4);
    const step = Math.round(blockLen * 0.25);
    const blockMS = [];
    for (let i = 0; i + blockLen <= filtered.length; i += step) {
        let s = 0;
        for (let j = i; j < i + blockLen; j++) s += filtered[j] * filtered[j];
        blockMS.push(s / blockLen);
    }
    if (!blockMS.length) return -Infinity;

    const toLUFS = (ms) => -0.691 + 10 * Math.log10(ms + 1e-30);

    // Absolute gate: discard blocks below -70 LUFS
    const pass1 = blockMS.filter(ms => toLUFS(ms) > -70);
    if (!pass1.length) return -Infinity;

    // Relative gate: discard blocks below (ungated average - 10) LUFS
    const avg1 = pass1.reduce((a, b) => a + b) / pass1.length;
    const relGate = toLUFS(avg1) - 10;
    const pass2 = pass1.filter(ms => toLUFS(ms) > relGate);
    if (!pass2.length) return -Infinity;

    // Gated integrated loudness
    const avg2 = pass2.reduce((a, b) => a + b) / pass2.length;
    return toLUFS(avg2);
}
