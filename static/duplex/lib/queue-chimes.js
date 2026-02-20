/**
 * queue-chimes.js — Web Audio API 音效（排队提示、闹铃、会话开始）
 *
 * 所有音效均通过 OscillatorNode 合成，无需加载音频文件。
 * 共享于 Audio Duplex 和 Omni Duplex 页面。
 *
 * 使用模块级共享 AudioContext：浏览器只允许在用户手势调用链中创建
 * AudioContext。playAlarmBell（WS onmessage 回调）首次创建后，
 * playSessionChime（async/await 微任务链）直接复用，避免被浏览器拒绝。
 */

let _ctx = null;

function getCtx() {
    if (!_ctx || _ctx.state === 'closed') {
        _ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return _ctx;
}

/**
 * 循环播放"叮-咚"提示音（position == 1 时使用）。
 * 每个周期 1s：0.00s 叮(880Hz)  0.50s 咚(660Hz)
 * @returns {() => void} 调用以停止循环
 */
export function startDingDongLoop() {
    let stopped = false;
    let timeoutId = null;

    function playOneCycle() {
        if (stopped) return;
        try {
            const ctx = getCtx();
            const now = ctx.currentTime;

            const g1 = ctx.createGain();
            g1.gain.setValueAtTime(0.18, now);
            g1.gain.exponentialRampToValueAtTime(0.001, now + 0.20);
            g1.connect(ctx.destination);
            const o1 = ctx.createOscillator();
            o1.type = 'sine';
            o1.frequency.value = 880;
            o1.connect(g1);
            o1.start(now);
            o1.stop(now + 0.20);

            const g2 = ctx.createGain();
            g2.gain.setValueAtTime(0.18, now + 0.50);
            g2.gain.exponentialRampToValueAtTime(0.001, now + 0.70);
            g2.connect(ctx.destination);
            const o2 = ctx.createOscillator();
            o2.type = 'sine';
            o2.frequency.value = 660;
            o2.connect(g2);
            o2.start(now + 0.50);
            o2.stop(now + 0.70);
        } catch (_) { /* AudioContext not available */ }

        timeoutId = setTimeout(playOneCycle, 2000);
    }

    playOneCycle();

    return () => {
        stopped = true;
        if (timeoutId !== null) { clearTimeout(timeoutId); timeoutId = null; }
    };
}

/**
 * 播放 Alarm 音效（queue_done / worker assigned 时使用）。
 * C大调琶音 C5→E5→G5→C6 × 2 遍，sine + 泛音，ADSR 包络。
 * @returns {Promise<void>}
 */
export function playAlarmBell() {
    return new Promise((resolve) => {
        try {
            const ctx = getCtx();
            const comp = ctx.createDynamicsCompressor();
            comp.threshold.value = -20;
            comp.knee.value = 12;
            comp.ratio.value = 4;
            comp.attack.value = 0.003;
            comp.release.value = 0.1;
            comp.connect(ctx.destination);

            const notes = [523.25, 659.25, 783.99, 1046.50];
            const noteS = 0.086, gain = 0.85, atkS = 0.015, decS = 0.094;
            const susLv = 0.84, relS = 0.155, overtone = 0.22;
            const repeats = 2, gapS = 0.120;
            const totalNote = atkS + decS + Math.max(0, noteS - atkS - decS) + relS;
            const onePass = notes.length * noteS;
            const now = ctx.currentTime;

            function adsrNote(freq, t, g) {
                const gn = ctx.createGain();
                gn.connect(comp);
                gn.gain.setValueAtTime(0.001, t);
                gn.gain.linearRampToValueAtTime(g, t + atkS);
                gn.gain.linearRampToValueAtTime(g * susLv, t + atkS + decS);
                const susEnd = t + atkS + decS + Math.max(0, noteS - atkS - decS);
                gn.gain.setValueAtTime(g * susLv, susEnd);
                gn.gain.linearRampToValueAtTime(0.001, susEnd + relS);
                const o = ctx.createOscillator();
                o.type = 'sine'; o.frequency.value = freq;
                o.connect(gn); o.start(t); o.stop(susEnd + relS + 0.02);
            }

            for (let r = 0; r < repeats; r++) {
                const base = now + r * (onePass + gapS);
                notes.forEach((f, i) => {
                    adsrNote(f, base + i * noteS, gain);
                    if (overtone > 0) adsrNote(f * 2, base + i * noteS, gain * overtone);
                });
            }

            const totalDur = repeats * (onePass + gapS);
            setTimeout(resolve, totalDur * 1000 + 100);
        } catch (_) { resolve(); }
    });
}

/**
 * 播放会话开始提示音（onPrepared 时使用）。
 * 大三度上行：C6 (1046Hz) → E6 (1319Hz)，sine，ADSR 包络。
 * @returns {Promise<void>}
 */
export async function playSessionChime() {
    try {
        const ctx = getCtx();
        if (ctx.state === 'suspended') await ctx.resume();

        const comp = ctx.createDynamicsCompressor();
        comp.threshold.value = -20;
        comp.knee.value = 12;
        comp.ratio.value = 4;
        comp.attack.value = 0.003;
        comp.release.value = 0.1;
        comp.connect(ctx.destination);

        const notes = [1046.50, 1318.51];
        const noteS = 0.178, gain = 0.90, atkS = 0.009, decS = 0.021;
        const susLv = 0.21, relS = 0.044, overlapS = 0.040;
        const step = Math.max(0.01, noteS - overlapS);
        const totalNote = atkS + decS + Math.max(0, noteS - atkS - decS) + relS;
        const now = ctx.currentTime;

        let lastOsc = null;
        notes.forEach((freq, i) => {
            const t = now + i * step;
            const gn = ctx.createGain();
            gn.connect(comp);
            gn.gain.setValueAtTime(0.001, t);
            gn.gain.linearRampToValueAtTime(gain, t + atkS);
            gn.gain.linearRampToValueAtTime(gain * susLv, t + atkS + decS);
            const susEnd = t + atkS + decS + Math.max(0, noteS - atkS - decS);
            gn.gain.setValueAtTime(gain * susLv, susEnd);
            gn.gain.linearRampToValueAtTime(0.001, susEnd + relS);
            const o = ctx.createOscillator();
            o.type = 'sine'; o.frequency.value = freq;
            o.connect(gn); o.start(t); o.stop(susEnd + relS + 0.02);
            lastOsc = o;
        });

        if (lastOsc) await new Promise((res) => { lastOsc.onended = res; });
    } catch (_) { /* AudioContext not available */ }
}
