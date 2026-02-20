/**
 * Queue Scenario Integration Tests
 *
 * 使用 BackendSimulator（JS port of _recalc_positions_and_eta）驱动 CountdownTimer，
 * 验证在真实后端行为下，前端倒计时的完整表现。
 *
 * 与 countdown-timer.test.js 的区别：
 * - countdown-timer.test.js: 手工注入值，验证 timer 机械行为
 * - 本文件: BackendSimulator 自动生成值，验证 "后端计算 + 前端展示" 全链路
 *
 * 场景：
 * S1: 单 Worker, pos=1, 无跳变（后端值 = 本地 tick）
 * S2: 多 Worker, position 跳变（前面的人完成 → ETA 下降）
 * S3: 超时 + floor 重置（Worker 超 baseline → floor=15s）
 * S4: Chat 模式 /status 估算（estimateChatEta 与 BackendSimulator 对齐）
 * S5: 长队列批量释放（3 Worker, 5 人排队）
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CountdownTimer } from '../../static/lib/countdown-timer.js';
import { estimateChatEta } from '../../static/lib/chat-eta-estimator.js';
import { BackendSimulator } from './backend-simulator.js';

describe('Queue Scenario Integration (BackendSimulator + CountdownTimer)', () => {
    beforeEach(() => { vi.useFakeTimers(); });
    afterEach(() => { vi.useRealTimers(); });

    /**
     * 辅助：模拟前端 polling 周期
     *
     * @param {CountdownTimer} timer
     * @param {BackendSimulator} sim
     * @param {string} requestId - 要追踪的排队请求 ID
     * @param {number} fromT - 起始仿真时间
     * @param {number} toT   - 结束仿真时间
     * @param {number} pollInterval - 轮询间隔（秒）
     * @returns {number} 最终仿真时间
     */
    function advanceWithPolling(timer, sim, requestId, fromT, toT, pollInterval = 3) {
        let t = fromT;
        while (t < toT) {
            const step = Math.min(pollInterval, toT - t);
            vi.advanceTimersByTime(step * 1000);
            t += step;
            const status = sim.getQueueStatus(requestId, t);
            if (status) {
                timer.update(status.estimated_wait_s, status.position, status.queue_length);
            }
        }
        return t;
    }

    // ================================================================
    // S1: 单 Worker, pos=1, 无跳变
    //
    // 验证核心认知：对于 pos=1 的单队列，后端 remaining = baseline - elapsed
    // 和本地 tick 衰减速率一致，每次 poll 不产生跳变。
    // Worker 提前完成 → stop() 时 remaining > 0。
    // ================================================================

    it('S1: 1 Worker busy 5s, baseline=15s — 后端值与本地 tick 完全一致，Worker 8s 完成时提前 stop', () => {
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));
        const sim = new BackendSimulator({ baselines: { chat: 15 } });

        // Worker 在 t=-5 开始处理
        sim.addWorker('w1', { busy: true, requestType: 'chat', taskStartedAt: -5 });
        sim.enqueue('req1', 'chat');

        // t=0: 入队，后端算出 remaining = 15 - 5 = 10s
        const s0 = sim.getQueueStatus('req1', 0);
        expect(s0.estimated_wait_s).toBe(10);
        expect(s0.position).toBe(1);
        timer.update(s0.estimated_wait_s, s0.position, s0.queue_length);
        expect(timer.remaining).toBe(10);

        // t=0~3: 本地 tick 3s → remaining=7
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(7);

        // t=3: poll → 后端 remaining = 15 - 8 = 7（与本地一致，无跳变）
        const s3 = sim.getQueueStatus('req1', 3);
        expect(s3.estimated_wait_s).toBe(7);
        timer.update(s3.estimated_wait_s, s3.position, s3.queue_length);
        expect(timer.remaining).toBe(7); // 无跳变

        // t=3~6: tick 3s → remaining=4
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(4);

        // t=6: poll → 后端 remaining = 15 - 11 = 4（仍无跳变）
        const s6 = sim.getQueueStatus('req1', 6);
        expect(s6.estimated_wait_s).toBe(4);
        timer.update(s6.estimated_wait_s, s6.position, s6.queue_length);
        expect(timer.remaining).toBe(4);

        // t=6~8: tick 2s → remaining=2
        vi.advanceTimersByTime(2000);
        expect(timer.remaining).toBe(2);

        // t=8: Worker 完成（比 baseline 15s 早 2s）→ stop
        timer.stop();
        expect(timer.active).toBe(false);
        expect(timer.remaining).toBe(2);

        timer.stop();
    });

    // ================================================================
    // S2: 多 Worker, 堆模拟精确估计
    //
    // 堆模拟的核心优势：模拟 dispatch 链后，ETA 已精确考虑交错完成。
    // 对于"按预期完成"的场景，每次 poll 和 dispatch 事件后后端值
    // 都等于本地 tick，不产生跳变。
    //
    // 跳变只在"现实偏离预测"时发生（如 Worker 提前/超时完成），
    // 见 S3 (floor) 和 countdown-timer.test.js E1/E2 (提前完成)。
    // ================================================================

    it('S2: 2 Worker, 3 人排队, user pos=3 — 堆模拟精确跟踪，平滑衰减无跳变', () => {
        const timer = new CountdownTimer(() => {});
        const sim = new BackendSimulator({ baselines: { chat: 15 } });

        // 2 Worker: w1(remaining=5), w2(remaining=12)
        sim.addWorker('w1', { busy: true, requestType: 'chat', taskStartedAt: -10 });
        sim.addWorker('w2', { busy: true, requestType: 'chat', taskStartedAt: -3 });
        sim.enqueue('p1', 'chat');  // pos=1
        sim.enqueue('p2', 'chat');  // pos=2
        sim.enqueue('me', 'chat');  // pos=3

        // t=0: 堆模拟 dispatch 链：
        // heap=[(5,w1),(12,w2)]
        // p1→pop(5,w1), push(20,w1). p2→pop(12,w2), push(27,w2). me→pop(20,w1). ETA=20
        const s0 = sim.getQueueStatus('me', 0);
        expect(s0.position).toBe(3);
        expect(s0.estimated_wait_s).toBe(20);
        timer.update(s0.estimated_wait_s, s0.position, s0.queue_length);

        // t=0~3: tick → remaining=17
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(17);

        // t=3: poll — heap=[(2,w1),(9,w2)] → p1:2, p2:9, me:17. 无跳变
        const s3 = sim.getQueueStatus('me', 3);
        expect(s3.estimated_wait_s).toBe(17);
        timer.update(s3.estimated_wait_s, s3.position, s3.queue_length);
        expect(timer.remaining).toBe(17);

        // t=3~5: tick 2s → remaining=15
        vi.advanceTimersByTime(2000);
        expect(timer.remaining).toBe(15);

        // t=5: w1 completes → dispatches p1
        // heap=[(7,w2),(15,w1/p1)] → p2:7, me:15. 无跳变（本地也是 15）
        sim.completeWorker('w1', 5);
        const s5 = sim.getQueueStatus('me', 5);
        expect(s5.position).toBe(2);
        expect(s5.estimated_wait_s).toBe(15);
        timer.update(s5.estimated_wait_s, s5.position, s5.queue_length);
        expect(timer.remaining).toBe(15);

        // t=5~8: tick 3s → remaining=12
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(12);

        // t=8: poll — heap=[(4,w2),(12,w1)] → p2:4, me:12. 无跳变
        const s8 = sim.getQueueStatus('me', 8);
        expect(s8.estimated_wait_s).toBe(12);
        timer.update(s8.estimated_wait_s, s8.position, s8.queue_length);
        expect(timer.remaining).toBe(12);

        // t=8~12: tick 4s → remaining=8
        vi.advanceTimersByTime(4000);
        expect(timer.remaining).toBe(8);

        // t=12: w2 completes → dispatches p2
        // heap=[(8,w1),(15,w2)] → me:8. 无跳变
        sim.completeWorker('w2', 12);
        const s12 = sim.getQueueStatus('me', 12);
        expect(s12.position).toBe(1);
        expect(s12.estimated_wait_s).toBe(8);
        timer.update(s12.estimated_wait_s, s12.position, s12.queue_length);
        expect(timer.remaining).toBe(8);

        // t=12~20: w1 finishes p1 → dispatches me → stop
        vi.advanceTimersByTime(8000);
        expect(timer.remaining).toBe(0);
        timer.stop();
    });

    // ================================================================
    // S3: 超时 + floor 重置
    //
    // Worker 运行超过 baseline → 后端 remaining = floor (15s)
    // countdown 到 0 后被 floor 重置，循环直到 Worker 完成。
    // ================================================================

    it('S3: baseline=15s, Worker 跑了 35s — 到 0 后持续超时计时', () => {
        const timer = new CountdownTimer(() => {});
        const sim = new BackendSimulator({ baselines: { duplex: 15 }, floor: 15 });

        sim.addWorker('w1', { busy: true, requestType: 'duplex', taskStartedAt: 0 });
        sim.enqueue('req1', 'duplex');

        // t=0: remaining = 15
        const s0 = sim.getQueueStatus('req1', 0);
        expect(s0.estimated_wait_s).toBe(15);
        timer.update(s0.estimated_wait_s, s0.position, s0.queue_length);

        // t=0~12: 正常衰减，每 3s poll 确认无跳变
        advanceWithPolling(timer, sim, 'req1', 0, 12, 3);
        expect(timer.remaining).toBe(3);

        // t=12~15: tick 3s → remaining=0
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(0);

        // t=15: poll → floor=15，但 remaining ≤ 0 → Math.min(0, 15) = 0，无重置
        const s15 = sim.getQueueStatus('req1', 15);
        expect(s15.estimated_wait_s).toBe(15);
        timer.update(s15.estimated_wait_s, s15.position, s15.queue_length);
        expect(timer.remaining).toBe(0);

        // t=15~27: 继续 tick + poll，remaining 持续为负
        advanceWithPolling(timer, sim, 'req1', 15, 27, 3);
        // 每 3s tick -3, poll floor 被 Math.min 忽略
        expect(timer.remaining).toBe(-12);

        // t=27~35: 继续超时计时
        vi.advanceTimersByTime(8000);
        expect(timer.remaining).toBe(-20);

        // t=35: Worker 终于完成 → stop
        timer.stop();
        expect(timer.remaining).toBe(-20);
    });

    // ================================================================
    // S4: Chat 模式 /status 估算
    //
    // 验证 estimateChatEta() 从 /status 响应算出的值
    // 和 BackendSimulator 的 ETA 一致。
    // ================================================================

    it('S4: Chat estimateChatEta 与 BackendSimulator 在不同时刻输出一致', () => {
        const sim = new BackendSimulator({ baselines: { chat: 15 } });
        sim.addWorker('w1', { busy: true, requestType: 'chat', taskStartedAt: 0 });
        sim.enqueue('chat_req', 'chat');

        // 单 Worker + 1 人排队 → Chat 估算应和精确 ETA 接近
        for (const t of [0, 3, 6, 9, 12]) {
            const precise = sim.getQueueStatus('chat_req', t);
            const statusResp = sim.getStatusResponse(t);
            const chatEta = estimateChatEta(statusResp);

            // Chat 模式用 avgRemaining + (queue_length-1)*15 估算
            // 单 Worker + 1 排队 → avgRemaining = precise remaining, queue_length=1
            // chatEta = max(1, round(avgRemaining + (1-1)*15)) = round(avgRemaining)
            // precise.estimated_wait_s 也等于 remaining（pos=1, rounds=0）
            // 所以应该相等（误差 ≤ 1 due to rounding）
            expect(Math.abs(chatEta - precise.estimated_wait_s)).toBeLessThanOrEqual(1);
        }
    });

    it('S4b: Chat 2 Worker + 2 排队 — estimateChatEta 粗估 vs 精确 ETA 差异合理', () => {
        const sim = new BackendSimulator({ baselines: { chat: 15 } });
        sim.addWorker('w1', { busy: true, requestType: 'chat', taskStartedAt: -3 });
        sim.addWorker('w2', { busy: true, requestType: 'chat', taskStartedAt: -8 });
        sim.enqueue('other', 'chat');
        sim.enqueue('me', 'chat');

        // t=0
        const precise = sim.getQueueStatus('me', 0);
        const statusResp = sim.getStatusResponse(0);
        const chatEta = estimateChatEta(statusResp);

        // Chat 是粗估（avg of all running tasks），精确 ETA 考虑了 position 和 rounds
        // 差异合理即可（不超过 1 个 baseline 周期）
        expect(chatEta).toBeGreaterThan(0);
        expect(precise.estimated_wait_s).toBeGreaterThan(0);
        expect(Math.abs(chatEta - precise.estimated_wait_s)).toBeLessThan(30);
    });

    // ================================================================
    // S5: 长队列批量释放
    //
    // 3 Worker, 5 人排队。Worker 陆续完成 → position 批量下降。
    // ================================================================

    it('S5: 3 Worker, 5 人排队 — 堆模拟精确跟踪，全程平滑衰减', () => {
        const timer = new CountdownTimer(() => {});
        const sim = new BackendSimulator({ baselines: { chat: 15 } });

        // 3 Worker，交错启动
        sim.addWorker('w1', { busy: true, requestType: 'chat', taskStartedAt: -10 }); // rem=5
        sim.addWorker('w2', { busy: true, requestType: 'chat', taskStartedAt: -5 });  // rem=10
        sim.addWorker('w3', { busy: true, requestType: 'chat', taskStartedAt: -2 });  // rem=13

        sim.enqueue('p1', 'chat');
        sim.enqueue('p2', 'chat');
        sim.enqueue('p3', 'chat');
        sim.enqueue('p4', 'chat');
        sim.enqueue('me', 'chat'); // pos=5

        // t=0: 堆模拟 dispatch 链：
        // heap=[(5,w1),(10,w2),(13,w3)]
        // p1→pop(5,w1),push(20). p2→pop(10,w2),push(25). p3→pop(13,w3),push(28).
        // p4→pop(20,w1),push(35). me→pop(25,w2). ETA=25
        const s0 = sim.getQueueStatus('me', 0);
        expect(s0.position).toBe(5);
        expect(s0.estimated_wait_s).toBe(25);
        timer.update(s0.estimated_wait_s, s0.position, s0.queue_length);

        // t=0~5: tick 5s → remaining=20
        vi.advanceTimersByTime(5000);
        expect(timer.remaining).toBe(20);

        // t=5: w1 completes → dispatches p1. Queue: [p2,p3,p4,me]
        // heap=[(5,w2),(8,w3),(15,w1/p1)]
        // p2→pop(5),push(20). p3→pop(8),push(23). p4→pop(15),push(30). me→pop(20). ETA=20
        sim.completeWorker('w1', 5);
        const s5 = sim.getQueueStatus('me', 5);
        expect(s5.position).toBe(4);
        expect(s5.estimated_wait_s).toBe(20);
        timer.update(s5.estimated_wait_s, s5.position, s5.queue_length);
        expect(timer.remaining).toBe(20); // 无跳变（精确跟踪）

        // t=5~10: tick 5s → remaining=15
        vi.advanceTimersByTime(5000);
        expect(timer.remaining).toBe(15);

        // t=10: w2 completes → dispatches p2. Queue: [p3,p4,me]
        // heap=[(3,w3),(10,w1),(15,w2/p2)]
        // p3→pop(3),push(18). p4→pop(10),push(25). me→pop(15). ETA=15
        sim.completeWorker('w2', 10);
        const s10 = sim.getQueueStatus('me', 10);
        expect(s10.position).toBe(3);
        expect(s10.estimated_wait_s).toBe(15);
        timer.update(s10.estimated_wait_s, s10.position, s10.queue_length);
        expect(timer.remaining).toBe(15); // 无跳变

        // t=10~13: tick 3s → remaining=12
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(12);

        // t=13: w3 completes → dispatches p3. Queue: [p4,me]
        // heap=[(7,w1),(12,w2),(15,w3/p3)]
        // p4→pop(7),push(22). me→pop(12). ETA=12
        sim.completeWorker('w3', 13);
        const s13 = sim.getQueueStatus('me', 13);
        expect(s13.position).toBe(2);
        expect(s13.estimated_wait_s).toBe(12);
        timer.update(s13.estimated_wait_s, s13.position, s13.queue_length);
        expect(timer.remaining).toBe(12); // 无跳变

        // t=13~20: tick 7s → remaining=5
        vi.advanceTimersByTime(7000);
        expect(timer.remaining).toBe(5);

        // t=20: w1 finishes p1 → dispatches p4. Queue: [me]
        // heap=[(5,w2),(8,w3),(15,w1/p4)] → me→pop(5). ETA=5
        sim.completeWorker('w1', 20);
        const s20 = sim.getQueueStatus('me', 20);
        expect(s20.position).toBe(1);
        expect(s20.queue_length).toBe(1);
        expect(s20.estimated_wait_s).toBe(5);
        timer.update(s20.estimated_wait_s, s20.position, s20.queue_length);
        expect(timer.remaining).toBe(5); // 无跳变

        // t=20~25: w2 finishes p2 → me 被分配 → stop
        vi.advanceTimersByTime(5000);
        expect(timer.remaining).toBe(0);
        timer.stop();
        expect(timer.active).toBe(false);
    });
});
