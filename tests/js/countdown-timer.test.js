/**
 * CountdownTimer 单元测试
 *
 * 覆盖场景：
 * A. 基础行为（1-5）：update/tick/stop/interval 不泄漏/remaining 不变负
 * B. 后端刷新（6-7）：中途重置 remaining、position 推进
 * C. 停止与重启（8-9）：stop 幂等、stop 后重启
 * D. 回调与类型（10-12）：onRender 调用次数、小数四舍五入、null position
 * E. Worker 比 baseline 快完成（13-14）：countdown 提前终止（pos=1 无跳变，只有 stop 提前）
 * F. Worker 比 baseline 慢/超时（15-17）：countdown 到 0 后被后端 floor=15s 重置
 * G. 完整轮询修正周期（18-23）：无跳变确认、position 跳变向下修正、floor+position 混合、长队列批量释放
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CountdownTimer } from '../../static/lib/countdown-timer.js';

describe('CountdownTimer', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    // ====== 1. 基本倒计时 ======

    it('update 设置 remaining 并启动 interval', () => {
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        timer.update(10, 1, 3);

        expect(timer.remaining).toBe(10);
        expect(timer.position).toBe(1);
        expect(timer.queueLength).toBe(3);
        expect(timer.active).toBe(true);
        expect(renders).toHaveLength(1);
        expect(renders[0]).toEqual({ remaining: 10, position: 1, queueLength: 3 });

        timer.stop();
    });

    it('每秒 tick 递减 remaining', () => {
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        timer.update(5, 1, 1);
        expect(timer.remaining).toBe(5);

        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(2);
        // 1 (update) + 3 (ticks) = 4 renders
        expect(renders).toHaveLength(4);

        timer.stop();
    });

    // ====== 2. 后端刷新重置 ======

    it('中途 update 重置 remaining 为新的后端值', () => {
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        timer.update(15, 1, 2);
        vi.advanceTimersByTime(5000); // remaining: 15 → 10
        expect(timer.remaining).toBe(10);

        // 后端推送新的 ETA（真实剩余 10s）
        timer.update(10, 1, 2);
        expect(timer.remaining).toBe(10);

        vi.advanceTimersByTime(3000); // remaining: 10 → 7
        expect(timer.remaining).toBe(7);

        timer.stop();
    });

    it('update 更新 position 和 queueLength', () => {
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        timer.update(20, 3, 5);
        vi.advanceTimersByTime(3000);

        // 前面的人离开了，position 从 3 变 2
        timer.update(15, 2, 4);
        expect(timer.position).toBe(2);
        expect(timer.queueLength).toBe(4);
        expect(timer.remaining).toBe(15);

        timer.stop();
    });

    // ====== 3. 停止与清理 ======

    it('stop 后 interval 不再 tick', () => {
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        timer.update(10, 1, 1);
        vi.advanceTimersByTime(2000); // 2 ticks
        expect(timer.remaining).toBe(8);

        timer.stop();
        expect(timer.active).toBe(false);

        vi.advanceTimersByTime(5000); // 不应再 tick
        expect(timer.remaining).toBe(8); // 保持不变
        // renders: 1 (update) + 2 (ticks) = 3, 之后不再增加
        expect(renders).toHaveLength(3);
    });

    // ====== 4. 多次 update 不泄漏 interval ======

    it('连续多次 update 只创建一个 interval', () => {
        const timer = new CountdownTimer(() => {});

        timer.update(10, 1, 1);
        timer.update(20, 2, 3);
        timer.update(30, 3, 5);

        // 只创建了 1 个 interval（_intervalId 不为 null）
        expect(timer.active).toBe(true);

        vi.advanceTimersByTime(1000);
        // 最后一次 update 设了 remaining=30, 1 tick 后 = 29
        expect(timer.remaining).toBe(29);

        timer.stop();
    });

    // ====== 5. remaining 到 0 后继续为负（超时计时） ======

    it('remaining 到 0 后继续为负数', () => {
        const timer = new CountdownTimer(() => {});

        timer.update(3, 1, 1);
        vi.advanceTimersByTime(5000); // 5 ticks: 3→2→1→0→-1→-2

        expect(timer.remaining).toBe(-2);

        timer.stop();
    });

    // ====== 6. 用户场景：eta=15s, 5s 后入队 ======

    it('模拟：Worker eta=15s, t=5s 入队，后端推送 ~10s', () => {
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        // t=5s: 入队，后端计算 remaining = 15 - 5 = 10s
        timer.update(10, 1, 1);
        expect(timer.remaining).toBe(10);

        // 前端本地 tick 5s → remaining=5
        vi.advanceTimersByTime(5000);
        expect(timer.remaining).toBe(5);

        // t=10s: 后端推送更新，remaining = 15 - 8 = 7s
        // 但本地已 tick 到 5，单调递减 → Math.min(5, 7) = 5（不跳回）
        timer.update(7, 1, 1);
        expect(timer.remaining).toBe(5);

        // 继续 tick 到 0
        vi.advanceTimersByTime(5000);
        expect(timer.remaining).toBe(0);

        // 继续 tick 为负（超时计时）
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(-3);

        timer.stop();
    });

    // ====== 7. 多 Worker 交错 ======

    it('模拟：2 Worker, 入队 pos=1 eta=10, 然后 pos 推进', () => {
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        // 初始：pos=2, total=3, eta=25s
        timer.update(25, 2, 3);
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(22);

        // 3s 后后端推送：前面有人完成，pos=1, total=2, eta=18s
        timer.update(18, 1, 2);
        expect(timer.remaining).toBe(18);
        expect(timer.position).toBe(1);
        expect(timer.queueLength).toBe(2);

        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(15);

        // 又 3s：pos=1, total=1, eta=12s
        timer.update(12, 1, 1);
        expect(timer.remaining).toBe(12);

        timer.stop();
    });

    // ====== 8. stop 幂等 ======

    it('多次 stop 不报错', () => {
        const timer = new CountdownTimer(() => {});
        timer.update(5, 1, 1);

        timer.stop();
        timer.stop();
        timer.stop();

        expect(timer.active).toBe(false);
    });

    it('未启动时 stop 不报错', () => {
        const timer = new CountdownTimer(() => {});
        timer.stop();
        expect(timer.active).toBe(false);
    });

    // ====== 9. stop 再 update — 重新启动 ======

    it('stop 后再 update 可以重新启动', () => {
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        timer.update(10, 1, 1);
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(7);

        timer.stop();
        expect(timer.active).toBe(false);

        // 重新启动
        timer.update(20, 2, 5);
        expect(timer.active).toBe(true);
        expect(timer.remaining).toBe(20);

        vi.advanceTimersByTime(2000);
        expect(timer.remaining).toBe(18);

        timer.stop();
    });

    // ====== 10. onRender 回调验证 ======

    it('每次 update 和 tick 都调用 onRender', () => {
        const fn = vi.fn();
        const timer = new CountdownTimer(fn);

        timer.update(3, 1, 1);
        expect(fn).toHaveBeenCalledTimes(1);
        expect(fn).toHaveBeenLastCalledWith({ remaining: 3, position: 1, queueLength: 1 });

        vi.advanceTimersByTime(1000);
        expect(fn).toHaveBeenCalledTimes(2);
        expect(fn).toHaveBeenLastCalledWith({ remaining: 2, position: 1, queueLength: 1 });

        vi.advanceTimersByTime(1000);
        expect(fn).toHaveBeenCalledTimes(3);
        expect(fn).toHaveBeenLastCalledWith({ remaining: 1, position: 1, queueLength: 1 });

        timer.stop();
    });

    // ====== 11. 小数 eta 四舍五入 ======

    it('eta 为小数时四舍五入', () => {
        const timer = new CountdownTimer(() => {});

        timer.update(10.7, 1, 1);
        expect(timer.remaining).toBe(11);

        timer.update(10.3, 1, 1);
        expect(timer.remaining).toBe(10);

        timer.stop();
    });

    // ====== 12. position/queueLength 可以为 null ======

    it('position 和 queueLength 可以为 null（Chat 模式粗估）', () => {
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        timer.update(15, null, null);
        expect(renders[0]).toEqual({ remaining: 15, position: null, queueLength: null });

        vi.advanceTimersByTime(2000);
        expect(renders[renders.length - 1]).toEqual({ remaining: 13, position: null, queueLength: null });

        timer.stop();
    });

    // ================================================================
    // E. Worker 比 baseline 快完成 — countdown 提前终止
    //
    // 关键认知：后端用 remaining = baseline - elapsed 计算。
    // 对于 pos=1 的单队列，后端值和本地 tick 衰减速率一致，不会产生跳变。
    // "比预期快"的体验 = 倒计时还没到 0 时 Worker 就完成了 → stop() 提前触发。
    // ================================================================

    it('E1: baseline=15s, Worker 8s 完成 — 倒计时还剩 7s 时提前连上', () => {
        // 1 Worker 忙碌，用户排 pos=1/1
        // 后端 baseline=15s，每 3s 轮询 → 后端给的值和本地一致（无跳变）
        // Worker 实际 8s 完成 → 用户在 remaining=7 时就被服务
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        // t=0: 入队，后端给 eta=15s
        timer.update(15, 1, 1);
        expect(timer.remaining).toBe(15);

        // t=0~3: 本地 tick
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(12);

        // t=3: 轮询 → 后端 remaining=15-3=12s（和本地一致，无跳变）
        timer.update(12, 1, 1);
        expect(timer.remaining).toBe(12);

        // t=3~6: 本地 tick
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(9);

        // t=6: 轮询 → 后端 remaining=15-6=9s（仍无跳变）
        timer.update(9, 1, 1);
        expect(timer.remaining).toBe(9);

        // t=6~8: 本地 tick 2s → remaining=7
        vi.advanceTimersByTime(2000);
        expect(timer.remaining).toBe(7);

        // t=8: Worker 完成！后端推 queue_done → stop
        // 用户体验：倒计时显示"还有~7s"时就连上了 → 比预期快
        timer.stop();
        expect(timer.active).toBe(false);
        expect(timer.remaining).toBe(7); // 冻结

        // stop 后不再 tick
        vi.advanceTimersByTime(5000);
        expect(timer.remaining).toBe(7);
    });

    it('E2: baseline=15s, Worker 2s 完成 — 首次轮询前就结束', () => {
        // Worker 极快完成（2s），甚至还没到第一次 3s 轮询就结束了
        const timer = new CountdownTimer(() => {});

        // t=0: 入队 eta=15s
        timer.update(15, 1, 1);

        // t=0~2: 本地 tick 2s → remaining=13
        vi.advanceTimersByTime(2000);
        expect(timer.remaining).toBe(13);

        // t=2: Worker 完成 → stop（还没来得及轮询）
        timer.stop();
        expect(timer.active).toBe(false);
        expect(timer.remaining).toBe(13); // 冻结，用户看到"~13s"时就连上了
    });

    // ================================================================
    // F. 后端比预期慢（超时） — countdown 到 0 后被后端 floor 重置
    // ================================================================

    it('F1: eta=15s 但 Worker 实际花了 30s — 到 0 后继续计超时时间', () => {
        // 到 0 后不再 floor 重置，而是继续 tick 为负数表示超时
        const timer = new CountdownTimer(() => {});

        timer.update(15, 1, 1);

        // 正常倒数 15s → 0
        vi.advanceTimersByTime(15000);
        expect(timer.remaining).toBe(0);

        // 继续 tick → 负数（超时计时）
        vi.advanceTimersByTime(5000);
        expect(timer.remaining).toBe(-5);

        // t=20: 后端轮询推 floor=15，但 remaining < 0 → Math.min(-5, 15) = -5
        timer.update(15, 1, 1);
        expect(timer.remaining).toBe(-5);

        // 继续超时计时
        vi.advanceTimersByTime(10000);
        expect(timer.remaining).toBe(-15);

        // t=30: Worker 终于完成 → stop
        timer.stop();
        expect(timer.remaining).toBe(-15); // 冻结在超时 15s
    });

    it('F2: eta=15s, Worker 20s 完成 — 超时 5s 后 Worker 完成', () => {
        const timer = new CountdownTimer(() => {});

        timer.update(15, 1, 1);

        // tick 15s → 0
        vi.advanceTimersByTime(15000);
        expect(timer.remaining).toBe(0);

        // 继续 tick 5s → -5（超时 5s）
        vi.advanceTimersByTime(5000);
        expect(timer.remaining).toBe(-5);

        // t=20: Worker 完成 → stop
        timer.stop();
        expect(timer.remaining).toBe(-5);
    });

    it('F3: eta=15s, Worker 超时很久（45s）— 超时计时持续递增', () => {
        const timer = new CountdownTimer(() => {});

        timer.update(15, 1, 1);

        // 倒数到 0
        vi.advanceTimersByTime(15000);
        expect(timer.remaining).toBe(0);

        // 继续 tick → 超时计时
        vi.advanceTimersByTime(15000); // t=30: -15
        expect(timer.remaining).toBe(-15);

        vi.advanceTimersByTime(15000); // t=45: -30
        expect(timer.remaining).toBe(-30);

        // t=45: Worker 终于完成 → stop
        timer.stop();
        expect(timer.remaining).toBe(-30);
    });

    // ================================================================
    // G. 完整轮询修正周期 — 模拟前端 3s 轮询间隔的实际交互
    // ================================================================

    it('G1: 单 Worker 排 1 人，前端每 3s 轮询，后端正常完成', () => {
        // 完整时间轴：前端表现 vs 后端实际
        // 后端 baseline=15s, 实际 Worker 12s 完成
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        // t=0: 入队，后端推 eta=15s
        timer.update(15, 1, 1);

        // t=0~3: 本地 tick 3 次
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(12);
        expect(renders[renders.length - 1].remaining).toBe(12);

        // t=3: 轮询 → 后端 remaining=15-3=12s（正常）
        timer.update(12, 1, 1);
        expect(timer.remaining).toBe(12); // 无跳变

        // t=3~6: 本地 tick 3 次
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(9);

        // t=6: 轮询 → 后端 remaining=15-6=9s（正常）
        timer.update(9, 1, 1);
        expect(timer.remaining).toBe(9); // 无跳变

        // t=6~9: tick 3s
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(6);

        // t=9: 轮询 → 后端 remaining=15-9=6s
        timer.update(6, 1, 1);

        // t=9~12: tick 3s → remaining=3
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(3);

        // t=12: Worker 提前完成（baseline=15 但实际 12s）→ stop
        timer.stop();
        expect(timer.remaining).toBe(3); // 冻结，用户看到"还有~3s"时就连上了
    });

    it('G2: 3 Worker 排 5 人，前端每 3s 轮询，逐步推进 position 直到轮到自己', () => {
        // 场景：3 Worker 满载，2 人排队，用户是第 2 位 (pos=2, total=2)
        // 后端 eta: 每个 Worker 15s baseline
        // Worker 释放顺序：t=8 第一个 Worker 完成 → pos=1，t=20 再完成一个 → 轮到自己
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        // t=0: 入队 pos=2/2, eta=25s（pos1 等 ~10s + 自己等 ~15s）
        timer.update(25, 2, 2);
        expect(timer.remaining).toBe(25);

        // t=0~3: 本地 tick
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(22);

        // t=3: 轮询 → 后端推 pos=2/2, eta=22s（正常衰减）
        timer.update(22, 2, 2);
        expect(timer.remaining).toBe(22);

        // t=3~6: tick
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(19);

        // t=6: 轮询 → 后端推 pos=2/2, eta=19s
        timer.update(19, 2, 2);

        // t=6~8: tick 2s
        vi.advanceTimersByTime(2000);
        expect(timer.remaining).toBe(17);

        // t=8: Worker 完成！pos 推进
        // （WS 推送不需要等轮询）
        timer.update(15, 1, 1);
        expect(timer.remaining).toBe(15);
        expect(timer.position).toBe(1);
        expect(timer.queueLength).toBe(1);

        // t=8~11: tick 3s
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(12);

        // t=11: 轮询 → 后端推 pos=1/1, eta=12s
        timer.update(12, 1, 1);

        // t=11~14: tick 3s
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(9);

        // t=14: 轮询 → 后端推 pos=1/1, eta=9s
        timer.update(9, 1, 1);

        // t=14~17: tick 3s
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(6);

        // t=17~20: tick 3s
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(3);

        // t=20: Worker 完成 → 轮到自己 → stop
        timer.stop();
        expect(timer.remaining).toBe(3);
    });

    it('G3: Chat 模式 — 无精确 pos，前端轮询 /status 推断 ETA，正常衰减 + 提前完成', () => {
        // Chat 模式没有 WS，前端 3s 轮询 /status
        // 前端从 running_tasks 的 estimated_remaining_s 推断 ETA
        // 该值也是后端 baseline - elapsed 算的，和本地 tick 一致
        const timer = new CountdownTimer(() => {});

        // t=3: 第一次轮询（发送请求后 3s），发现有 busy worker
        // 后端 running_tasks[0].estimated_remaining_s = 15 - 3 = 12s
        timer.update(12, null, null);
        expect(timer.remaining).toBe(12);

        // t=3~6: 本地 tick
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(9);

        // t=6: 轮询 → 后端 remaining = 15 - 6 = 9s（和本地一致）
        timer.update(9, null, null);
        expect(timer.remaining).toBe(9);

        // t=6~9: tick
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(6);

        // t=9: 轮询 → 后端 remaining = 15 - 9 = 6s
        timer.update(6, null, null);
        expect(timer.remaining).toBe(6);

        // t=9~10: Worker 10s 完成（比 baseline 15s 快）
        vi.advanceTimersByTime(1000);
        expect(timer.remaining).toBe(5);

        // t=10: Chat 请求被处理 → stop
        // 用户看到"~5s"时请求已返回结果
        timer.stop();
        expect(timer.remaining).toBe(5);
    });

    it('G4: 2 Worker 排 3 人 — 前面的人完成导致 position 跳变，ETA 向下修正', () => {
        // 场景：2 Worker 满载，3 人排队（pos=3/3）
        // 跳变只在 position 变化时发生（前面的人被服务完成）
        // 这是 remaining 向下修正的唯一合理来源
        const timer = new CountdownTimer(() => {});

        // t=0: 入队 pos=3/3, eta=30s（前 2 人各约 15s）
        timer.update(30, 3, 3);

        // t=0~3: 本地 tick
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(27);

        // t=3: 轮询 → 后端 eta=30-3=27（无跳变）
        timer.update(27, 3, 3);
        expect(timer.remaining).toBe(27);

        // t=3~6: tick
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(24);

        // t=6: 轮询 → 后端 eta=30-6=24
        timer.update(24, 3, 3);

        // t=6~8: tick 2s
        vi.advanceTimersByTime(2000);
        expect(timer.remaining).toBe(22);

        // t=8: Worker 完成！pos=2/2, 后端重算 eta=15s
        // 跳变：本地 22 → 后端推 15（向下修正，因为 position 变了）
        timer.update(15, 2, 2);
        expect(timer.remaining).toBe(15);
        expect(timer.position).toBe(2);

        // t=8~11: tick
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(12);

        // t=11: 轮询 → 后端 eta=15-3=12（无跳变）
        timer.update(12, 2, 2);

        // t=11~14: tick
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(9);

        // t=14: 又一个 Worker 完成！pos=1/1, 后端重算 eta=10s
        // 跳变：本地 9 → 后端推 10（position 变了，但实际差异不大）
        timer.update(10, 1, 1);
        expect(timer.remaining).toBe(10);
        expect(timer.position).toBe(1);

        // t=14~20: tick 6s
        vi.advanceTimersByTime(6000);
        expect(timer.remaining).toBe(4);

        // t=20: Worker 完成 → 轮到自己 → stop
        timer.stop();
        expect(timer.remaining).toBe(4);
    });

    it('G5: Duplex 超时 + position 变化混合 — position 重置后倒数到 0 继续超时计时', () => {
        // 2 Worker，用户 pos=2/2，Worker 1 超时
        const timer = new CountdownTimer(() => {});

        timer.update(20, 2, 2);

        vi.advanceTimersByTime(12000);
        expect(timer.remaining).toBe(8);

        // t=12: Worker 2 完成！pos=1/1
        // position 变化 (2→1) → 重置为 15（Worker 1 超时中）
        timer.update(15, 1, 1);
        expect(timer.remaining).toBe(15);
        expect(timer.position).toBe(1);

        // t=12~27: 倒数 15s → 0
        vi.advanceTimersByTime(15000);
        expect(timer.remaining).toBe(0);

        // t=27~30: 超时计时
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(-3);

        // 后端轮询 floor=15 → 不重置
        timer.update(15, 1, 1);
        expect(timer.remaining).toBe(-3);

        // t=30~32: Worker 1 终于完成
        vi.advanceTimersByTime(2000);
        expect(timer.remaining).toBe(-5);

        timer.stop();
        expect(timer.remaining).toBe(-5);
    });

    it('G6: 长队列（pos=5）— 每次 Worker 完成时后端推送 position 递减 + ETA 修正', () => {
        // 模拟 3 Worker, 队列 5 人，baseline 15s
        // Worker 平均 15s 一波释放，每释放一波 position-=1 或更多
        const renders = [];
        const timer = new CountdownTimer(s => renders.push({ ...s }));

        // t=0: 入队 pos=5/5, eta=35s（约 2 轮等待 × 15s + 5s 余量）
        timer.update(35, 5, 5);

        // t=3: 轮询 → eta=32
        vi.advanceTimersByTime(3000);
        timer.update(32, 5, 5);

        // t=6: 轮询 → eta=29
        vi.advanceTimersByTime(3000);
        timer.update(29, 5, 5);

        // t=10: Worker 释放！ 3 人一波完成 → pos=2/2, eta=12s
        vi.advanceTimersByTime(4000);
        expect(timer.remaining).toBe(25); // 本地 tick 4s: 29-4=25
        timer.update(12, 2, 2); // 后端大幅修正
        expect(timer.remaining).toBe(12); // 向下跳变
        expect(timer.position).toBe(2);
        expect(timer.queueLength).toBe(2);

        // t=13: 轮询 → eta=9
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(9);
        timer.update(9, 2, 2);

        // t=15: 又一个 Worker 完成 → pos=1/1, eta=10s
        vi.advanceTimersByTime(2000);
        timer.update(10, 1, 1);
        expect(timer.position).toBe(1);

        // t=18: tick → 7
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(7);

        // t=21: tick → 4
        vi.advanceTimersByTime(3000);
        expect(timer.remaining).toBe(4);

        // t=25: Worker 完成 → stop
        vi.advanceTimersByTime(4000);
        expect(timer.remaining).toBe(0);
        timer.stop();
    });

    // ================================================================
    // H. 回归测试 — 验证 update() 不会导致倒计时向上跳变
    //
    // 用户报告症状："倒计时一直在跳动，一会 15 一会增加一会减少，
    // 到 0 后甚至跳回 15"
    //
    // 根因：update() 无条件覆盖 remaining，每 3s 轮询都重置，
    // 导致本地 tick 和后端推送打架。
    // ================================================================

    it('H1: 轮询返回值高于本地 tick 时，remaining 不应向上跳变', () => {
        // 用户症状："一会 15 一会增加一会减少"
        // 复现：本地 tick 到 9，后端轮询返回 10（精度差异），remaining 跳回 10
        const timer = new CountdownTimer(() => {});

        timer.update(12, 1, 1);
        vi.advanceTimersByTime(3000); // tick 3s → remaining=9
        expect(timer.remaining).toBe(9);

        // 后端轮询返回 10（稍高于本地），不应跳回
        timer.update(10, 1, 1);
        expect(timer.remaining).toBe(9); // 核心断言：单调递减
    });

    it('H2: 倒计时归零后轮询返回 floor=15 不应重置，继续超时计时', () => {
        // 用户症状："到 0 之后甚至还有可能跳回 15"
        // 修复：到 0 后 tick 为负数，floor 更新被 Math.min 忽略
        const timer = new CountdownTimer(() => {});

        timer.update(15, 1, 1);
        vi.advanceTimersByTime(15000); // tick → 0
        expect(timer.remaining).toBe(0);

        vi.advanceTimersByTime(3000); // tick → -3
        expect(timer.remaining).toBe(-3);

        // 后端轮询推 floor=15 → Math.min(-3, 15) = -3
        timer.update(15, 1, 1);
        expect(timer.remaining).toBe(-3); // 核心断言：不跳回 15

        vi.advanceTimersByTime(3000); // tick → -6
        timer.update(15, 1, 1);
        expect(timer.remaining).toBe(-6); // 继续超时计时
    });

    it('H3: 模拟真实 3s 轮询周期 — 倒计时应平滑递减，无任何向上跳变', () => {
        // 端到端模拟：1 Worker busy, baseline=15s
        // 每 3s 轮询，后端返回值比本地稍高（round 偏差 +1）
        // 倒计时应始终平滑递减，后端高值被 Math.min 忽略
        const renders = [];
        const timer = new CountdownTimer(s => renders.push(s.remaining));

        timer.update(15, 1, 1); // t=0: remaining=15

        // 4 轮 polling，后端每次比本地高 1（round 偏差）
        vi.advanceTimersByTime(3000); // t=3: local=12
        timer.update(13, 1, 1);      // 后端返回 13 → min(12,13)=12

        vi.advanceTimersByTime(3000); // t=6: local=9
        timer.update(10, 1, 1);      // 后端返回 10 → min(9,10)=9

        vi.advanceTimersByTime(3000); // t=9: local=6
        timer.update(7, 1, 1);       // 后端返回 7 → min(6,7)=6

        vi.advanceTimersByTime(3000); // t=12: local=3
        timer.update(4, 1, 1);       // 后端返回 4 → min(3,4)=3

        // 验证整个 render 历史严格单调递减（无任何向上跳变）
        for (let i = 1; i < renders.length; i++) {
            expect(renders[i]).toBeLessThanOrEqual(renders[i - 1]);
        }

        timer.stop();
    });

    it('H4: 倒计时归零后继续 tick 为负数（超时计时）', () => {
        // 用户期望：到 0 后不跳回 15，而是 0→-1→-2... 表示超时
        const timer = new CountdownTimer(() => {});

        timer.update(3, 1, 1);
        vi.advanceTimersByTime(3000); // tick → 0
        expect(timer.remaining).toBe(0);

        vi.advanceTimersByTime(3000); // tick → -3（超时 3s）
        expect(timer.remaining).toBe(-3);

        // 后端轮询返回 floor=15 → 不应重置，Math.min(-3, 15) = -3
        timer.update(15, 1, 1);
        expect(timer.remaining).toBe(-3);

        vi.advanceTimersByTime(2000); // tick → -5
        expect(timer.remaining).toBe(-5);

        timer.stop();
    });

    it('H5: 超时状态下 position 变化仍然重置', () => {
        // position 变化是唯一合理的重置场景
        const timer = new CountdownTimer(() => {});

        timer.update(3, 2, 2);
        vi.advanceTimersByTime(5000); // tick → -2
        expect(timer.remaining).toBe(-2);

        // position 变化：前面的人完成了
        timer.update(15, 1, 1);
        expect(timer.remaining).toBe(15); // 合理重置
        expect(timer.position).toBe(1);

        timer.stop();
    });

    it('H6: position 变化允许重置，但同 position 下严格单调递减', () => {
        // position 变化（前面的人完成）是唯一合理的"向上重置"场景（position 不同嘛）
        // 同 position 下必须单调递减
        const timer = new CountdownTimer(() => {});

        timer.update(25, 2, 2);
        vi.advanceTimersByTime(5000); // tick → 20

        // position 变化：2→1，ETA 重置为 15 是合理的
        timer.update(15, 1, 1);
        expect(timer.remaining).toBe(15); // 合理：position changed

        vi.advanceTimersByTime(3000); // tick → 12

        // 同 position 下，后端返回 14（比本地高）→ 不应跳回
        timer.update(14, 1, 1);
        expect(timer.remaining).toBe(12); // 核心断言：同 pos 单调递减
    });
});
