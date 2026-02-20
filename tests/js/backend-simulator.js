/**
 * BackendSimulator — JS port of WorkerPool._recalc_positions_and_eta
 *
 * 复刻后端 ETA 计算逻辑，用于场景集成测试。
 * 保证测试中注入 CountdownTimer 的值和真实后端一致，
 * 避免手工编写不合理的 update() 参数。
 *
 * 对应 Python 源码：gateway_modules/worker_pool.py L650-705
 */

/**
 * @typedef {Object} WorkerState
 * @property {string}  id
 * @property {boolean} busy
 * @property {string|null}  requestType  - 'chat' | 'streaming' | 'audio_duplex' | 'omni_duplex'
 * @property {number|null}  taskStartedAt - 任务开始的仿真时间（秒）
 */

/**
 * @typedef {Object} QueueItem
 * @property {string} id
 * @property {string} requestType
 * @property {number} position       - 1-based
 * @property {number} estimatedWaitS
 */

export class BackendSimulator {
    /**
     * @param {Object} opts
     * @param {Object<string, number>} opts.baselines - 每种请求类型的 baseline ETA（秒）
     * @param {number} [opts.floor=15] - 超时兜底值
     */
    constructor({ baselines = { chat: 15, streaming: 15, audio_duplex: 120, omni_duplex: 90 }, floor = 15 } = {}) {
        /** @type {Object<string, number>} */
        this.baselines = { ...baselines };
        this.floor = floor;

        /** @type {Map<string, WorkerState>} */
        this.workers = new Map();

        /** @type {Array<QueueItem>} FIFO 队列 */
        this.queue = [];
    }

    /**
     * 添加 Worker
     * @param {string} id
     * @param {Object} [opts]
     * @param {boolean} [opts.busy=false]
     * @param {string|null} [opts.requestType=null]
     * @param {number|null} [opts.taskStartedAt=null] - 仿真时间（秒）
     */
    addWorker(id, { busy = false, requestType = null, taskStartedAt = null } = {}) {
        this.workers.set(id, { id, busy, requestType, taskStartedAt });
    }

    /**
     * 入队请求（不自动分配 Worker，模拟已满载场景）
     * @param {string} id
     * @param {string} requestType
     */
    enqueue(id, requestType) {
        this.queue.push({ id, requestType, position: 0, estimatedWaitS: 0 });
    }

    /**
     * Worker 完成任务，自动 dispatch 队头（如有）
     * @param {string} workerId
     * @param {number} atTime - 完成时的仿真时间（秒）
     */
    completeWorker(workerId, atTime) {
        const w = this.workers.get(workerId);
        if (!w) throw new Error(`Worker ${workerId} not found`);
        w.busy = false;
        w.requestType = null;
        w.taskStartedAt = null;

        if (this.queue.length > 0) {
            const next = this.queue.shift();
            w.busy = true;
            w.requestType = next.requestType;
            w.taskStartedAt = atTime;
        }
    }

    /**
     * 核心：在仿真时间 t 重算所有排队项的 position 和 ETA
     *
     * 直接对应 Python WorkerPool._recalc_positions_and_eta()
     *
     * @param {number} t - 当前仿真时间（秒）
     */
    recalc(t) {
        if (this.queue.length === 0) return;

        // 1. 初始化：收集 busy Worker 的 (remaining, index) 并排序
        const heap = [];
        let idx = 0;
        for (const w of this.workers.values()) {
            if (w.busy && w.taskStartedAt !== null && w.requestType) {
                const elapsed = t - w.taskStartedAt;
                const eta = this.baselines[w.requestType] || 15;
                const remaining = elapsed < eta ? eta - elapsed : this.floor;
                heap.push({ finishTime: remaining, idx: idx++ });
            }
        }

        if (heap.length === 0) {
            for (const item of this.queue) {
                item.estimatedWaitS = 0;
            }
            return;
        }

        heap.sort((a, b) => a.finishTime - b.finishTime);

        // 2. 模拟 dispatch 链（W<=8，用 sorted array 代替堆，复杂度 O(Q*W)）
        for (let i = 0; i < this.queue.length; i++) {
            const item = this.queue[i];
            item.position = i + 1;

            const earliest = heap.shift();
            item.estimatedWaitS = Math.round(Math.max(0, earliest.finishTime) * 10) / 10;

            const nextBaseline = this.baselines[item.requestType] || 15;
            const entry = { finishTime: earliest.finishTime + nextBaseline, idx: earliest.idx };
            const insertIdx = heap.findIndex(h => h.finishTime > entry.finishTime);
            if (insertIdx === -1) {
                heap.push(entry);
            } else {
                heap.splice(insertIdx, 0, entry);
            }
        }
    }

    /**
     * 获取指定排队请求在仿真时间 t 的状态
     * @param {string} requestId
     * @param {number} t
     * @returns {{ estimated_wait_s: number, position: number, queue_length: number } | null}
     */
    getQueueStatus(requestId, t) {
        this.recalc(t);
        const item = this.queue.find(q => q.id === requestId);
        if (!item) return null;
        return {
            estimated_wait_s: item.estimatedWaitS,
            position: item.position,
            queue_length: this.queue.length,
        };
    }

    /**
     * 生成模拟 /status API 的 running_tasks（供 Chat ETA 测试用）
     * @param {number} t - 仿真时间
     * @returns {{ running_tasks: Array<{estimated_remaining_s: number}>, queue_length: number, idle_workers: number, total_workers: number }}
     */
    getStatusResponse(t) {
        const runningTasks = [];
        let idleCount = 0;
        for (const w of this.workers.values()) {
            if (w.busy && w.taskStartedAt !== null && w.requestType) {
                const elapsed = t - w.taskStartedAt;
                const eta = this.baselines[w.requestType] || 15;
                const remaining = elapsed < eta ? eta - elapsed : this.floor;
                runningTasks.push({ estimated_remaining_s: remaining });
            } else if (!w.busy) {
                idleCount++;
            }
        }
        return {
            running_tasks: runningTasks,
            queue_length: this.queue.length,
            idle_workers: idleCount,
            total_workers: this.workers.size,
        };
    }
}
