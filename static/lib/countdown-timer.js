/**
 * CountdownTimer — 排队倒计时纯状态机（zero DOM dependency）
 *
 * 职责：管理 remaining / position / queueLength 状态 + 1s tick 递减。
 * 通过 onRender 回调将状态推送给 UI 层，自身不操作任何 DOM。
 * 支持 Vitest fake timers 精确测试。
 */

export class CountdownTimer {
    /**
     * @param {(state: {remaining: number, position: number|null, queueLength: number|null}) => void} onRender
     *   每次状态变化时调用的渲染回调
     */
    constructor(onRender) {
        this.onRender = onRender;
        this.remaining = 0;
        this.position = null;
        this.queueLength = null;
        this._intervalId = null;
    }

    /**
     * 后端推送/轮询返回新数据时调用。
     * 首次调用设定初始值并启动 tick；后续调用仅允许向下修正（单调递减），
     * 除非 position 发生变化（队列推进），此时重置为新值。
     * remaining 可为负数，表示超时等待时间。
     */
    update(estimatedWaitS, position, queueLength) {
        const newRemaining = Math.round(estimatedWaitS);
        const positionChanged = position !== null && position !== this.position;

        if (this._intervalId === null) {
            this.remaining = newRemaining;
            this._intervalId = setInterval(() => this.tick(), 1000);
        } else if (positionChanged) {
            this.remaining = newRemaining;
        } else {
            this.remaining = Math.min(this.remaining, newRemaining);
        }

        this.position = position;
        this.queueLength = queueLength;
        this._render();
    }

    /** 停止倒计时，清理 interval */
    stop() {
        if (this._intervalId !== null) {
            clearInterval(this._intervalId);
            this._intervalId = null;
        }
    }

    /** 单次 tick（public 用于测试注入，内部由 setInterval 驱动） */
    tick() {
        this.remaining = this.remaining - 1;
        this._render();
    }

    /** 是否正在运行 */
    get active() {
        return this._intervalId !== null;
    }

    /** @private */
    _render() {
        this.onRender({
            remaining: this.remaining,
            position: this.position,
            queueLength: this.queueLength,
        });
    }
}
