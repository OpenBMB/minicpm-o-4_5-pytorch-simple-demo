/**
 * Chat 模式 ETA 估算 — 从 /status 响应推断排队等待时间
 *
 * 逻辑来源：index.html checkServiceStatus() 中的 Chat 排队倒计时计算。
 * 提取为纯函数以便独立测试。
 *
 * @param {Object} statusResponse - /status API 返回值
 * @param {Array}  statusResponse.running_tasks - 正在运行的任务列表
 * @param {number} statusResponse.running_tasks[].estimated_remaining_s - 预估剩余秒数
 * @param {number} statusResponse.queue_length - 当前队列长度
 * @returns {number} 预估等待秒数（整数，≥1）
 */
export function estimateChatEta(statusResponse) {
    const runningTasks = statusResponse.running_tasks || [];
    const avgRemaining = runningTasks.length > 0
        ? runningTasks.reduce((s, t) => s + (t.estimated_remaining_s ?? 15), 0) / runningTasks.length
        : 15;
    return Math.max(1, Math.round(avgRemaining + (statusResponse.queue_length - 1) * 15));
}
