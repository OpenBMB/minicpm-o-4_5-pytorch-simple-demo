"""队列引擎压力测试

测试高并发场景下队列的正确性和性能。
无需 GPU，纯逻辑测试。

覆盖：
- 高并发入队（100+ 并发）
- 快速释放风暴
- 混合负载（Chat + Streaming + Duplex）
- 取消风暴
- 满队保护
- 高并发入队 + 释放交错
"""

import asyncio
import random
import pytest
from datetime import datetime, timedelta

from gateway_modules.models import GatewayWorkerStatus, EtaConfig
from gateway_modules.worker_pool import WorkerPool, WorkerConnection


def make_pool(
    num_workers: int = 3,
    max_queue_size: int = 1000,
    all_busy: bool = False,
) -> WorkerPool:
    """创建测试用 WorkerPool"""
    addresses = [f"localhost:{22400 + i}" for i in range(num_workers)]
    pool = WorkerPool(
        worker_addresses=addresses,
        max_queue_size=max_queue_size,
        eta_config=EtaConfig(eta_chat_s=5.0, eta_streaming_s=10.0, eta_duplex_s=60.0),
    )
    for w in pool.workers.values():
        if all_busy:
            w.mark_busy(GatewayWorkerStatus.DUPLEX_ACTIVE, "duplex")
        else:
            w.status = GatewayWorkerStatus.IDLE
    return pool


# ============ 1. 高并发入队 ============

class TestHighConcurrencyEnqueue:
    """100+ 并发入队"""

    @pytest.mark.asyncio
    async def test_100_concurrent_enqueue(self) -> None:
        """100 个协程同时入队，所有位置唯一"""
        pool = make_pool(num_workers=3, max_queue_size=200, all_busy=True)

        positions: list = []

        async def enqueue_one() -> int:
            t, _ = pool.enqueue("chat")
            return t.position

        tasks = [enqueue_one() for _ in range(100)]
        results = await asyncio.gather(*tasks)
        positions = list(results)

        # 位置唯一
        assert len(set(positions)) == 100
        assert pool.queue_length == 100

        # 位置范围正确（1 到 100）
        assert min(positions) == 1
        assert max(positions) == 100

    @pytest.mark.asyncio
    async def test_200_enqueue_with_3_workers(self) -> None:
        """200 入队，3 Worker，最终全部完成"""
        pool = make_pool(num_workers=3, max_queue_size=300, all_busy=True)

        completed: list = []

        async def enqueue_and_wait(idx: int) -> None:
            t, f = pool.enqueue("chat")
            worker = await f
            if worker:
                completed.append(idx)
                # 模拟处理
                await asyncio.sleep(random.uniform(0.001, 0.01))
                worker.mark_busy(GatewayWorkerStatus.BUSY_CHAT, "chat")
                pool.release_worker(worker, "chat", 0.01)

        # 先释放 Worker 让调度启动
        async def release_workers() -> None:
            await asyncio.sleep(0.01)
            for w in pool.workers.values():
                pool.release_worker(w, "duplex", 1.0)

        tasks = [enqueue_and_wait(i) for i in range(200)]
        tasks.append(release_workers())

        await asyncio.gather(*tasks)

        assert len(completed) == 200
        assert pool.queue_length == 0


# ============ 2. 快速释放风暴 ============

class TestRapidRelease:
    """多个 Worker 快速连续释放"""

    @pytest.mark.asyncio
    async def test_rapid_release_50_requests(self) -> None:
        """50 个请求排队，3 个 Worker 快速轮转释放"""
        pool = make_pool(num_workers=3, all_busy=True)

        futures: list = []
        for i in range(50):
            _, f = pool.enqueue("chat")
            futures.append(f)

        # 连续释放 Worker，模拟快速处理
        workers = list(pool.workers.values())
        for round_num in range(20):  # 足够多轮
            for w in workers:
                if w.is_busy:
                    pool.release_worker(w, "chat", 0.1)
                    # 下一轮立即标记忙
                    if pool.queue_length > 0:
                        w.mark_busy(GatewayWorkerStatus.BUSY_CHAT, "chat")

        # 释放所有剩余
        for w in workers:
            if w.is_busy:
                pool.release_worker(w, "chat", 0.1)

        # 检查所有 future 都完成
        done_count = sum(1 for f in futures if f.done())
        assert done_count == 50
        assert pool.queue_length == 0


# ============ 3. 混合负载 ============

class TestMixedLoad:
    """Chat + Streaming + Duplex 混合请求"""

    @pytest.mark.asyncio
    async def test_mixed_types_fifo(self) -> None:
        """混合类型遵守 FIFO"""
        pool = make_pool(num_workers=1, all_busy=True)

        types_assigned: list = []
        request_types = ["chat", "streaming", "duplex", "chat", "streaming"]
        futures = []

        for rt in request_types:
            _, f = pool.enqueue(rt, history_hash="hash_x" if rt == "streaming" else None)
            futures.append((rt, f))

        # 依次释放，验证顺序
        w = list(pool.workers.values())[0]
        for rt, f in futures:
            pool.release_worker(w, "duplex", 1.0)
            assert f.done()
            types_assigned.append(rt)
            w.mark_busy(GatewayWorkerStatus.BUSY_CHAT, rt)

        assert types_assigned == request_types

    @pytest.mark.asyncio
    async def test_mixed_50_each(self) -> None:
        """50 Chat + 50 Streaming + 50 Duplex 混合"""
        pool = make_pool(num_workers=5, max_queue_size=200, all_busy=True)

        completed_types: dict = {"chat": 0, "streaming": 0, "duplex": 0}

        async def enqueue_and_process(rtype: str) -> None:
            t, f = pool.enqueue(rtype, history_hash="h" if rtype == "streaming" else None)
            worker = await f
            if worker:
                await asyncio.sleep(random.uniform(0.001, 0.005))
                worker.mark_busy(GatewayWorkerStatus.BUSY_CHAT, rtype)
                pool.release_worker(worker, rtype, 0.01)
                completed_types[rtype] += 1

        # 释放 Worker
        async def release_workers() -> None:
            await asyncio.sleep(0.01)
            for w in pool.workers.values():
                pool.release_worker(w, "duplex", 1.0)

        tasks = []
        for _ in range(50):
            tasks.append(enqueue_and_process("chat"))
            tasks.append(enqueue_and_process("streaming"))
            tasks.append(enqueue_and_process("duplex"))
        tasks.append(release_workers())

        await asyncio.gather(*tasks)

        assert completed_types["chat"] == 50
        assert completed_types["streaming"] == 50
        assert completed_types["duplex"] == 50


# ============ 4. 取消风暴 ============

class TestCancelStorm:
    """大量随机取消"""

    @pytest.mark.asyncio
    async def test_cancel_half(self) -> None:
        """入队 100 个，取消 50 个，剩余 50 个正确分配"""
        pool = make_pool(num_workers=1, all_busy=True)

        tickets: list = []
        futures: list = []
        for i in range(100):
            t, f = pool.enqueue("chat")
            tickets.append(t)
            futures.append(f)

        # 取消偶数位
        for i in range(0, 100, 2):
            pool.cancel(tickets[i].ticket_id)

        assert pool.queue_length == 50

        # 释放 Worker，处理剩余 50 个
        w = list(pool.workers.values())[0]
        completed = 0
        for _round in range(60):  # 足够多轮
            pool.release_worker(w, "chat", 0.1)
            for f in futures:
                if f.done() and not f.cancelled():
                    pass
            if pool.queue_length == 0:
                break
            w.mark_busy(GatewayWorkerStatus.BUSY_CHAT, "chat")

        # 统计完成数
        done_count = sum(1 for f in futures if f.done() and not f.cancelled())
        assert done_count == 50

    @pytest.mark.asyncio
    async def test_cancel_all(self) -> None:
        """入队后全部取消"""
        pool = make_pool(num_workers=1, all_busy=True)

        tickets = []
        for i in range(50):
            t, _ = pool.enqueue("chat")
            tickets.append(t)

        for t in tickets:
            pool.cancel(t.ticket_id)

        assert pool.queue_length == 0


# ============ 5. 满队保护 ============

class TestFullQueueProtection:
    """队列满后的行为"""

    @pytest.mark.asyncio
    async def test_reject_at_capacity(self) -> None:
        """精确到容量边界"""
        pool = make_pool(num_workers=1, max_queue_size=10, all_busy=True)

        for i in range(10):
            pool.enqueue("chat")

        assert pool.queue_length == 10
        assert pool.queue_full

        with pytest.raises(WorkerPool.QueueFullError):
            pool.enqueue("chat")

        # 取消一个后可以入队
        first_ticket_id = list(pool._queue.keys())[0]
        pool.cancel(first_ticket_id)
        assert pool.queue_length == 9
        assert not pool.queue_full

        pool.enqueue("chat")  # 应该成功
        assert pool.queue_length == 10

    @pytest.mark.asyncio
    async def test_full_queue_with_mixed_types(self) -> None:
        """混合类型也受容量限制"""
        pool = make_pool(num_workers=1, max_queue_size=5, all_busy=True)

        pool.enqueue("chat")
        pool.enqueue("streaming", history_hash="h1")
        pool.enqueue("duplex")
        pool.enqueue("chat")
        pool.enqueue("streaming", history_hash="h2")

        with pytest.raises(WorkerPool.QueueFullError):
            pool.enqueue("duplex")


# ============ 6. 高并发入队 + 释放交错 ============

class TestEnqueueReleaseInterleave:
    """入队和释放同时高频发生"""

    @pytest.mark.asyncio
    async def test_interleave_100(self) -> None:
        """100 入队 + 频繁释放同时进行"""
        pool = make_pool(num_workers=5, max_queue_size=200, all_busy=True)

        total_completed = 0

        async def producer(count: int) -> None:
            nonlocal total_completed
            for _ in range(count):
                t, f = pool.enqueue("chat")
                worker = await f
                if worker:
                    total_completed += 1
                    await asyncio.sleep(0.001)
                    worker.mark_busy(GatewayWorkerStatus.BUSY_CHAT, "chat")
                    pool.release_worker(worker, "chat", 0.001)

        async def initial_release() -> None:
            await asyncio.sleep(0.005)
            for w in pool.workers.values():
                pool.release_worker(w, "duplex", 1.0)
                await asyncio.sleep(0.001)

        await asyncio.gather(
            producer(100),
            initial_release(),
        )

        assert total_completed == 100
        assert pool.queue_length == 0

    @pytest.mark.asyncio
    async def test_positions_always_valid(self) -> None:
        """任何时刻，队列中的位置都是连续且唯一的"""
        pool = make_pool(num_workers=2, max_queue_size=50, all_busy=True)

        for i in range(20):
            pool.enqueue("chat")

        # 检查初始位置
        positions = [e.ticket.position for e in pool._queue.values()]
        assert positions == list(range(1, 21))

        # 取消一些
        to_cancel = list(pool._queue.keys())[::3]  # 每3个取消1个
        for tid in to_cancel:
            pool.cancel(tid)

        # 检查剩余位置连续
        remaining = [e.ticket.position for e in pool._queue.values()]
        assert remaining == list(range(1, len(remaining) + 1))

        # 释放 Worker
        for w in pool.workers.values():
            pool.release_worker(w, "duplex", 1.0)

        # 再次检查位置连续
        remaining2 = [e.ticket.position for e in pool._queue.values()]
        if remaining2:
            assert remaining2 == list(range(1, len(remaining2) + 1))
