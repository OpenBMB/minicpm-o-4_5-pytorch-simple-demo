"""Session 清理任务

定期清理过期或超容量的 session 数据。按 meta.json 的 created_at 判断过期，
超容量时按时间 LRU（最久未创建）删除。

使用方式：
    - Gateway 启动时注册为 BackgroundTask（每天执行一次）
    - 也可手动调用：
        python session_cleanup.py --data-dir data --retention-days 30 --max-gb 50
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

logger = logging.getLogger(__name__)


def get_session_info(session_dir: str) -> Tuple[str, datetime, int]:
    """获取 session 的基本信息

    Args:
        session_dir: session 目录绝对路径

    Returns:
        (session_id, created_at, size_bytes)

    Raises:
        ValueError: meta.json 无效或不存在
    """
    meta_path = os.path.join(session_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise ValueError(f"No meta.json in {session_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    session_id = meta.get("session_id", os.path.basename(session_dir))
    created_str = meta.get("created_at")
    if created_str:
        created_at = datetime.fromisoformat(created_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
    else:
        stat = os.stat(meta_path)
        created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    total_size = 0
    for root, _dirs, files in os.walk(session_dir):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    return session_id, created_at, total_size


def cleanup_sessions(
    data_dir: str,
    retention_days: int = -1,
    max_storage_gb: float = -1,
) -> dict:
    """清理过期和超容量的 session

    Args:
        data_dir: 数据根目录（相对于 minicpmo45_service/）
        retention_days: 保留天数（-1 = 不按时间清理）
        max_storage_gb: 最大存储容量 GB（-1 = 不按容量清理）

    Returns:
        清理报告 dict
    """
    if retention_days < 0 and max_storage_gb < 0:
        return {"status": "skipped", "message": "Cleanup disabled (retention_days=-1, max_storage_gb=-1)", "deleted": 0}

    base_dir = os.path.dirname(__file__)
    sessions_dir = os.path.join(base_dir, data_dir, "sessions")

    if not os.path.isdir(sessions_dir):
        return {"status": "ok", "message": "No sessions directory", "deleted": 0}

    now = datetime.now(timezone.utc)
    enable_expiry = retention_days > 0
    enable_capacity = max_storage_gb > 0
    cutoff = now - timedelta(days=retention_days) if enable_expiry else now
    max_bytes = max_storage_gb * 1024 * 1024 * 1024 if enable_capacity else float("inf")

    sessions: List[Tuple[str, str, datetime, int]] = []
    errors: List[str] = []

    for entry in os.listdir(sessions_dir):
        sdir = os.path.join(sessions_dir, entry)
        if not os.path.isdir(sdir):
            continue
        try:
            sid, created_at, size = get_session_info(sdir)
            sessions.append((sdir, sid, created_at, size))
        except Exception as e:
            errors.append(f"{entry}: {e}")

    sessions.sort(key=lambda x: x[2])

    deleted_expired: List[str] = []
    deleted_capacity: List[str] = []
    remaining: List[Tuple[str, str, datetime, int]] = []

    for sdir, sid, created_at, size in sessions:
        if enable_expiry and created_at < cutoff:
            try:
                shutil.rmtree(sdir)
                deleted_expired.append(sid)
                logger.info(f"[Cleanup] Deleted expired session: {sid} (created {created_at.isoformat()})")
            except Exception as e:
                errors.append(f"Failed to delete {sid}: {e}")
        else:
            remaining.append((sdir, sid, created_at, size))

    total_size = sum(s[3] for s in remaining)

    while enable_capacity and total_size > max_bytes and remaining:
        sdir, sid, created_at, size = remaining.pop(0)
        try:
            shutil.rmtree(sdir)
            deleted_capacity.append(sid)
            total_size -= size
            logger.info(f"[Cleanup] Deleted for capacity: {sid} ({size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            errors.append(f"Failed to delete {sid}: {e}")

    total_deleted = len(deleted_expired) + len(deleted_capacity)
    report = {
        "status": "ok",
        "scanned": len(sessions),
        "deleted": total_deleted,
        "deleted_expired": len(deleted_expired),
        "deleted_capacity": len(deleted_capacity),
        "remaining": len(remaining),
        "remaining_size_mb": round(total_size / 1024 / 1024, 1),
        "errors": errors,
    }
    logger.info(
        f"[Cleanup] Done: scanned={report['scanned']}, "
        f"deleted={total_deleted} (expired={report['deleted_expired']}, capacity={report['deleted_capacity']}), "
        f"remaining={report['remaining']} ({report['remaining_size_mb']} MB)"
    )
    return report


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Session cleanup")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--retention-days", type=int, default=30)
    parser.add_argument("--max-gb", type=float, default=50.0)
    args = parser.parse_args()

    result = cleanup_sessions(args.data_dir, args.retention_days, args.max_gb)
    print(json.dumps(result, indent=2))
