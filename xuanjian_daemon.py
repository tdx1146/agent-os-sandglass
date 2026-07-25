"""
玄鉴守护进程
==============

作为后台守护线程运行，定时执行四维检测，自动发送信号到丰碑网络。

职责：
  1. 定时执行四维检测（造假/偏离/矛盾/事件风暴）
  2. 自动获取目的树偏离检测结果
  3. 自动统计事件总线事件计数（事件风暴检测）
  4. 检测到 FAIL/SUSPECT 时自动发送信号到丰碑
  5. 记录巡检日志

设计原则：
  - 守护线程（daemon=True），主进程退出时自动终止
  - 线程安全，可启停
  - 单次巡检异常不影响后续巡检
  - 信号发送失败只记录日志，不重试
  - 沙漏不可用时检测降级为 SUSPECT，不崩溃

用法：
    from xuanjian_daemon import XuanjianDaemon

    # 启动守护线程
    daemon = XuanjianDaemon(
        interval=300,              # 巡检间隔（秒）
        monument_id="agent_001",   # 目标丰碑 ID
    )
    daemon.start()

    # 运行中...
    # daemon.stop()  # 停止

命令行启动：
    python xuanjian_daemon.py
    python xuanjian_daemon.py --interval 60 --monument-id agent_001
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# 确保能导入同目录模块
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


class XuanjianDaemon:
    """玄鉴守护进程。

    定时执行四维检测，自动发送信号到丰碑网络。

    Attributes:
        interval: 巡检间隔（秒），默认 300（5 分钟）
        monument_id: 目标丰碑 ID
        event_bus_path: 事件总线日志路径（用于事件风暴检测）
        send_signals: 是否自动发送信号到丰碑（False=仅检测不发送）
    """

    def __init__(self,
                 interval: float = 300.0,
                 monument_id: str = "",
                 event_bus_path: str = None,
                 send_signals: bool = True,
                 monument_url: str = None,
                 api_key: str = None):
        self.interval = interval
        self.monument_id = monument_id or os.environ.get("XUANJIAN_MONUMENT_ID", "")
        self.event_bus_path = event_bus_path
        self.send_signals = send_signals
        self.monument_url = monument_url
        self.api_key = api_key

        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # 巡检统计
        self._inspection_count = 0
        self._last_result = None
        self._last_inspection_time = None

        # 延迟初始化检测器和发送器（避免 import 时副作用）
        self._detector = None
        self._sender = None

    def _get_detector(self):
        """延迟初始化检测器。"""
        if self._detector is None:
            from xuanjian_quantify import get_detector
            self._detector = get_detector()
        return self._detector

    def _get_sender(self):
        """延迟初始化信号发送器。"""
        if self._sender is None:
            from signal_sender import SignalSender
            url = self.monument_url or os.environ.get(
                "MONUMENT_API_URL", "http://127.0.0.1:5000"
            )
            self._sender = SignalSender(monument_url=url, api_key=self.api_key)
        return self._sender

    # ═══════════════════════════════════════════════
    # 事件计数（事件风暴检测）
    # ═══════════════════════════════════════════════

    def _find_event_bus(self) -> str:
        """查找事件总线日志文件路径。

        优先级：
        1. 构造函数指定的 event_bus_path
        2. 环境变量 EVENT_BUS_PATH
        3. 丰碑网络的 data/event_bus.jsonl
        4. 沙漏的 ~/.neurobase/ 下的日志（回退）
        """
        if self.event_bus_path and os.path.exists(self.event_bus_path):
            return self.event_bus_path

        env_path = os.environ.get("EVENT_BUS_PATH", "")
        if env_path and os.path.exists(env_path):
            return env_path

        # 丰碑网络路径
        monument_paths = [
            os.path.join(os.environ.get("MONUMENT_BASE_DIR", ""), "data", "event_bus.jsonl"),
            "D:\\丰碑网络\\data\\event_bus.jsonl",
        ]
        for p in monument_paths:
            if p and os.path.exists(p):
                return p

        return ""

    def _count_recent_events(self, window_seconds: int = 60) -> int:
        """统计最近时间窗口内的事件数。

        读取事件总线日志，统计最近 window_seconds 秒内的事件数。
        文件不存在或读取失败时返回 0（安全降级）。
        """
        bus_path = self._find_event_bus()
        if not bus_path:
            return 0

        try:
            # E05 修复：从文件尾部按块读取，避免全量加载到内存
            count = 0
            now = datetime.now(timezone.utc)
            max_scan_lines = 5000  # 最多扫描5000行

            with open(bus_path, "rb") as f:
                f.seek(0, 2)
                file_size = f.tell()
                chunk_size = 8192
                position = file_size
                remaining = b""
                scanned = 0
                stop = False

                while position > 0 and not stop and scanned < max_scan_lines:
                    read_size = min(chunk_size, position)
                    position -= read_size
                    f.seek(position)
                    chunk = f.read(read_size)
                    data = chunk + remaining
                    lines = data.split(b"\n")
                    remaining = lines[0]

                    for line in reversed(lines[1:]):
                        scanned += 1
                        if scanned > max_scan_lines:
                            stop = True
                            break
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            event = json.loads(line_str.decode("utf-8"))
                            ts_str = event.get("t", "")
                            if ts_str:
                                try:
                                    event_time = datetime.fromisoformat(
                                        ts_str.replace("Z", "+00:00")
                                    )
                                except (ValueError, TypeError):
                                    count += 1
                                    continue
                                if (now - event_time).total_seconds() <= window_seconds:
                                    count += 1
                                else:
                                    stop = True
                                    break
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue

                # E05 修复：处理循环结束后的剩余首行（文件第一行）
                if remaining.strip() and not stop and scanned < max_scan_lines:
                    try:
                        event = json.loads(remaining.strip().decode("utf-8"))
                        ts_str = event.get("t", "")
                        if ts_str:
                            try:
                                event_time = datetime.fromisoformat(
                                    ts_str.replace("Z", "+00:00")
                                )
                                if (now - event_time).total_seconds() <= window_seconds:
                                    count += 1
                            except (ValueError, TypeError):
                                count += 1
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

            return count

        except Exception as e:
            logger.warning("事件计数失败: %s", e)
            return 0

    # ═══════════════════════════════════════════════
    # 巡检
    # ═══════════════════════════════════════════════

    def run_inspection(self) -> dict:
        """执行一次完整巡检。

        流程：
        1. 获取目的树偏离检测结果
        2. 统计事件计数
        3. 执行四维综合检测
        4. 发送信号到丰碑（FAIL/SUSPECT）
        5. 记录巡检日志

        Returns:
            巡检结果摘要
        """
        from purpose_tree_tracker import get_cached_purpose_check

        # E08 修复：先递增计数，确保异常时也能反映巡检尝试
        with self._lock:
            self._inspection_count += 1
            self._last_inspection_time = datetime.now(timezone.utc)

        # 1. 目的树偏离检测
        purpose_check = get_cached_purpose_check()

        # 2. 事件计数
        event_count = self._count_recent_events(window_seconds=60)

        # 3. 综合检测
        detector = self._get_detector()
        result = detector.detect_all(
            claim="守护进程定时巡检",
            evidence={},
            purpose_check=purpose_check,
            event_count=event_count,
        )

        # 4. 发送信号
        send_result = None
        if self.send_signals and self.monument_id:
            if result.is_fail() or result.is_suspect():
                sender = self._get_sender()
                send_result = sender.send(result, monument_id=self.monument_id)

        # 5. 更新结果
        with self._lock:
            self._last_result = result

        # 记录日志
        logger.info(
            "巡检 #%d: signal=%s detector=%s events=%d aligned=%s sent=%s",
            self._inspection_count,
            result.signal_type,
            result.detector,
            event_count,
            purpose_check.get("aligned", True),
            "yes" if (send_result and send_result.get("success")) else "no",
        )

        return {
            "inspection_id": self._inspection_count,
            "signal_type": result.signal_type,
            "detector": result.detector,
            "evidence_summary": result.evidence_summary,
            "event_count": event_count,
            "purpose_aligned": purpose_check.get("aligned", True),
            "send_result": send_result,
            "timestamp": self._last_inspection_time.isoformat() if self._last_inspection_time else "",
        }

    # ═══════════════════════════════════════════════
    # 守护循环
    # ═══════════════════════════════════════════════

    def _loop(self):
        """守护循环。"""
        logger.info(
            "玄鉴守护进程启动: interval=%ss monument=%s",
            self.interval,
            self.monument_id or "(未设置)",
        )

        while not self._stop_event.is_set():
            try:
                self.run_inspection()
            except Exception as e:
                logger.error("巡检异常: %s", e, exc_info=True)

            self._stop_event.wait(self.interval)

        logger.info("玄鉴守护进程已停止 (共巡检 %d 次)", self._inspection_count)

    def start(self):
        """启动守护线程。"""
        if self._thread and self._thread.is_alive():
            logger.warning("守护进程已在运行")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="xuanjian-daemon",
        )
        self._thread.start()
        logger.info("守护线程已启动")

    def stop(self, timeout: float = 10.0):
        """停止守护线程。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            logger.info("守护线程已停止")

    # ═══════════════════════════════════════════════
    # 状态查询
    # ═══════════════════════════════════════════════

    def is_running(self) -> bool:
        """守护线程是否在运行。"""
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        """获取守护进程状态。"""
        return {
            "running": self.is_running(),
            "interval": self.interval,
            "monument_id": self.monument_id,
            "inspection_count": self._inspection_count,
            "last_signal": self._last_result.signal_type if self._last_result else None,
            "last_detector": self._last_result.detector if self._last_result else None,
            "last_inspection": self._last_inspection_time.isoformat() if self._last_inspection_time else None,
            "send_signals": self.send_signals,
        }


# ═══════════════════════════════════════════════
# 全局单例（线程安全）
# ═══════════════════════════════════════════════

_default_daemon = None
_daemon_lock = threading.Lock()


def get_daemon(**kwargs) -> XuanjianDaemon:
    """获取全局 XuanjianDaemon 单例（线程安全）。"""
    global _default_daemon
    if _default_daemon is None:
        with _daemon_lock:
            if _default_daemon is None:
                _default_daemon = XuanjianDaemon(**kwargs)
    return _default_daemon


def start_daemon(**kwargs):
    """启动守护进程的便捷函数。"""
    daemon = get_daemon(**kwargs)
    daemon.start()
    return daemon


def stop_daemon():
    """停止守护进程的便捷函数。"""
    global _default_daemon
    if _default_daemon is not None:
        _default_daemon.stop()
        _default_daemon = None


# ═══════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════

def main():
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="玄鉴守护进程")
    parser.add_argument("--interval", type=float, default=300,
                        help="巡检间隔（秒），默认 300")
    parser.add_argument("--monument-id", default="",
                        help="目标丰碑 ID")
    parser.add_argument("--no-send", action="store_true",
                        help="仅检测不发送信号")
    parser.add_argument("--monument-url", default=None,
                        help="丰碑网络 URL")
    parser.add_argument("--api-key", default=None,
                        help="API Key")
    parser.add_argument("--once", action="store_true",
                        help="执行一次巡检后退出（测试用）")
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    daemon = XuanjianDaemon(
        interval=args.interval,
        monument_id=args.monument_id,
        send_signals=not args.no_send,
        monument_url=args.monument_url,
        api_key=args.api_key,
    )

    if args.once:
        # 单次巡检模式
        result = daemon.run_inspection()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 守护模式
    daemon.start()

    try:
        while daemon.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止...")
        daemon.stop()
        print(f"守护进程已停止，共巡检 {daemon._inspection_count} 次")


if __name__ == "__main__":
    main()
