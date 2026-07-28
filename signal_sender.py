"""
玄鉴信号发送器
================

将玄鉴的检测结果（PASS/FAIL/SUSPECT）发送到丰碑网络的信号接收端点。

信号通路：
  玄鉴检测 → DetectionResult → 信号发送器 → HTTP POST → 丰碑 /signal/receive

设计原则：
  - 每个信号有且仅有一个消费者
  - 发送失败只记录日志，不重试/补偿/死信队列
  - 信号格式不对的直接丢弃
"""

import os
import json
import logging
import threading
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SignalSender:
    """信号发送器。将玄鉴检测结果发送到丰碑网络。"""

    def __init__(self, monument_url: str = "http://127.0.0.1:18891",
                 timeout: int = 5, api_key: str = None):
        self.monument_url = monument_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("MONUMENT_API_KEY", "")
        self._endpoint = f"{self.monument_url}/signal/receive"

    def send(self, result, monument_id: str = "") -> dict:
        """发送检测结果到丰碑。

        Args:
            result: DetectionResult 对象
            monument_id: 目标丰碑 ID

        Returns:
            {
                "success": bool,
                "status_code": int,
                "response": dict,
            }
        """
        payload = result.to_signal_payload(monument_id)

        try:
            import urllib.request
            import urllib.error

            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            req = urllib.request.Request(
                self._endpoint,
                data=data,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status_code = resp.getcode()
                body = resp.read().decode("utf-8")
                try:
                    response_data = json.loads(body)
                except json.JSONDecodeError:
                    response_data = {"raw": body}

                logger.info(
                    "信号发送成功: type=%s monument=%s action=%s",
                    result.signal_type,
                    monument_id,
                    response_data.get("action", ""),
                )

                return {
                    "success": True,
                    "status_code": status_code,
                    "response": response_data,
                }

        except urllib.error.URLError as e:
            logger.warning("信号发送失败（连接错误）: %s", e)
            return {
                "success": False,
                "status_code": 0,
                "response": {"error": f"连接错误: {e}"},
            }
        except Exception as e:
            logger.error("信号发送异常: %s", e)
            return {
                "success": False,
                "status_code": 0,
                "response": {"error": str(e)},
            }

    def send_fail(self, result, monument_id: str = "") -> dict:
        """便捷方法：仅当结果是 FAIL 时发送。"""
        if not result.is_fail():
            return {"success": False, "status_code": 0, "skipped": True,
                    "response": {"reason": "非FAIL信号"}}
        return self.send(result, monument_id)

    def send_pass(self, result, monument_id: str = "") -> dict:
        """便捷方法：仅当结果是 PASS 时发送。"""
        if not result.is_pass():
            return {"success": False, "status_code": 0, "skipped": True,
                    "response": {"reason": "非PASS信号"}}
        return self.send(result, monument_id)

    def send_suspect(self, result, monument_id: str = "") -> dict:
        """便捷方法：仅当结果是 SUSPECT 时发送。"""
        if not result.is_suspect():
            return {"success": False, "status_code": 0, "skipped": True,
                    "response": {"reason": "非SUSPECT信号"}}
        return self.send(result, monument_id)


# ═══════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════

_default_sender = None
_sender_lock = threading.Lock()


def get_sender(monument_url: str = None, reset: bool = False) -> SignalSender:
    """获取全局 SignalSender 单例（线程安全）。M-6 修复：使用 reset 参数控制重建。"""
    global _default_sender
    if _default_sender is None or reset:
        with _sender_lock:
            if _default_sender is None or reset:
                url = monument_url or os.environ.get("MONUMENT_API_URL", "http://127.0.0.1:18891")
                _default_sender = SignalSender(monument_url=url)
    return _default_sender


def send_signal(result, monument_id: str = "") -> dict:
    """便捷函数：发送信号。"""
    return get_sender().send(result, monument_id)
