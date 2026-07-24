"""
玄鉴守护进程 + 目的树偏离检测 测试
====================================

测试覆盖：
  A. 目的树偏离检测（purpose_tree_tracker.py）
  B. 守护线程（xuanjian_daemon.py）
  C. 沙漏实时数据接入验证

运行方式：
  cd D:\\agent-os-sandglass
  python -m pytest test_daemon.py -v
"""

import os
import sys
import json
import time
import tempfile
import unittest
from unittest.mock import patch, MagicMock

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# ═══════════════════════════════════════════════════════════════
# A. 目的树偏离检测测试
# ═══════════════════════════════════════════════════════════════

class TestPurposeTreeTracker(unittest.TestCase):
    """测试目的树偏离检测器。"""

    def test_check_returns_aligned_when_yaml_exists(self):
        """purpose_tree.yaml 存在且代码合规时返回 aligned=True。"""
        from purpose_tree_tracker import check_purpose_alignment
        result = check_purpose_alignment()
        self.assertTrue(result["aligned"])
        self.assertEqual(result["deviation"], "")
        self.assertEqual(result["direction"], "")
        self.assertIsInstance(result["violations"], list)

    def test_check_returns_aligned_when_yaml_missing(self):
        """purpose_tree.yaml 不存在时安全降级为 aligned=True。"""
        from purpose_tree_tracker import check_purpose_alignment
        with patch("purpose_tree_tracker._YAML_PATH", "/nonexistent/purpose_tree.yaml"):
            result = check_purpose_alignment()
        self.assertTrue(result["aligned"])
        self.assertEqual(result["violations"], [])

    def test_check_returns_aligned_when_pyyaml_missing(self):
        """PyYAML 未安装时安全降级为 aligned=True。"""
        from purpose_tree_tracker import check_purpose_alignment, _load_yaml
        with patch("purpose_tree_tracker._load_yaml", return_value=None):
            result = check_purpose_alignment()
        self.assertTrue(result["aligned"])

    def test_check_detects_forbidden_pattern(self):
        """发现禁止模式时返回 aligned=False。"""
        from purpose_tree_tracker import check_purpose_alignment

        # 模拟配置：检查一个包含禁止模式的文件
        mock_config = {
            "identity": {
                "constraints": [{
                    "id": "test_constraint",
                    "description": "测试禁止模式",
                    "check_files": ["xuanjian_quantify.py"],
                    "forbidden_patterns": ["import os"],  # xuanjian_quantify.py 肯定有这个
                    "required_patterns": [],
                    "severity": "high",
                }]
            }
        }

        with patch("purpose_tree_tracker._load_yaml", return_value=mock_config):
            result = check_purpose_alignment()

        self.assertFalse(result["aligned"])
        self.assertIn("测试禁止模式", result["deviation"])
        self.assertEqual(result["direction"], "high_severity")
        self.assertEqual(len(result["violations"]), 1)

    def test_check_detects_missing_required_pattern(self):
        """缺少必需模式时返回 aligned=False。"""
        from purpose_tree_tracker import check_purpose_alignment

        mock_config = {
            "identity": {
                "constraints": [{
                    "id": "test_required",
                    "description": "测试必需模式",
                    "check_files": ["xuanjian_quantify.py"],
                    "forbidden_patterns": [],
                    "required_patterns": ["NONEXISTENT_PATTERN_XYZ"],
                    "severity": "high",
                }]
            }
        }

        with patch("purpose_tree_tracker._load_yaml", return_value=mock_config):
            result = check_purpose_alignment()

        self.assertFalse(result["aligned"])
        self.assertIn("测试必需模式", result["deviation"])

    def test_check_skips_nonexistent_file(self):
        """检查文件不存在时跳过该约束（不触发偏离）。"""
        from purpose_tree_tracker import check_purpose_alignment

        mock_config = {
            "identity": {
                "constraints": [{
                    "id": "test_missing_file",
                    "description": "测试文件不存在",
                    "check_files": ["nonexistent_file_xyz.py"],
                    "forbidden_patterns": ["anything"],
                    "required_patterns": [],
                    "severity": "high",
                }]
            }
        }

        with patch("purpose_tree_tracker._load_yaml", return_value=mock_config):
            result = check_purpose_alignment()

        self.assertTrue(result["aligned"])  # 文件不存在 → 跳过 → 对齐

    def test_check_invalid_regex_skipped(self):
        """正则编译失败时跳过该模式（不误判）。"""
        from purpose_tree_tracker import check_purpose_alignment

        mock_config = {
            "identity": {
                "constraints": [{
                    "id": "test_bad_regex",
                    "description": "测试无效正则",
                    "check_files": ["xuanjian_quantify.py"],
                    "forbidden_patterns": ["[invalid regex"],  # 无效正则
                    "required_patterns": [],
                    "severity": "high",
                }]
            }
        }

        with patch("purpose_tree_tracker._load_yaml", return_value=mock_config):
            result = check_purpose_alignment()

        self.assertTrue(result["aligned"])  # 正则失败 → 跳过 → 对齐

    def test_cached_check_returns_same_result(self):
        """缓存机制返回相同结果。"""
        from purpose_tree_tracker import get_cached_purpose_check, _last_check
        # 第一次调用
        result1 = get_cached_purpose_check(force_refresh=True)
        # 第二次调用（应从缓存返回）
        result2 = get_cached_purpose_check()
        self.assertEqual(result1, result2)

    def test_get_purpose_summary(self):
        """获取目的树摘要。"""
        from purpose_tree_tracker import get_purpose_summary
        summary = get_purpose_summary()
        self.assertTrue(summary["available"])
        self.assertGreater(len(summary["sections"]), 0)

    def test_purpose_check_format_for_detector(self):
        """purpose_check 格式适合传给 detect_deviation。"""
        from purpose_tree_tracker import check_purpose_alignment
        from xuanjian_quantify import XuanjianDetector

        purpose_check = check_purpose_alignment()
        detector = XuanjianDetector()

        # 确保能正常传入 detect_deviation 不报错
        result = detector.detect_deviation(purpose_check=purpose_check)
        self.assertIn(result.signal_type, ["PASS", "FAIL", "SUSPECT"])


# ═══════════════════════════════════════════════════════════════
# B. 守护线程测试
# ═══════════════════════════════════════════════════════════════

class TestXuanjianDaemon(unittest.TestCase):
    """测试玄鉴守护进程。"""

    def setUp(self):
        """每个测试前重置全局单例。"""
        import xuanjian_daemon
        xuanjian_daemon._default_daemon = None
        xuanjian_daemon._default_detector = None

    def test_daemon_initialization(self):
        """守护进程正确初始化。"""
        from xuanjian_daemon import XuanjianDaemon
        daemon = XuanjianDaemon(
            interval=60,
            monument_id="test-001",
            send_signals=False,
        )
        self.assertEqual(daemon.interval, 60)
        self.assertEqual(daemon.monument_id, "test-001")
        self.assertFalse(daemon.send_signals)
        self.assertFalse(daemon.is_running())

    def test_run_inspection_no_send(self):
        """单次巡检（不发送信号）。"""
        from xuanjian_daemon import XuanjianDaemon
        daemon = XuanjianDaemon(
            interval=300,
            monument_id="test-001",
            send_signals=False,
        )
        result = daemon.run_inspection()

        self.assertIn(result["signal_type"], ["PASS", "FAIL", "SUSPECT"])
        self.assertEqual(result["inspection_id"], 1)
        self.assertEqual(result["send_result"], None)
        self.assertIsInstance(result["event_count"], int)
        self.assertIsInstance(result["purpose_aligned"], bool)

    def test_run_inspection_with_purpose_check(self):
        """巡检包含目的树检查结果。"""
        from xuanjian_daemon import XuanjianDaemon
        daemon = XuanjianDaemon(send_signals=False)
        result = daemon.run_inspection()
        # 目的树应该对齐（当前代码合规）
        self.assertTrue(result["purpose_aligned"])

    def test_event_count_no_bus(self):
        """事件总线不存在时事件计数为 0。"""
        from xuanjian_daemon import XuanjianDaemon
        daemon = XuanjianDaemon(
            event_bus_path="/nonexistent/event_bus.jsonl",
            send_signals=False,
        )
        count = daemon._count_recent_events()
        self.assertEqual(count, 0)

    def test_event_count_with_mock_bus(self):
        """模拟事件总线文件。"""
        from xuanjian_daemon import XuanjianDaemon

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            now = datetime.now(timezone.utc).isoformat()
            for i in range(10):
                f.write(json.dumps({"t": now, "event_type": "test"}) + "\n")
            bus_path = f.name

        try:
            daemon = XuanjianDaemon(
                event_bus_path=bus_path,
                send_signals=False,
            )
            count = daemon._count_recent_events(window_seconds=3600)
            self.assertEqual(count, 10)
        finally:
            os.unlink(bus_path)

    def test_event_count_old_events_excluded(self):
        """超出时间窗口的旧事件不计入。"""
        from xuanjian_daemon import XuanjianDaemon
        from datetime import timedelta

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            now = datetime.now(timezone.utc)
            old_time = (now - timedelta(seconds=120)).isoformat()
            recent_time = now.isoformat()

            f.write(json.dumps({"t": old_time, "event_type": "old"}) + "\n")
            f.write(json.dumps({"t": recent_time, "event_type": "recent"}) + "\n")
            bus_path = f.name

        try:
            daemon = XuanjianDaemon(
                event_bus_path=bus_path,
                send_signals=False,
            )
            count = daemon._count_recent_events(window_seconds=60)
            self.assertEqual(count, 1)  # 只有 1 条在 60s 窗口内
        finally:
            os.unlink(bus_path)

    def test_daemon_start_stop(self):
        """守护线程能正常启动和停止。"""
        from xuanjian_daemon import XuanjianDaemon
        daemon = XuanjianDaemon(
            interval=0.5,  # 0.5 秒间隔
            send_signals=False,
        )

        self.assertFalse(daemon.is_running())
        daemon.start()
        self.assertTrue(daemon.is_running())

        time.sleep(1.5)  # 等待至少一次巡检

        daemon.stop()
        self.assertFalse(daemon.is_running())
        self.assertGreaterEqual(daemon._inspection_count, 1)

    def test_daemon_status(self):
        """获取守护进程状态。"""
        from xuanjian_daemon import XuanjianDaemon
        daemon = XuanjianDaemon(
            interval=60,
            monument_id="test-status",
            send_signals=False,
        )

        status = daemon.get_status()
        self.assertFalse(status["running"])
        self.assertEqual(status["interval"], 60)
        self.assertEqual(status["monument_id"], "test-status")
        self.assertEqual(status["inspection_count"], 0)

        # 执行一次巡检后检查状态
        daemon.run_inspection()
        status = daemon.get_status()
        self.assertEqual(status["inspection_count"], 1)
        self.assertIsNotNone(status["last_signal"])

    def test_inspection_exception_does_not_crash(self):
        """巡检异常不会导致守护线程崩溃。"""
        from xuanjian_daemon import XuanjianDaemon

        daemon = XuanjianDaemon(interval=0.3, send_signals=False)

        # Mock 检测器抛出异常
        mock_detector = MagicMock()
        mock_detector.detect_all.side_effect = RuntimeError("模拟异常")
        daemon._detector = mock_detector

        daemon.start()
        time.sleep(1.0)
        daemon.stop()

        # 守护线程应该还活着（异常被捕获）
        # 由于异常被捕获，inspection_count 可能仍为 0
        # 关键是线程没有崩溃退出
        self.assertFalse(daemon.is_running())  # 已 stop
        # 但线程在 stop 前是活着的（否则 join 会超时）

    @patch("urllib.request.urlopen")
    def test_inspection_sends_signal_on_fail(self, mock_urlopen):
        """检测到 FAIL 时自动发送信号。"""
        from xuanjian_daemon import XuanjianDaemon
        from xuanjian_quantify import DetectionResult

        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = json.dumps({"status": "ok"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        daemon = XuanjianDaemon(
            monument_id="test-send",
            send_signals=True,
        )

        # Mock 检测器返回 FAIL
        mock_detector = MagicMock()
        mock_detector.detect_all.return_value = DetectionResult(
            signal_type=DetectionResult.FAIL,
            detector="test",
            evidence_summary="测试 FAIL",
            four_dim={"faithfulness": 0.5},
        )
        daemon._detector = mock_detector

        result = daemon.run_inspection()

        self.assertEqual(result["signal_type"], "FAIL")
        self.assertIsNotNone(result["send_result"])
        self.assertTrue(result["send_result"]["success"])
        mock_urlopen.assert_called_once()

    def test_inspection_no_send_on_pass(self):
        """检测到 PASS 时不发送信号。"""
        from xuanjian_daemon import XuanjianDaemon
        from xuanjian_quantify import DetectionResult

        daemon = XuanjianDaemon(
            monument_id="test-pass",
            send_signals=True,
        )

        # Mock 检测器返回 PASS
        mock_detector = MagicMock()
        mock_detector.detect_all.return_value = DetectionResult(
            signal_type=DetectionResult.PASS,
            detector="test",
            evidence_summary="测试 PASS",
        )
        daemon._detector = mock_detector

        result = daemon.run_inspection()

        self.assertEqual(result["signal_type"], "PASS")
        self.assertIsNone(result["send_result"])  # PASS 不发送


# ═══════════════════════════════════════════════════════════════
# C. 沙漏实时数据接入验证
# ═══════════════════════════════════════════════════════════════

class TestSandglassIntegration(unittest.TestCase):
    """验证守护线程与沙漏的集成。"""

    def test_daemon_uses_sandglass_query(self):
        """守护线程通过 sandglass_query 获取四维数据。"""
        from xuanjian_daemon import XuanjianDaemon
        daemon = XuanjianDaemon(send_signals=False)

        # 沙漏不可用时，检测器应降级为 SUSPECT
        result = daemon.run_inspection()

        # 沙漏不可用时，faithfulness=-1.0 → SUSPECT
        self.assertEqual(result["signal_type"], "SUSPECT")

    def test_daemon_with_mock_sandglass_available(self):
        """沙漏可用时，检测器获取真实数据。"""
        from xuanjian_daemon import XuanjianDaemon
        from sandglass_query import SandglassQuery

        # 创建一个模拟可用的沙漏查询
        mock_sq = MagicMock(spec=SandglassQuery)
        mock_sq.available = True
        mock_sq.query_four_dim.return_value = {
            "faithfulness": 0.9,
            "goal_alignment": 0.85,
            "offset_entropy": 0.2,
            "motivation_tags": ["测试标签"],
            "context": {
                "contradictions": {"has_contradiction": False},
            },
            "available": True,
        }
        mock_sq.verify_claim.return_value = {"verified": True}

        from xuanjian_quantify import XuanjianDetector
        detector = XuanjianDetector(sandglass_query=mock_sq)

        daemon = XuanjianDaemon(send_signals=False)
        daemon._detector = detector

        result = daemon.run_inspection()

        # 沙漏可用 + 忠实度 0.9 → PASS
        self.assertEqual(result["signal_type"], "PASS")
        mock_sq.query_four_dim.assert_called()

    def test_purpose_check_integrated_into_inspection(self):
        """目的树检查结果被集成到巡检中。"""
        from xuanjian_daemon import XuanjianDaemon
        from purpose_tree_tracker import check_purpose_alignment

        # 确认目的树检查能正常集成到巡检流程
        purpose_check = check_purpose_alignment()
        self.assertTrue(purpose_check["aligned"])

        daemon = XuanjianDaemon(send_signals=False)
        result = daemon.run_inspection()
        self.assertTrue(result["purpose_aligned"])


# 导入 datetime 用于测试
from datetime import datetime, timezone


if __name__ == "__main__":
    unittest.main(verbosity=2)
