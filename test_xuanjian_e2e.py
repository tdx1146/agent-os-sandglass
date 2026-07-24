"""
玄鉴→丰碑 端到端测试
======================

测试完整信号通路：
  1. 沙漏查询接口（sandglass_query.py）
  2. 玄鉴四维量化检测（xuanjian_quantify.py）
  3. 信号发送（signal_sender.py）
  4. 丰碑信号接收（signal_routes.py）

测试场景：
  A. 造假检测 → FAIL → 丰碑磨灭
  B. 偏离检测 → SUSPECT → 丰碑记录
  C. 矛盾检测 → PASS → 丰碑加固
  D. 综合检测 → 最严重结果
  E. 四维量化数据完整性

运行方式：
  cd D:\\agent-os-sandglass
  python -m pytest test_xuanjian_e2e.py -v
  或
  python test_xuanjian_e2e.py
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# 确保能导入模块
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


class TestSandglassQuery(unittest.TestCase):
    """测试沙漏查询接口。"""

    def test_query_without_sandglass(self):
        """沙漏不可用时返回安全默认值。

        H-2 修复：忠实度默认值从 1.0 改为 -1.0（标记不可用，而非默认满分）。
        """
        from sandglass_query import SandglassQuery

        sq = SandglassQuery(base_dir="/nonexistent/path")
        self.assertFalse(sq.available)

        faith = sq.query_faithfulness()
        self.assertEqual(faith["faithfulness_score"], -1.0)

        offset = sq.query_offset_entropy()
        self.assertEqual(offset["entropy"], 0.0)

        tags = sq.query_motivation_tags()
        self.assertEqual(tags["tags"], [])

    def test_four_dim_structure(self):
        """四维综合查询返回完整结构。"""
        from sandglass_query import SandglassQuery

        sq = SandglassQuery(base_dir="/nonexistent/path")
        result = sq.query_four_dim()

        # 验证四维字段存在
        self.assertIn("faithfulness", result)
        self.assertIn("goal_alignment", result)
        self.assertIn("offset_entropy", result)
        self.assertIn("motivation_tags", result)

        # 验证上下文字段存在
        self.assertIn("context", result)
        self.assertIn("persona_3d", result["context"])
        self.assertIn("scene", result["context"])
        self.assertIn("contradictions", result["context"])
        self.assertIn("spo_triples", result["context"])
        self.assertIn("emotion_entropy", result["context"])

        # 验证值范围（H-2: 沙漏不可用时 faithfulness=-1.0 标记不可用）
        self.assertGreaterEqual(result["faithfulness"], -1.0)
        self.assertLessEqual(result["faithfulness"], 1.0)
        self.assertGreaterEqual(result["goal_alignment"], 0.0)
        self.assertLessEqual(result["goal_alignment"], 1.0)
        self.assertGreaterEqual(result["offset_entropy"], 0.0)
        self.assertLessEqual(result["offset_entropy"], 1.0)


class TestXuanjianDetector(unittest.TestCase):
    """测试玄鉴检测器。"""

    def setUp(self):
        from xuanjian_quantify import XuanjianDetector
        self.detector = XuanjianDetector()

    def test_fraud_detection_file_not_exist(self):
        """造假检测：文件路径越界 → FAIL。

        H-3 修复：路径安全校验先于文件存在性检查，
        /nonexistent/file.py 不在安全基目录内 → fraud_path_traversal。
        """
        result = self.detector.detect_fraud(
            claim="修改了某个文件",
            evidence={"file_path": "/nonexistent/file.py"},
        )
        self.assertEqual(result.signal_type, "FAIL")
        self.assertEqual(result.detector, "fraud_path_traversal")
        self.assertIn("文件路径越界", result.evidence_summary)

    def test_fraud_detection_no_evidence(self):
        """造假检测：无证据且沙漏不可用 → SUSPECT。

        H-2 修复：沙漏不可用时忠实度=-1.0，无法验证 → SUSPECT（不再默认 PASS）。
        """
        result = self.detector.detect_fraud(claim="测试声明", evidence={})
        self.assertEqual(result.signal_type, "SUSPECT")
        self.assertEqual(result.detector, "fraud_faithfulness")

    def test_deviation_detection_default(self):
        """偏离检测：默认 → PASS（沙漏不可用时默认满分）。"""
        result = self.detector.detect_deviation()
        self.assertEqual(result.signal_type, "PASS")

    def test_deviation_detection_purpose_misaligned(self):
        """偏离检测：目的树不对齐 → FAIL。"""
        result = self.detector.detect_deviation(
            purpose_check={"aligned": False, "deviation": "玄鉴变成了评分器"}
        )
        self.assertEqual(result.signal_type, "FAIL")
        self.assertIn("目的树检测偏离", result.evidence_summary)

    def test_contradiction_detection_default(self):
        """矛盾检测：默认 → PASS。"""
        result = self.detector.detect_contradiction()
        self.assertEqual(result.signal_type, "PASS")

    def test_event_storm_detection_normal(self):
        """事件风暴检测：正常事件流 → PASS。"""
        result = self.detector.detect_event_storm(event_count=10)
        self.assertEqual(result.signal_type, "PASS")

    def test_event_storm_detection_storm(self):
        """事件风暴检测：事件洪流 → FAIL。"""
        result = self.detector.detect_event_storm(
            event_count=250, threshold=100
        )
        self.assertEqual(result.signal_type, "FAIL")
        self.assertIn("事件风暴", result.evidence_summary)

    def test_event_storm_detection_warning(self):
        """事件风暴检测：达到阈值 → SUSPECT。"""
        result = self.detector.detect_event_storm(
            event_count=100, threshold=100
        )
        self.assertEqual(result.signal_type, "SUSPECT")

    def test_detect_all_returns_worst(self):
        """综合检测返回最严重结果。"""
        result = self.detector.detect_all(
            claim="测试",
            evidence={"file_path": "/nonexistent"},
            event_count=10,
        )
        self.assertEqual(result.signal_type, "FAIL")

    def test_four_dim_attached_to_result(self):
        """检测结果包含四维量化附件。"""
        result = self.detector.detect_fraud(claim="测试", evidence={})
        self.assertIsNotNone(result.four_dim)
        self.assertIn("faithfulness", result.four_dim)
        self.assertIn("goal_alignment", result.four_dim)
        self.assertIn("offset_entropy", result.four_dim)
        self.assertIn("motivation_tags", result.four_dim)

    def test_signal_payload_format(self):
        """信号 payload 格式正确。"""
        result = self.detector.detect_fraud(
            claim="测试",
            evidence={"file_path": "/nonexistent"},
        )
        payload = result.to_signal_payload(monument_id="test-monument-001")

        self.assertEqual(payload["signal_type"], "FAIL")
        self.assertEqual(payload["monument_id"], "test-monument-001")
        self.assertIn("detail", payload)
        self.assertIn("four_dim", payload)
        self.assertIn("detector", payload)
        self.assertIn("timestamp", payload)


class TestSignalSender(unittest.TestCase):
    """测试信号发送器。"""

    def test_sender_initialization(self):
        """信号发送器正确初始化。"""
        from signal_sender import SignalSender

        sender = SignalSender(monument_url="http://127.0.0.1:5000")
        self.assertEqual(sender._endpoint, "http://127.0.0.1:5000/signal/receive")

    @patch("urllib.request.urlopen")
    def test_send_success(self, mock_urlopen):
        """模拟发送成功。"""
        from signal_sender import SignalSender
        from xuanjian_quantify import XuanjianDetector, DetectionResult

        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = json.dumps({
            "status": "ok",
            "action": "eroded",
            "ranking_weight": 0.35,
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        detector = XuanjianDetector()
        result = detector.detect_fraud(
            claim="测试",
            evidence={"file_path": "/nonexistent"},
        )

        sender = SignalSender()
        send_result = sender.send(result, monument_id="test-001")

        self.assertTrue(send_result["success"])
        self.assertEqual(send_result["status_code"], 200)

    @patch("urllib.request.urlopen")
    def test_send_connection_error(self, mock_urlopen):
        """模拟连接失败。"""
        from signal_sender import SignalSender
        from xuanjian_quantify import DetectionResult
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        result = DetectionResult(
            signal_type="FAIL",
            detector="test",
            evidence_summary="测试",
        )

        sender = SignalSender()
        send_result = sender.send(result, monument_id="test-001")

        self.assertFalse(send_result["success"])
        self.assertIn("error", send_result["response"])


class TestEndToEndFlow(unittest.TestCase):
    """端到端流程测试。"""

    def test_full_flow_fraud_to_erode(self):
        """完整流程：造假检测 → FAIL → 信号 payload → 丰碑格式验证。"""
        from xuanjian_quantify import XuanjianDetector

        detector = XuanjianDetector()

        # 1. 检测
        result = detector.detect_fraud(
            claim="声称修改了 verify_daemon.py",
            evidence={"file_path": "/nonexistent/verify_daemon.py"},
        )

        # 2. 验证信号（H-3: 路径越界 → fraud_path_traversal）
        self.assertEqual(result.signal_type, "FAIL")
        self.assertEqual(result.detector, "fraud_path_traversal")

        # 3. 验证 payload 格式（丰碑 signal_routes.py 需要的格式）
        payload = result.to_signal_payload(monument_id="agent_001")

        self.assertEqual(payload["signal_type"], "FAIL")
        self.assertEqual(payload["monument_id"], "agent_001")
        self.assertIsInstance(payload["detail"], str)
        self.assertIsInstance(payload["four_dim"], dict)

        # 4. 验证四维量化数据
        four_dim = payload["four_dim"]
        self.assertIn("faithfulness", four_dim)
        self.assertIn("goal_alignment", four_dim)
        self.assertIn("offset_entropy", four_dim)
        self.assertIn("motivation_tags", four_dim)

    def test_full_flow_pass_to_reinforce(self):
        """完整流程：沙漏不可用时无证据 → SUSPECT → 信号 payload。

        H-2 修复：沙漏不可用时忠实度=-1.0 → SUSPECT（无法验证，降级而非放行）。
        """
        from xuanjian_quantify import XuanjianDetector

        detector = XuanjianDetector()
        result = detector.detect_fraud(claim="正常声明", evidence={})

        self.assertEqual(result.signal_type, "SUSPECT")

        payload = result.to_signal_payload(monument_id="agent_002")
        self.assertEqual(payload["signal_type"], "SUSPECT")

    def test_full_flow_suspect_to_log(self):
        """完整流程：疑似 → SUSPECT → 信号 payload。"""
        from xuanjian_quantify import XuanjianDetector

        detector = XuanjianDetector()
        result = detector.detect_event_storm(event_count=100, threshold=100)

        self.assertEqual(result.signal_type, "SUSPECT")

        payload = result.to_signal_payload(monument_id="agent_003")
        self.assertEqual(payload["signal_type"], "SUSPECT")

    def test_quantification_constraint(self):
        """约束验证：四维量化是附件，不是评分。

        确保检测结果只有三态（PASS/FAIL/SUSPECT），
        四维量化数据只出现在 four_dim 附件中。
        """
        from xuanjian_quantify import XuanjianDetector, DetectionResult

        detector = XuanjianDetector()

        # 所有检测方法的返回值必须是三态之一
        valid_signals = {DetectionResult.PASS, DetectionResult.FAIL, DetectionResult.SUSPECT}

        results = [
            detector.detect_fraud(claim="test", evidence={}),
            detector.detect_deviation(),
            detector.detect_contradiction(),
            detector.detect_event_storm(event_count=10),
            detector.detect_all(claim="test", evidence={}),
        ]

        for r in results:
            self.assertIn(r.signal_type, valid_signals)
            # 四维量化只在 four_dim 字段中，不在 signal_type 中
            self.assertIsInstance(r.four_dim, dict)
            # signal_type 不是数字分数
            self.assertNotEqual(r.signal_type, r.four_dim.get("faithfulness"))



class TestSPOConfidenceFilter(unittest.TestCase):
    """测试 SPO 三元组 confidence 过滤。

    约束：confidence 字段仅用于过滤低质量数据（<0.3 不参与检测），
    不用于排名或评分。
    """

    def setUp(self):
        from sandglass_query import SandglassQuery
        self.sq = SandglassQuery(base_dir="/nonexistent/path")
        self.sq._available = True  # 强制标记为可用以测试过滤逻辑

    def _make_mock_weavethread(self, triples):
        """创建模拟的 weavethread 模块。"""
        mock_module = MagicMock()
        mock_module.wthread_query = MagicMock(return_value=triples)
        mock_module.wthread_stats = MagicMock(return_value={
            "total_triples": len(triples),
            "relations": [("使用", len(triples))],
        })
        return mock_module

    def test_high_confidence_kept(self):
        """confidence >= 0.3 的三元组全部保留。"""
        triples = [
            {"subject": "user", "relation": "使用", "object": "Python", "confidence": 0.9},
            {"subject": "user", "relation": "使用", "object": "Git", "confidence": 0.5},
            {"subject": "user", "relation": "使用", "object": "Docker", "confidence": 0.3},
        ]
        mock_module = self._make_mock_weavethread(triples)
        with patch.dict("sys.modules", {"weavethread": mock_module}):
            result = self.sq.query_spo_triples()
        self.assertEqual(result["count"], 3)
        self.assertEqual(len(result["triples"]), 3)

    def test_low_confidence_filtered(self):
        """confidence < 0.3 的三元组被过滤掉。"""
        triples = [
            {"subject": "user", "relation": "使用", "object": "Python", "confidence": 0.9},
            {"subject": "user", "relation": "使用", "object": "BadTool", "confidence": 0.1},
            {"subject": "user", "relation": "使用", "object": "WorseTool", "confidence": 0.2},
        ]
        mock_module = self._make_mock_weavethread(triples)
        with patch.dict("sys.modules", {"weavethread": mock_module}):
            result = self.sq.query_spo_triples()
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["triples"][0]["object"], "Python")

    def test_missing_confidence_defaults_to_0_5(self):
        """缺少 confidence 字段时默认 0.5，通过过滤。"""
        triples = [
            {"subject": "user", "relation": "使用", "object": "Python"},
        ]
        mock_module = self._make_mock_weavethread(triples)
        with patch.dict("sys.modules", {"weavethread": mock_module}):
            result = self.sq.query_spo_triples()
        self.assertEqual(result["count"], 1)

    def test_all_filtered_returns_empty(self):
        """所有三元组 confidence < 0.3 时返回空列表。"""
        triples = [
            {"subject": "user", "relation": "使用", "object": "A", "confidence": 0.1},
            {"subject": "user", "relation": "使用", "object": "B", "confidence": 0.2},
        ]
        mock_module = self._make_mock_weavethread(triples)
        with patch.dict("sys.modules", {"weavethread": mock_module}):
            result = self.sq.query_spo_triples()
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["triples"], [])

    def test_boundary_confidence(self):
        """边界值：0.30 保留，0.29 过滤。"""
        triples = [
            {"subject": "user", "relation": "使用", "object": "Below", "confidence": 0.29},
            {"subject": "user", "relation": "使用", "object": "AtThreshold", "confidence": 0.30},
        ]
        mock_module = self._make_mock_weavethread(triples)
        with patch.dict("sys.modules", {"weavethread": mock_module}):
            result = self.sq.query_spo_triples()
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["triples"][0]["object"], "AtThreshold")

    def test_stats_not_affected_by_filter(self):
        """stats 返回原始总数，不受 confidence 过滤影响。"""
        triples = [
            {"subject": "user", "relation": "使用", "object": "A", "confidence": 0.9},
            {"subject": "user", "relation": "使用", "object": "B", "confidence": 0.1},
        ]
        mock_module = self._make_mock_weavethread(triples)
        with patch.dict("sys.modules", {"weavethread": mock_module}):
            result = self.sq.query_spo_triples()
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["stats"]["total_triples"], 2)

    def test_confidence_not_used_for_ranking(self):
        """约束验证：confidence 仅用于过滤，不影响三元组顺序。"""
        triples = [
            {"subject": "user", "relation": "使用", "object": "Low", "confidence": 0.4},
            {"subject": "user", "relation": "使用", "object": "High", "confidence": 0.95},
            {"subject": "user", "relation": "使用", "object": "Mid", "confidence": 0.6},
        ]
        mock_module = self._make_mock_weavethread(triples)
        with patch.dict("sys.modules", {"weavethread": mock_module}):
            result = self.sq.query_spo_triples()
        # 顺序应保持原始顺序（按数据库返回），不按 confidence 排序
        objects = [t["object"] for t in result["triples"]]
        self.assertEqual(objects, ["Low", "High", "Mid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
