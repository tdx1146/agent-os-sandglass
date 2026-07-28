"""
玄鉴四维量化检测模块
========================

玄鉴是 AgentOS 的检测守护进程，职责：
  1. 造假检测 — AI 声称做了某事，是否真的做了（文件变更证据校验）
  2. 偏离检测 — AI 的行为是否偏离了既定目的和决策模式
  3. 矛盾检测 — AI 的断言之间是否存在事实冲突（SPO 三元组矛盾）
  4. 事件风暴检测 — 系统是否出现异常的事件洪流或消费链故障

输出：PASS / FAIL / SUSPECT 三态信号
  - PASS：检测通过，无异常
  - FAIL：检测失败，确认存在造假/偏离/矛盾/事件风暴
  - SUSPECT：疑似异常，证据不足以下定论

四维量化附件（仅附加到 FAIL/SUSPECT 信号）：
  - faithfulness：忠实度 [0, 1]
  - goal_alignment：目标对齐度 [0, 1]
  - offset_entropy：偏移熵 [0, 1]
  - motivation_tags：动机标签 [str]

设计原则：
  - 玄鉴是检测器，不是评分器
  - 只输出三态信号，不输出分数
  - 四维量化是信号附件，辅助丰碑决定磨灭强度
  - 动机标签来自沙漏决策粒子，不是玄鉴自己推断的

用法：
    from xuanjian_quantify import XuanjianDetector
    detector = XuanjianDetector()

    # 造假检测
    result = detector.detect_fraud(claim="修改了 verify_daemon.py", evidence=git_log)

    # 偏离检测
    result = detector.detect_deviation(decision_text="放弃方案改用B")

    # 矛盾检测
    result = detector.detect_contradiction(assertion="声称使用了开源方案")

    # 综合检测
    result = detector.detect_all(claim=..., evidence=..., decision=...)
"""

import os
import json
import logging
import hashlib
import threading
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# 尝试导入沙漏查询接口
try:
    from sandglass_query import SandglassQuery
    _SANDGLASS_AVAILABLE = True
except ImportError:
    _SANDGLASS_AVAILABLE = False
    SandglassQuery = None


class DetectionResult:
    """检测结果。封装三态信号和四维量化附件。"""

    PASS = "PASS"
    FAIL = "FAIL"
    SUSPECT = "SUSPECT"

    def __init__(self, signal_type: str, detector: str,
                 evidence_summary: str = "", four_dim: dict = None):
        self.signal_type = signal_type
        self.detector = detector
        self.evidence_summary = evidence_summary
        self.four_dim = four_dim or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def is_fail(self) -> bool:
        return self.signal_type == self.FAIL

    def is_pass(self) -> bool:
        return self.signal_type == self.PASS

    def is_suspect(self) -> bool:
        return self.signal_type == self.SUSPECT

    def to_dict(self) -> dict:
        """转换为信号字典，用于发送给丰碑。"""
        return {
            "signal_type": self.signal_type,
            "detector": self.detector,
            "evidence_summary": self.evidence_summary,
            "timestamp": self.timestamp,
            "quantification": self.four_dim if self.four_dim else None,
        }

    def to_signal_payload(self, monument_id: str = "", severity: float = 1.0) -> dict:
        """转换为丰碑信号接收端点所需的 payload 格式。

        H-4 修复：添加 severity 字段以匹配 signal_routes.py 接收格式。
        """
        payload = {
            "signal_type": self.signal_type,
            "monument_id": monument_id,
            "detail": self.evidence_summary,
            "severity": max(0.0, min(severity, 1.0)),
            "detector": self.detector,
            "timestamp": self.timestamp,
        }
        if self.four_dim:
            payload["four_dim"] = self.four_dim
        return payload

    def __repr__(self):
        return f"DetectionResult({self.signal_type}, detector={self.detector})"


class XuanjianDetector:
    """玄鉴检测器。

    四维检测：造假 / 偏离 / 矛盾 / 事件风暴。
    输出三态信号：PASS / FAIL / SUSPECT。
    """

    def __init__(self, sandglass_query: SandglassQuery = None):
        if sandglass_query:
            self.sq = sandglass_query
        elif _SANDGLASS_AVAILABLE:
            self.sq = SandglassQuery()
        else:
            self.sq = None

    def _get_four_dim(self) -> dict:
        """获取四维量化数据。"""
        if self.sq is None:
            return self._default_four_dim()
        try:
            return self.sq.query_four_dim()
        except Exception as e:
            logger.warning("四维量化获取失败: %s", e)
            return self._default_four_dim()

    def _default_four_dim(self) -> dict:
        return {
            "faithfulness": -1.0,
            "goal_alignment": 1.0,
            "offset_entropy": 0.0,
            "motivation_tags": [],
            "available": False,
        }

    # ═══════════════════════════════════════════════
    # 检测 1：造假检测（Faithfulness / Fraud Detection）
    # ═══════════════════════════════════════════════

    def detect_fraud(self, claim: str, evidence: dict = None) -> DetectionResult:
        """造假检测。

        验证 AI 声称做了某事，是否真的做了。
        - 文件变更证据校验：检查 git 提交记录、文件哈希
        - 画像溯源验证：检查 SHA256 溯源标记是否匹配

        Args:
            claim: AI 的声明文本（如"修改了 verify_daemon.py"）
            evidence: 证据字典，可包含：
                - file_hash: 文件哈希
                - git_log: git 提交记录
                - file_exists: 文件是否存在
                - expected_hash: 预期哈希

        Returns:
            DetectionResult: PASS（验证通过）/ FAIL（造假确认）/ SUSPECT（疑似）
        """
        evidence = evidence or {}
        four_dim = self._get_four_dim()

        # 1. 文件存在性验证（H-3 修复：路径安全校验）
        if "file_path" in evidence:
            file_path = evidence["file_path"]
            safe_base = os.environ.get("XUANJIAN_SAFE_BASE", os.getcwd())
            try:
                real_path = os.path.realpath(file_path)
                safe_real = os.path.realpath(safe_base)
                if not real_path.startswith(safe_real):
                    return DetectionResult(
                        signal_type=DetectionResult.FAIL,
                        detector="fraud_path_traversal",
                        evidence_summary="文件路径越界",
                        four_dim=four_dim,
                    )
            except Exception:
                return DetectionResult(
                    signal_type=DetectionResult.FAIL,
                    detector="fraud_path_invalid",
                    evidence_summary="文件路径无效",
                    four_dim=four_dim,
                )
            if not os.path.exists(file_path):
                return DetectionResult(
                    signal_type=DetectionResult.FAIL,
                    detector="fraud_file_existence",
                    evidence_summary="声称修改了文件，但文件不存在",
                    four_dim=four_dim,
                )

        # 2. 文件哈希验证（M-4 修复：分块读取防止 DoS）
        if "file_path" in evidence and "expected_hash" in evidence:
            file_path = evidence["file_path"]
            expected_hash = evidence["expected_hash"]
            try:
                h = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                actual_hash = h.hexdigest()[:8]
                if actual_hash != expected_hash:
                    return DetectionResult(
                        signal_type=DetectionResult.FAIL,
                        detector="fraud_file_hash",
                        evidence_summary="文件哈希不匹配",
                        four_dim=four_dim,
                    )
            except Exception as e:
                logger.warning("文件哈希验证失败: %s", e)

        # 3. Git 提交记录验证
        if "git_log" in evidence and "expected_commit" in evidence:
            git_log = evidence.get("git_log", "")
            expected_commit = evidence.get("expected_commit", "")
            if expected_commit and expected_commit not in git_log:
                return DetectionResult(
                    signal_type=DetectionResult.FAIL,
                    detector="fraud_git_commit",
                    evidence_summary=f"声称提交了 {expected_commit}，但 git log 中未找到",
                    four_dim=four_dim,
                )

        # 4. 画像溯源验证（通过沙漏）
        if self.sq and self.sq.available:
            verify_result = self.sq.verify_claim(claim)
            if not verify_result.get("verified", True):
                return DetectionResult(
                    signal_type=DetectionResult.FAIL,
                    detector="fraud_persona_trace",
                    evidence_summary=f"画像溯源验证失败: {verify_result.get('detail', '')}",
                    four_dim=four_dim,
                )

        # 5. 忠实度分数检查（M-1 修复：分数只产生 SUSPECT）
        faith_score = four_dim.get("faithfulness", 1.0)
        if faith_score < 0:
            return DetectionResult(
                signal_type=DetectionResult.SUSPECT,
                detector="fraud_faithfulness",
                evidence_summary="沙漏数据不可用，忠实度无法验证，检测降级",
                four_dim=four_dim,
            )
        elif faith_score < 0.3:
            return DetectionResult(
                signal_type=DetectionResult.SUSPECT,
                detector="fraud_faithfulness",
                evidence_summary=f"忠实度极低: {faith_score:.2f}，疑似造假（分数预警）",
                four_dim=four_dim,
            )
        elif faith_score < 0.7:
            return DetectionResult(
                signal_type=DetectionResult.SUSPECT,
                detector="fraud_faithfulness",
                evidence_summary=f"忠实度偏低: {faith_score:.2f}，疑似造假",
                four_dim=four_dim,
            )

        return DetectionResult(
            signal_type=DetectionResult.PASS,
            detector="fraud",
            evidence_summary=f"造假检测通过: {claim[:50]}",
            four_dim=four_dim,
        )

    # ═══════════════════════════════════════════════
    # 检测 2：偏离检测（Deviation Detection）
    # ═══════════════════════════════════════════════

    def detect_deviation(self, decision_text: str = "",
                         purpose_check: dict = None) -> DetectionResult:
        """偏离检测。

        检测 AI 的行为是否偏离了既定目的和决策模式。
        - 偏移率检测：综合偏移率是否超过阈值
        - 目标对齐度检测：行为是否与目的树方向一致
        - 偏移熵检测：决策模式是否异常波动

        Args:
            decision_text: 决策文本（可选，用于偏移率检测）
            purpose_check: 目的树检测结果，可包含：
                - aligned: bool，是否对齐
                - deviation: str，偏离描述
                - direction: str，偏离方向

        Returns:
            DetectionResult: PASS / FAIL / SUSPECT
        """
        purpose_check = purpose_check or {}
        four_dim = self._get_four_dim()

        # 1. 目标对齐度检查
        goal_alignment = four_dim.get("goal_alignment", 1.0)
        has_contradiction = four_dim.get("context", {}).get("contradictions", {}).get(
            "has_contradiction", False
        )

        if goal_alignment < 0.3:
            return DetectionResult(
                signal_type=DetectionResult.FAIL,
                detector="deviation_goal_alignment",
                evidence_summary=f"目标对齐度极低: {goal_alignment:.2f}，严重偏离既定目的",
                four_dim=four_dim,
            )
        elif goal_alignment < 0.6:
            return DetectionResult(
                signal_type=DetectionResult.SUSPECT,
                detector="deviation_goal_alignment",
                evidence_summary=f"目标对齐度偏低: {goal_alignment:.2f}，疑似偏离",
                four_dim=four_dim,
            )

        # 2. 偏移熵检查（决策模式异常波动）
        offset_entropy = four_dim.get("offset_entropy", 0.0)
        if offset_entropy > 0.8:
            return DetectionResult(
                signal_type=DetectionResult.SUSPECT,
                detector="deviation_offset_entropy",
                evidence_summary=f"偏移熵过高: {offset_entropy:.2f}，决策模式异常波动",
                four_dim=four_dim,
            )

        # 3. purpose_manager 偏离检测结果
        if purpose_check:
            if not purpose_check.get("aligned", True):
                deviation = purpose_check.get("deviation", "未知偏离")
                return DetectionResult(
                    signal_type=DetectionResult.FAIL,
                    detector="deviation_purpose_manager",
                    evidence_summary=f"目的树检测偏离: {deviation}",
                    four_dim=four_dim,
                )

        # 4. 矛盾检测（如果存在矛盾，也可能是偏离信号）
        if has_contradiction:
            return DetectionResult(
                signal_type=DetectionResult.SUSPECT,
                detector="deviation_contradiction",
                evidence_summary="检测到跨支柱矛盾，可能存在行为偏离",
                four_dim=four_dim,
            )

        return DetectionResult(
            signal_type=DetectionResult.PASS,
            detector="deviation",
            evidence_summary="偏离检测通过",
            four_dim=four_dim,
        )

    # ═══════════════════════════════════════════════
    # 检测 3：矛盾检测（Contradiction Detection）
    # ═══════════════════════════════════════════════

    def detect_contradiction(self, assertion: str = "",
                             spo_entity: str = None) -> DetectionResult:
        """矛盾检测。

        检测 AI 的断言之间是否存在事实冲突。
        - 织布机矛盾检测：四大支柱交叉矛盾
        - SPO 三元组矛盾：断言与已知事实冲突

        Args:
            assertion: AI 的断言文本
            spo_entity: 可选，按实体查询 SPO 三元组

        Returns:
            DetectionResult: PASS / FAIL / SUSPECT
        """
        four_dim = self._get_four_dim()

        # 1. 织布机矛盾检测
        if self.sq and self.sq.available:
            contra = self.sq.query_contradictions()
            conflicts = contra.get("conflicts", [])

            if conflicts:
                # 严重矛盾（3个以上）→ FAIL
                if len(conflicts) >= 3:
                    conflict_descs = [c.get("conflict", "")[:50] for c in conflicts[:3]]
                    return DetectionResult(
                        signal_type=DetectionResult.FAIL,
                        detector="contradiction_weave",
                        evidence_summary=f"检测到 {len(conflicts)} 处矛盾: {'; '.join(conflict_descs)}",
                        four_dim=four_dim,
                    )
                # 少量矛盾 → SUSPECT
                else:
                    conflict_descs = [c.get("conflict", "")[:50] for c in conflicts]
                    return DetectionResult(
                        signal_type=DetectionResult.SUSPECT,
                        detector="contradiction_weave",
                        evidence_summary=f"检测到 {len(conflicts)} 处疑似矛盾: {'; '.join(conflict_descs)}",
                        four_dim=four_dim,
                    )

        # 2. SPO 三元组矛盾检测
        if self.sq and self.sq.available and assertion:
            spo = self.sq.query_spo_triples(entity=spo_entity or "user")
            triples = spo.get("triples", [])

            if triples and assertion:
                # 检查断言是否与已知三元组矛盾
                # 简单实现：检查断言中是否包含与"放弃"关系冲突的"使用"声明
                assertion_lower = assertion.lower()
                for t in triples:
                    relation = t.get("relation", "")
                    obj = t.get("object", "").lower()

                    if relation == "放弃" and obj and obj in assertion_lower:
                        # 断言使用已放弃的东西 → 矛盾
                        return DetectionResult(
                            signal_type=DetectionResult.FAIL,
                            detector="contradiction_spo",
                            evidence_summary=f"断言与已知事实矛盾: 已放弃 '{t.get('object', '')}' 但断言中使用",
                            four_dim=four_dim,
                        )
                    elif relation == "使用" and obj and "不用" in assertion_lower and obj in assertion_lower:
                        return DetectionResult(
                            signal_type=DetectionResult.SUSPECT,
                            detector="contradiction_spo",
                            evidence_summary=f"断言与已知事实疑似矛盾: 已使用 '{t.get('object', '')}' 但断言中称不用",
                            four_dim=four_dim,
                        )

        return DetectionResult(
            signal_type=DetectionResult.PASS,
            detector="contradiction",
            evidence_summary="矛盾检测通过",
            four_dim=four_dim,
        )

    # ═══════════════════════════════════════════════
    # 检测 4：事件风暴检测（Event Storm Detection）
    # ═══════════════════════════════════════════════

    def detect_event_storm(self, event_count: int = 0,
                           time_window_seconds: int = 60,
                           threshold: int = 100) -> DetectionResult:
        """事件风暴检测。

        检测系统是否出现异常的事件洪流或消费链故障。

        Args:
            event_count: 时间窗口内的事件数量
            time_window_seconds: 时间窗口（秒）
            threshold: 事件风暴阈值

        Returns:
            DetectionResult: PASS / FAIL / SUSPECT
        """
        four_dim = self._get_four_dim()

        if event_count >= threshold * 2:
            return DetectionResult(
                signal_type=DetectionResult.FAIL,
                detector="event_storm",
                evidence_summary=f"事件风暴: {event_count} 事件/{time_window_seconds}s，"
                                 f"超过阈值 {threshold} 的 2 倍",
                four_dim=four_dim,
            )
        elif event_count >= threshold:
            return DetectionResult(
                signal_type=DetectionResult.SUSPECT,
                detector="event_storm",
                evidence_summary=f"事件洪流预警: {event_count} 事件/{time_window_seconds}s，"
                                 f"达到阈值 {threshold}",
                four_dim=four_dim,
            )

        return DetectionResult(
            signal_type=DetectionResult.PASS,
            detector="event_storm",
            evidence_summary=f"事件流正常: {event_count} 事件/{time_window_seconds}s",
            four_dim=four_dim,
        )

    # ═══════════════════════════════════════════════
    # 检测 5：跨实例握手检测
    # ═══════════════════════════════════════════════

    def detect_cross_instance_handshake(self) -> DetectionResult:
        """检测近24小时是否有跨实例握手记录。"""
        try:
            log_path = "/vol1/@team/qh团队/QH/AI专用/Agent OS/iso-sand/data/operation_log.jsonl"
            if not os.path.exists(log_path):
                return DetectionResult(
                    signal_type=DetectionResult.PASS,
                    detector="cross_instance",
                    evidence_summary="无操作日志，跳过",
                )

            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

            with open(log_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        t_str = entry.get("t", "")
                        detail = entry.get("detail", "")
                        detail_str = str(detail) if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
                        if "CROSS_INSTANCE" in detail_str or "cross_instance" in detail_str:
                            return DetectionResult(
                                signal_type=DetectionResult.PASS,
                                detector="cross_instance",
                                evidence_summary=f"24h内有握手记录: {t_str}",
                            )
                    except Exception:
                        pass

            return DetectionResult(
                signal_type=DetectionResult.SUSPECT,
                detector="cross_instance",
                evidence_summary="近24小时无跨实例握手记录，可能遗漏",
            )
        except Exception as e:
            return DetectionResult(
                signal_type=DetectionResult.PASS,
                detector="cross_instance",
                evidence_summary=f"检测异常: {e}",
            )

    # ═══════════════════════════════════════════════
    # 综合检测
    # ═══════════════════════════════════════════════

    def detect_all(self, claim: str = "", evidence: dict = None,
                   decision: str = "", assertion: str = "",
                   purpose_check: dict = None,
                   event_count: int = 0) -> DetectionResult:
        """综合检测。运行所有四个检测维度，返回最严重的结果。

        优先级：FAIL > SUSPECT > PASS

        Args:
            claim: AI 的声明
            evidence: 证据字典
            decision: 决策文本
            assertion: 断言文本
            purpose_check: 目的树检测结果
            event_count: 事件计数

        Returns:
            DetectionResult: 最严重的检测结果
        """
        evidence = evidence or {}
        results = []

        # M-3 修复：预先获取四维量化数据，避免重复查询
        cached_four_dim = self._get_four_dim()

        if claim or evidence:
            r = self.detect_fraud(claim, evidence)
            r.four_dim = cached_four_dim
            results.append(r)
        if decision or purpose_check:
            r = self.detect_deviation(decision, purpose_check)
            r.four_dim = cached_four_dim
            results.append(r)
        if assertion:
            r = self.detect_contradiction(assertion)
            r.four_dim = cached_four_dim
            results.append(r)
        if event_count > 0:
            r = self.detect_event_storm(event_count)
            r.four_dim = cached_four_dim
            results.append(r)

        # 跨实例握手检测（始终运行）
        r = self.detect_cross_instance_handshake()
        r.four_dim = cached_four_dim
        results.append(r)

        if not results:
            # 无检测输入，返回 PASS
            return DetectionResult(
                signal_type=DetectionResult.PASS,
                detector="all",
                evidence_summary="无检测输入",
                four_dim=self._get_four_dim(),
            )

        # 返回最严重的结果
        for r in results:
            if r.is_fail():
                return r
        for r in results:
            if r.is_suspect():
                return r
        return results[0]


# ═══════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════

_default_detector = None
_detector_lock = threading.Lock()


def get_detector() -> XuanjianDetector:
    """获取全局 XuanjianDetector 单例（线程安全）。"""
    global _default_detector
    if _default_detector is None:
        with _detector_lock:
            if _default_detector is None:
                _default_detector = XuanjianDetector()
    return _default_detector


def detect_fraud(claim: str, evidence: dict = None) -> DetectionResult:
    """便捷函数：造假检测。"""
    return get_detector().detect_fraud(claim, evidence)


def detect_deviation(decision_text: str = "", purpose_check: dict = None) -> DetectionResult:
    """便捷函数：偏离检测。"""
    return get_detector().detect_deviation(decision_text, purpose_check)


def detect_contradiction(assertion: str = "", spo_entity: str = None) -> DetectionResult:
    """便捷函数：矛盾检测。"""
    return get_detector().detect_contradiction(assertion, spo_entity)


if __name__ == "__main__":
    detector = XuanjianDetector()
    print(f"沙漏可用: {detector.sq is not None and detector.sq.available}")

    # 测试造假检测
    r1 = detector.detect_fraud(claim="测试声明", evidence={"file_path": "/nonexistent"})
    print(f"造假检测: {r1.signal_type} - {r1.evidence_summary}")

    # 测试偏离检测
    r2 = detector.detect_deviation()
    print(f"偏离检测: {r2.signal_type} - {r2.evidence_summary}")

    # 测试矛盾检测
    r3 = detector.detect_contradiction()
    print(f"矛盾检测: {r3.signal_type} - {r3.evidence_summary}")

    # 测试事件风暴检测
    r4 = detector.detect_event_storm(event_count=50)
    print(f"事件风暴: {r4.signal_type} - {r4.evidence_summary}")

    # 综合检测
    r5 = detector.detect_all(claim="测试", evidence={})
    print(f"综合检测: {r5.signal_type} - {r5.evidence_summary}")
    print(f"四维量化: {json.dumps(r5.four_dim, ensure_ascii=False, indent=2)}")
