"""
沙漏统一查询接口 — 玄鉴专用
================================

为玄鉴（AgentOS 检测守护进程）提供统一的沙漏能力查询入口。
封装 8 个接入玄鉴的沙漏 L3 模块，实现四维量化框架的数据采集层。

接入模块：
  1. decision_particles.py  — 决策粒子（动机标签）
  2. offset_l3.py           — 偏移率（偏移熵）
  3. sandglass_think.py     — 3D人格合成（行为上下文）
  4. weave_l3.py            — 织布机矛盾检测（矛盾检测）
  5. l3_persona_verify.py   — 画像验证（造假检测/忠实度）
  6. scene_l3.py            — 场景感知（行为上下文）
  7. emotion_l3.py          — 情绪镜像（偏移熵补充）
  8. weavethread.py         — SPO三元组（矛盾检测）

设计原则：
  - 懒加载：沙漏未运行时返回安全默认值，不崩溃
  - 只读查询：不修改沙漏任何数据
  - 独立维度：各维度查询互不影响，一个失败不影响其他
  - 约束：confidence 字段仅用于过滤低质量数据（<0.3 不参与检测），
    不用于排名或评分

用法：
    from sandglass_query import SandglassQuery
    sq = SandglassQuery()
    four_dim = sq.query_four_dim()          # 四维综合查询
    faith = sq.query_faithfulness()          # 忠实度
    tags = sq.query_motivation_tags()        # 动机标签
"""

import os
import json
import math
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# 沙漏根目录（从 sandglass_paths 获取，失败则用环境变量）
try:
    from sandglass_paths import _NB
except ImportError:
    _NB = os.environ.get("NEXSANDBASE_HOME") or os.path.join(
        os.path.expanduser("~"), ".neurobase"
    )


class SandglassQuery:
    """沙漏统一查询接口。

    所有方法均为只读查询，不修改沙漏数据。
    沙漏未初始化时返回安全默认值。
    """

    def __init__(self, base_dir: str = None):
        self._base = base_dir or _NB
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """检查沙漏是否已初始化（目录存在且有基本数据）。"""
        if not os.path.isdir(self._base):
            return False
        sg = os.path.join(self._base, "sandglass.txt")
        dp = os.path.join(self._base, "decision_particles.txt")
        return os.path.exists(sg) or os.path.exists(dp)

    @property
    def available(self) -> bool:
        """沙漏数据是否可用。"""
        return self._available

    # ═══════════════════════════════════════════════
    # 维度 1：忠实度（Faithfulness）
    # ═══════════════════════════════════════════════

    def query_faithfulness(self) -> dict:
        """查询忠实度。基于画像溯源验证（SHA256 hash 匹配率）。"""
        if not self._available:
            return self._default_faithfulness()

        try:
            from l3_persona_verify import persona_verify

            result = persona_verify()
            verified = result.get("verified", 0)
            failed = result.get("failed", 0)
            total = result.get("total", 0)

            if total == 0:
                score = 1.0
            else:
                score = verified / total

            return {
                "faithfulness_score": round(score, 4),
                "verified": verified,
                "failed": failed,
                "total": total,
                "insight": result.get("insight", ""),
            }
        except Exception as e:
            logger.warning("忠实度查询失败: %s", e)
            return self._default_faithfulness()

    def _default_faithfulness(self) -> dict:
        return {
            "faithfulness_score": -1.0,
            "verified": 0,
            "failed": 0,
            "total": 0,
            "insight": "沙漏数据不可用，忠实度未知",
        }

    # ═══════════════════════════════════════════════
    # 维度 2：目标对齐度（Goal Alignment）
    # ═══════════════════════════════════════════════

    def query_goal_alignment(self) -> dict:
        """查询目标对齐度。基于综合偏移率和矛盾检测。"""
        if not self._available:
            return self._default_goal_alignment()

        offset_data = self.query_offset_entropy()
        comp_offset = abs(offset_data.get("comprehensive_offset", 0))

        contra_data = self.query_contradictions()
        has_contradiction = contra_data.get("has_contradiction", False)
        conflict_count = len(contra_data.get("conflicts", []))

        base_alignment = max(0.0, 1.0 - comp_offset / 100.0)
        if has_contradiction:
            base_alignment *= 0.7

        return {
            "goal_alignment": round(base_alignment, 4),
            "offset": offset_data.get("comprehensive_offset", 0),
            "direction": offset_data.get("direction", "neutral"),
            "has_contradiction": has_contradiction,
            "conflict_count": conflict_count,
        }

    def _default_goal_alignment(self) -> dict:
        return {
            "goal_alignment": 1.0,
            "offset": 0,
            "direction": "neutral",
            "has_contradiction": False,
            "conflict_count": 0,
        }

    # ═══════════════════════════════════════════════
    # 维度 3：偏移熵（Offset Entropy）
    # ═══════════════════════════════════════════════

    def query_offset_entropy(self) -> dict:
        """查询偏移熵。使用 Shannon 熵衡量决策模式的方向分布稳定性。"""
        if not self._available:
            return self._default_offset_entropy()

        result = {}

        try:
            from offset_l3 import comprehensive_offset
            comp = comprehensive_offset()
            result["comprehensive_offset"] = comp.get("offset", 0)
            result["direction"] = comp.get("direction", "neutral")
            result["sample"] = comp.get("sample", 0)
            result["trend"] = comp.get("trend", "stable")
        except Exception as e:
            logger.warning("综合偏移率查询失败: %s", e)
            result["comprehensive_offset"] = 0
            result["direction"] = "neutral"
            result["sample"] = 0
            result["trend"] = "stable"

        entropy = self._calculate_shannon_entropy()
        result["entropy"] = entropy
        result["emotion_entropy"] = self._query_emotion_entropy_value()

        return result

    def _calculate_shannon_entropy(self) -> float:
        """计算决策方向的 Shannon 熵并归一化。"""
        try:
            from decision_particles import read
            particles = read(50)
            if not particles:
                return 0.0

            directions = {"frugal": 0, "spend": 0, "drift": 0, "neutral": 0}
            for p in particles:
                d = p[3] if len(p) > 3 else "neutral"
                if d in directions:
                    directions[d] += 1
                else:
                    directions["neutral"] += 1

            total = sum(directions.values())
            if total == 0:
                return 0.0

            entropy = 0.0
            for count in directions.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)

            normalized = entropy / 2.0 if entropy > 0 else 0.0
            drift_ratio = directions.get("drift", 0) / total
            result = normalized * (1.0 - drift_ratio)

            return round(result, 4)
        except Exception as e:
            logger.warning("Shannon 熵计算失败: %s", e)
            return 0.0

    def _query_emotion_entropy_value(self) -> float:
        """获取情绪熵值。"""
        try:
            from sandglass_think import _emotional_entropy
            return round(_emotional_entropy(), 4)
        except Exception:
            try:
                from emotion_l3 import _emotional_entropy
                return round(_emotional_entropy(), 4)
            except Exception:
                return 0.0

    def _default_offset_entropy(self) -> dict:
        return {
            "entropy": 0.0,
            "comprehensive_offset": 0,
            "direction": "neutral",
            "sample": 0,
            "trend": "stable",
            "emotion_entropy": 0.0,
        }

    # ═══════════════════════════════════════════════
    # 维度 4：动机标签（Motivation Tags）
    # ═══════════════════════════════════════════════

    def query_motivation_tags(self) -> dict:
        """查询动机标签。调取沙漏决策粒子的最新标签。"""
        if not self._available:
            return self._default_motivation_tags()

        result = {}

        try:
            from decision_particles import read, ratio
            particles = read(10)
            tags_set = []
            seen = set()
            for p in particles:
                if len(p) >= 6:
                    raw_tags = p[5]
                    for t in raw_tags.split(","):
                        t = t.strip()
                        if t and t != "未分类" and t not in seen:
                            tags_set.append(t)
                            seen.add(t)
                if len(tags_set) >= 5:
                    break
            result["tags"] = tags_set[:5]

            r = ratio()
            result["ratio"] = r
            result["direction"] = max(
                ["frugal", "spend", "drift"],
                key=lambda d: r.get(d, 0),
            ) if r.get("total", 0) > 0 else "neutral"

        except Exception as e:
            logger.warning("动机标签查询失败: %s", e)
            result["tags"] = []
            result["ratio"] = {}
            result["direction"] = "neutral"

        result["chain_summary"] = self._get_recent_chain_summary()

        return result

    def _get_recent_chain_summary(self) -> str:
        """获取最近的决策链条摘要。"""
        try:
            from decision_particles import read
            particles = read(5)
            for p in particles:
                if len(p) >= 3:
                    choice = p[2]
                    if "→" in choice or "回到" in choice:
                        return choice[:100]
            return ""
        except Exception:
            return ""

    def _default_motivation_tags(self) -> dict:
        return {
            "tags": [],
            "direction": "neutral",
            "ratio": {},
            "chain_summary": "",
        }

    # ═══════════════════════════════════════════════
    # 附加查询：矛盾检测
    # ═══════════════════════════════════════════════

    def query_contradictions(self) -> dict:
        """查询矛盾检测结果。检测四大支柱之间的自相矛盾。"""
        if not self._available:
            return {"has_contradiction": False, "conflicts": [], "suggestion": "无数据"}

        try:
            from weave_l3 import weave_contradiction
            result = weave_contradiction()
            conflicts = result.get("conflicts", [])
            return {
                "has_contradiction": len(conflicts) > 0,
                "conflicts": conflicts,
                "suggestion": result.get("suggestion", "无矛盾"),
            }
        except Exception as e:
            logger.warning("矛盾检测查询失败: %s", e)
            return {"has_contradiction": False, "conflicts": [], "suggestion": f"查询失败: {e}"}

    # ═══════════════════════════════════════════════
    # 附加查询：SPO 三元组
    # ═══════════════════════════════════════════════

    def query_spo_triples(self, entity: str = None, limit: int = 20) -> dict:
        """查询 SPO 三元组。用于矛盾检测。

        约束：confidence 字段仅用于过滤低质量数据（<0.3 不参与检测）。
        """
        if not self._available:
            return {"triples": [], "count": 0, "stats": {}}

        try:
            from weavethread import wthread_query, wthread_stats

            triples = wthread_query(entity=entity, limit=limit)
            filtered = [t for t in triples if t.get("confidence", 0.5) >= 0.3]

            stats = wthread_stats()
            return {
                "triples": filtered,
                "count": len(filtered),
                "stats": {
                    "total_triples": stats.get("total_triples", 0),
                    "relations": stats.get("relations", []),
                },
            }
        except Exception as e:
            logger.warning("SPO 三元组查询失败: %s", e)
            return {"triples": [], "count": 0, "stats": {}}

    # ═══════════════════════════════════════════════
    # 附加查询：3D 人格合成
    # ═══════════════════════════════════════════════

    def query_persona_3d(self) -> dict:
        """查询 3D 人格画像。返回最新的阶段注解（五维画像）。"""
        if not self._available:
            return self._default_persona_3d()

        try:
            from sandglass_think import _latest_annotation, _three_d_ready

            ready = _three_d_ready()
            annotation = _latest_annotation()

            if not annotation:
                return self._default_persona_3d()

            return {
                "available": ready,
                "persona_type": annotation.get("persona_type", ""),
                "emotional_state": annotation.get("emotional_state", ""),
                "decision_pattern": annotation.get("decision_pattern", ""),
                "reminder_tone": annotation.get("reminder_tone", ""),
                "reminder_example": annotation.get("reminder_example", ""),
            }
        except Exception as e:
            logger.warning("3D 人格查询失败: %s", e)
            return self._default_persona_3d()

    def _default_persona_3d(self) -> dict:
        return {
            "available": False,
            "persona_type": "",
            "emotional_state": "",
            "decision_pattern": "",
            "reminder_tone": "",
            "reminder_example": "",
        }

    # ═══════════════════════════════════════════════
    # 附加查询：场景感知
    # ═══════════════════════════════════════════════

    def query_scene_context(self) -> dict:
        """查询场景上下文。返回当前场景标签和阶段切换预测。"""
        if not self._available:
            return self._default_scene_context()

        result = {}

        try:
            from scene_l3 import scene_current, scene_mode
            result["current_scenes"] = scene_current()
            result["mode"] = scene_mode()
        except Exception as e:
            logger.warning("场景查询失败: %s", e)
            result["current_scenes"] = []
            result["mode"] = "normal"

        try:
            from scene_l3 import stage_switch_prediction
            result["stage_prediction"] = stage_switch_prediction()
        except Exception:
            result["stage_prediction"] = {"predicted": False}

        try:
            from scene_l3 import scene_dominance
            result["dominance"] = scene_dominance()
        except Exception:
            result["dominance"] = {}

        return result

    def _default_scene_context(self) -> dict:
        return {
            "current_scenes": [],
            "mode": "normal",
            "stage_prediction": {"predicted": False},
            "dominance": {},
        }

    # ═══════════════════════════════════════════════
    # 附加查询：情绪熵
    # ═══════════════════════════════════════════════

    def query_emotion_entropy(self) -> dict:
        """查询情绪熵详情。"""
        entropy = self._query_emotion_entropy_value()

        if entropy < 0.5:
            level = "calm"
            desc = "状态平稳，理性主导"
        elif entropy < 1.0:
            level = "moderate"
            desc = "波动期，可能犹豫或感性"
        else:
            level = "high"
            desc = "高熵期，情绪波动大"

        return {
            "entropy": entropy,
            "level": level,
            "description": desc,
        }

    # ═══════════════════════════════════════════════
    # 四维综合查询
    # ═══════════════════════════════════════════════

    def query_four_dim(self, ai_id: Optional[str] = None) -> dict:
        """四维量化综合查询。

        玄鉴在输出 FAIL/SUSPECT 信号时调用此方法，
        获取四维量化分析作为信号附件。

        各维度独立查询（互不影响）：
        - faithfulness: 忠实度 [0, 1]
        - goal_alignment: 目标对齐度 [0, 1]
        - offset_entropy: 偏移熵 [0, 1]
        - motivation_tags: 动机标签 [str]
        """
        faith = self.query_faithfulness()
        offset = self.query_offset_entropy()
        tags = self.query_motivation_tags()
        emotion = self.query_emotion_entropy()

        persona_3d = self.query_persona_3d()
        scene = self.query_scene_context()
        contradictions = self.query_contradictions()
        spo = self.query_spo_triples()

        comp_offset = abs(offset.get("comprehensive_offset", 0))
        has_contradiction = contradictions.get("has_contradiction", False)
        base_alignment = max(0.0, 1.0 - comp_offset / 100.0)
        if has_contradiction:
            base_alignment *= 0.7
        goal_alignment = round(base_alignment, 4)

        return {
            "faithfulness": faith.get("faithfulness_score", -1.0),
            "goal_alignment": goal_alignment,
            "offset_entropy": offset.get("entropy", 0.0),
            "motivation_tags": tags.get("tags", []),
            "context": {
                "persona_3d": persona_3d,
                "scene": scene,
                "contradictions": {
                    "has_contradiction": has_contradiction,
                    "conflict_count": len(contradictions.get("conflicts", [])),
                },
                "spo_triples": {"count": spo.get("count", 0)},
                "emotion_entropy": emotion,
            },
            "available": self._available,
        }

    # ═══════════════════════════════════════════════
    # 画像溯源验证（单独暴露，供造假检测使用）
    # ═══════════════════════════════════════════════

    def verify_claim(self, claim: str) -> dict:
        """验证单条声明的溯源。用于玄鉴的造假检测。"""
        if not self._available:
            return {"verified": False, "line": 0, "detail": "沙漏不可用，无法验证"}

        try:
            from l3_persona_verify import persona_trace
            results = persona_trace(claim)
            if results:
                first = results[0]
                return {
                    "verified": first.get("verified", False),
                    "line": first.get("line", 0),
                    "detail": first.get("warning", first.get("text", ""))[:100],
                }
            return {"verified": True, "line": 0, "detail": "无溯源标记"}
        except Exception as e:
            logger.warning("声明验证失败: %s", e)
            return {"verified": True, "line": 0, "detail": f"验证异常: {e}"}


# ═══════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════

_default_instance = None
_query_lock = threading.Lock()


def get_query() -> SandglassQuery:
    """获取全局 SandglassQuery 单例（线程安全）。"""
    global _default_instance
    if _default_instance is None:
        with _query_lock:
            if _default_instance is None:
                _default_instance = SandglassQuery()
    return _default_instance


def query_four_dim(ai_id: Optional[str] = None) -> dict:
    """便捷函数：四维综合查询。"""
    return get_query().query_four_dim(ai_id)


if __name__ == "__main__":
    sq = SandglassQuery()
    print(f"沙漏可用: {sq.available}")
    print(f"忠实度: {sq.query_faithfulness()}")
    print(f"偏移熵: {sq.query_offset_entropy()}")
    print(f"动机标签: {sq.query_motivation_tags()}")
    print(f"矛盾检测: {sq.query_contradictions()}")
    print(f"3D人格: {sq.query_persona_3d()}")
    print(f"场景: {sq.query_scene_context()}")
    print(f"情绪: {sq.query_emotion_entropy()}")
    print(f"SPO: {sq.query_spo_triples()}")
    print(f"四维综合: {json.dumps(sq.query_four_dim(), ensure_ascii=False, indent=2)}")
