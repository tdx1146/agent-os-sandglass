"""
目的树偏离检测器
==================

读取 purpose_tree.yaml 约束配置，检查代码文件是否违反约束。
为玄鉴的 detect_deviation() 提供 purpose_check 参数。

设计原则（用户特别要求）：
  - 宁可漏报不可误报
  - 只有明确发现违规模式时才返回 aligned=False
  - 无法确定时返回 aligned=True（安全放行）
  - 文件不存在时返回 aligned=True（无法检查就放行）
  - 正则匹配失败时返回 aligned=True（不因正则错误误判）

用法：
    from purpose_tree_tracker import check_purpose_alignment

    purpose_check = check_purpose_alignment()
    # → {"aligned": True, "deviation": "", "direction": ""}

    # 传给玄鉴偏离检测
    detector.detect_deviation(purpose_check=purpose_check)

约束检查逻辑：
  1. 读取 purpose_tree.yaml
  2. 遍历每个约束
  3. 读取约束指定的代码文件
  4. 检查 forbidden_patterns（正则匹配）
  5. 检查 required_patterns（必须存在）
  6. 发现违规 → aligned=False
  7. 全部通过 → aligned=True
"""

import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# purpose_tree.yaml 的路径
_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "purpose_tree.yaml")


def _load_yaml():
    """加载 purpose_tree.yaml。

    返回 None 表示文件不存在或解析失败（安全降级）。
    """
    if not os.path.exists(_YAML_PATH):
        logger.warning("purpose_tree.yaml 不存在: %s", _YAML_PATH)
        return None

    try:
        import yaml
        with open(_YAML_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.warning("PyYAML 未安装，目的树检查跳过")
        return None
    except Exception as e:
        logger.warning("purpose_tree.yaml 解析失败: %s", e)
        return None


def _read_file_safe(filepath: str, base_dir: str = None) -> Optional[str]:
    """安全读取文件内容。

    文件不存在或读取失败时返回 None（不触发偏离）。
    """
    if base_dir:
        filepath = os.path.join(base_dir, filepath)

    if not os.path.exists(filepath):
        logger.debug("检查文件不存在，跳过: %s", filepath)
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning("文件读取失败: %s, error: %s", filepath, e)
        return None


def _check_constraint(constraint: dict, base_dir: str) -> Optional[dict]:
    """检查单个约束。

    返回 None 表示通过（或无法检查）。
    返回 dict 表示发现违规: {"id": ..., "description": ..., "evidence": ...}
    """
    constraint_id = constraint.get("id", "unknown")
    description = constraint.get("description", "")
    check_files = constraint.get("check_files", [])
    forbidden = constraint.get("forbidden_patterns", [])
    required = constraint.get("required_patterns", [])
    severity = constraint.get("severity", "medium")

    for filename in check_files:
        content = _read_file_safe(filename, base_dir)
        if content is None:
            # 文件不存在 → 无法检查 → 放行
            continue

        # 检查禁止模式
        for pattern in forbidden:
            try:
                if re.search(pattern, content):
                    return {
                        "id": constraint_id,
                        "description": description,
                        "evidence": f"文件 {filename} 中发现禁止模式: {pattern}",
                        "severity": severity,
                    }
            except re.error:
                # 正则编译失败 → 跳过此模式，不误判
                logger.warning("正则编译失败，跳过: %s", pattern)
                continue

        # 检查必需模式
        for pattern in required:
            try:
                if not re.search(pattern, content):
                    return {
                        "id": constraint_id,
                        "description": description,
                        "evidence": f"文件 {filename} 中缺少必需模式: {pattern}",
                        "severity": severity,
                    }
            except re.error:
                logger.warning("正则编译失败，跳过: %s", pattern)
                continue

    return None


def check_purpose_alignment(base_dir: str = None) -> dict:
    """检查当前代码是否与目的树对齐。

    这是主入口函数，返回 purpose_check 字典，
    直接传给 XuanjianDetector.detect_deviation(purpose_check=...)。

    Args:
        base_dir: 代码根目录（默认为 purpose_tree_tracker.py 所在目录）

    Returns:
        {
            "aligned": bool,      # True=对齐, False=偏离
            "deviation": str,     # 偏离描述（aligned=True 时为空）
            "direction": str,     # 偏离方向（aligned=True 时为空）
            "violations": list,   # 违规详情列表
        }

    安全保证：
        - purpose_tree.yaml 不存在 → aligned=True
        - PyYAML 未安装 → aligned=True
        - 检查文件不存在 → 该约束跳过（不触发偏离）
        - 正则编译失败 → 该模式跳过（不触发偏离）
        - 任何异常 → aligned=True（不因检查器故障误判）
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    config = _load_yaml()
    if config is None:
        return {
            "aligned": True,
            "deviation": "",
            "direction": "",
            "violations": [],
        }

    violations = []

    # 遍历所有约束组
    for section_key in ["identity", "signal_pathway", "erosion_mechanism"]:
        section = config.get(section_key, {})
        constraints = section.get("constraints", [])

        for constraint in constraints:
            result = _check_constraint(constraint, base_dir)
            if result is not None:
                violations.append(result)

    if violations:
        # 发现违规 → 偏离
        high_violations = [v for v in violations if v.get("severity") == "high"]
        if high_violations:
            # 高严重度违规 → FAIL
            deviation_desc = high_violations[0]["description"]
            evidence = high_violations[0]["evidence"]
            return {
                "aligned": False,
                "deviation": f"{deviation_desc} ({evidence})",
                "direction": "high_severity",
                "violations": violations,
            }
        else:
            # 中低严重度违规 → 也标记偏离，但 severity 较低
            deviation_desc = violations[0]["description"]
            evidence = violations[0]["evidence"]
            return {
                "aligned": False,
                "deviation": f"{deviation_desc} ({evidence})",
                "direction": "medium_severity",
                "violations": violations,
            }

    # 全部通过
    return {
        "aligned": True,
        "deviation": "",
        "direction": "",
        "violations": [],
    }


def get_purpose_summary() -> dict:
    """获取目的树摘要信息（不含检查结果）。"""
    config = _load_yaml()
    if config is None:
        return {"available": False, "sections": []}

    sections = []
    for key in ["identity", "signal_pathway", "erosion_mechanism"]:
        section = config.get(key, {})
        if section:
            constraints = section.get("constraints", [])
            sections.append({
                "name": key,
                "description": section.get("description", ""),
                "constraint_count": len(constraints),
            })

    return {
        "available": True,
        "sections": sections,
    }


# ═══════════════════════════════════════════════════════════════
# 便捷函数（线程安全单例）
# ═══════════════════════════════════════════════════════════════

import threading

_last_check = None
_last_check_time = 0
_check_lock = threading.Lock()
_CHECK_CACHE_SECONDS = 60  # 缓存 60 秒，避免频繁文件 IO


def get_cached_purpose_check(force_refresh: bool = False) -> dict:
    """获取带缓存的目的树检查结果。

    60 秒内复用上次检查结果，避免频繁读取文件。
    守护线程每次巡检调用此函数即可。

    Args:
        force_refresh: 强制刷新缓存

    Returns:
        purpose_check 字典
    """
    global _last_check, _last_check_time

    import time
    now = time.time()

    if not force_refresh and _last_check is not None:
        if now - _last_check_time < _CHECK_CACHE_SECONDS:
            return _last_check

    with _check_lock:
        # Double-checked locking
        if not force_refresh and _last_check is not None:
            if now - _last_check_time < _CHECK_CACHE_SECONDS:
                return _last_check

        _last_check = check_purpose_alignment()
        _last_check_time = time.time()
        return _last_check


if __name__ == "__main__":
    print("=" * 50)
    print("目的树偏离检测 — 自检")
    print("=" * 50)

    summary = get_purpose_summary()
    print(f"\n目的树可用: {summary['available']}")
    for s in summary.get("sections", []):
        print(f"  {s['name']}: {s['constraint_count']} 个约束 - {s['description']}")

    result = check_purpose_alignment()
    print(f"\n对齐状态: {'✅ 对齐' if result['aligned'] else '❌ 偏离'}")
    if not result["aligned"]:
        print(f"偏离描述: {result['deviation']}")
        print(f"偏离方向: {result['direction']}")
        print(f"违规数量: {len(result['violations'])}")
        for v in result["violations"]:
            print(f"  [{v['severity']}] {v['id']}: {v['evidence']}")
    else:
        print("所有约束检查通过。")
