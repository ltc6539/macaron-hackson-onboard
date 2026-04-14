"""
ArchetypeMapper - 从 5 维向量映射到用户可见的人格类型

当 Agent 判断 confidence 足够（至少 3 个维度 > threshold），
就可以给用户一个 archetype 标签——一个有趣的、可分享的身份。
"""

from typing import Optional

import yaml
from pathlib import Path

from core import ProfileState


class ArchetypeMapper:
    """将画像向量映射为 archetype"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config" / "archetypes.yaml")
        self.config = self._load_config(config_path)

    def match(self, profile: ProfileState, behavior: Optional[dict] = None) -> dict:
        """
        找到最匹配的 archetype

        Args:
            profile: 5 维画像
            behavior: 可选行为快照，含 profiling_mode / turn_count /
                avg_user_msg_len / rapport。用于淡人等行为型档的判定。

        Returns:
            {
                "key": str,
                "name": str,
                "emoji": str,
                "description": str,
                "agent_promise": str,
                "match_score": float,
                "is_fallback": bool,
            }
        """
        # Step 0: 行为型档优先（在 5 维判定之前）
        behavioral = self._match_behavioral(behavior)
        if behavioral is not None:
            return behavioral

        matching = self.config.get("matching", {})
        min_dims = matching.get("min_confident_dimensions", 3)
        threshold = matching.get("confidence_threshold", 0.45)
        soft_threshold = 0.35  # 软匹配用更松的阈值

        # 硬匹配
        if profile.confident_dimensions_count(threshold) >= min_dims:
            hard = self._compose_from_candidates(profile, threshold)
            if hard is not None:
                return hard

        # 软匹配：≥2 维达到软阈值就尝试
        if profile.confident_dimensions_count(soft_threshold) >= 2:
            soft = self._compose_from_candidates(profile, soft_threshold, soft=True)
            if soft is not None:
                return soft

        return self._fallback_result()

    def _compose_from_candidates(
        self,
        profile: ProfileState,
        threshold: float,
        soft: bool = False,
    ) -> Optional[dict]:
        """按 threshold 跑打分，返回最佳 archetype dict，或 None。"""
        candidates = []
        for key, archetype in self.config.get("archetypes", {}).items():
            if "match_rules" not in archetype:
                continue
            score = self._compute_match_score(profile, archetype, threshold)
            if score > 0:
                candidates.append((key, archetype, score))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[2], reverse=True)
        key, archetype, score = candidates[0]
        return {
            "key": key,
            "name": archetype["name"],
            "emoji": archetype["emoji"],
            "description": archetype["description"].strip(),
            "agent_promise": archetype["agent_promise"].strip(),
            "match_score": score,
            "is_fallback": False,
            "soft_match": soft,
        }

    def _match_behavioral(self, behavior: Optional[dict] = None) -> Optional[dict]:
        """
        根据行为快照判断是否命中行为型 archetype（目前仅淡人）。
        命中条件（low_engagement_skipper）:
          - profiling_mode != "active"（用户主动跳过或被 fatigue 降级）
          - turn_count <= 6（短会话内）
          - avg_user_msg_len <= 4（全程回复极短）
          - rapport < 0.35（关系没暖起来）
        """
        if not behavior:
            return None

        if behavior.get("profiling_mode") == "active":
            return None
        if behavior.get("turn_count", 99) > 6:
            return None
        if behavior.get("avg_user_msg_len", 99) > 4:
            return None
        if behavior.get("rapport", 1.0) >= 0.35:
            return None

        dan_ren = self.config.get("archetypes", {}).get("dan_ren")
        if not dan_ren:
            return None

        return {
            "key": "dan_ren",
            "name": dan_ren["name"],
            "emoji": dan_ren["emoji"],
            "description": dan_ren["description"].strip(),
            "agent_promise": dan_ren["agent_promise"].strip(),
            "match_score": 1.0,
            "is_fallback": False,
            "soft_match": False,
        }

    def _compute_match_score(
        self,
        profile: ProfileState,
        archetype: dict,
        confidence_threshold: float,
    ) -> float:
        """计算 profile 与 archetype 的匹配分"""
        rules = archetype.get("match_rules", {})
        score = 0.0
        total_rules = len(rules)

        if total_rules == 0:
            return 0.0

        for dim_name, condition in rules.items():
            dim = profile.get_dimension(dim_name)

            # 如果这个维度 confidence 不够，跳过（不扣分但也不加分）
            if dim.confidence < confidence_threshold:
                total_rules -= 1
                continue

            # 解析条件
            if condition == ">0" and dim.value > 0:
                score += 1.0 + dim.confidence  # confidence 越高匹配越强
            elif condition == "<0" and dim.value < 0:
                score += 1.0 + dim.confidence
            else:
                # 不匹配，扣分
                score -= 0.5

        # 归一化
        return score / max(total_rules, 1)

    def _fallback_result(self) -> dict:
        """返回 fallback archetype"""
        fallback = self.config.get("fallback", {})
        return {
            "key": "fallback",
            "name": fallback.get("name", "探索中"),
            "emoji": fallback.get("emoji", "🧭"),
            "description": fallback.get("description", "我还在了解你...").strip(),
            "agent_promise": fallback.get("agent_promise", "我会慢慢了解你的偏好。").strip(),
            "match_score": 0.0,
            "is_fallback": True,
            "soft_match": False,
        }

    def get_all_archetypes(self) -> list[dict]:
        """返回所有 archetype（用于展示）"""
        result = []
        for key, arch in self.config.get("archetypes", {}).items():
            result.append({
                "key": key,
                "name": arch["name"],
                "emoji": arch["emoji"],
                "description": arch["description"].strip(),
            })
        return result

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {"archetypes": {}, "fallback": {}}
