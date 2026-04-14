"""
ProfileAccumulator - 用户画像的 Bayesian 更新引擎

核心逻辑：
- 每个维度维护 (value, confidence) 对
- 新信号通过加权融合更新现有估计
- confidence 单调递增但有上限（单信号不会让 confidence 爆表）
- 支持多信号源，不同来源有不同的可信度权重
"""

from core import ProfileState, DimensionState, SignalSource


class ProfileAccumulator:
    """管理 5 维画像的持续更新"""

    # 不同信号源的基础置信度系数
    SOURCE_CONFIDENCE_MULTIPLIER = {
        SignalSource.QUIZ: 1.0,
        SignalSource.DIRECT_PREFERENCE: 1.2,
        SignalSource.CHOICE_INFERENCE: 0.8,
        SignalSource.REACTION_OBSERVATION: 0.5,
        SignalSource.BEHAVIORAL_PATTERN: 0.3,
    }

    def __init__(self, profile: ProfileState = None):
        self.profile = profile or ProfileState()

    def update(
        self,
        dimension: str,
        value: float,
        confidence: float,
        source: SignalSource = SignalSource.QUIZ,
    ):
        """
        Bayesian 风格更新一个维度

        Args:
            dimension: 维度名 (e.g. "novelty_appetite")
            value: 信号值 (-1.0 ~ +1.0)
            confidence: 信号的置信度 (0.0 ~ 1.0)
            source: 信号来源
        """
        dim = self.profile.get_dimension(dimension)

        # 应用来源权重
        multiplier = self.SOURCE_CONFIDENCE_MULTIPLIER.get(source, 0.5)
        adjusted_confidence = confidence * multiplier

        # Bayesian 加权更新
        w_old = dim.confidence
        w_new = adjusted_confidence

        if w_old + w_new == 0:
            return

        # 值的加权平均
        new_value = (dim.value * w_old + value * w_new) / (w_old + w_new)

        # confidence 递增但递减收益（越来越难提升）
        new_confidence = w_old + w_new * (1.0 - w_old)

        # Clamp
        dim.value = max(-1.0, min(1.0, new_value))
        dim.confidence = min(1.0, new_confidence)
        dim.signal_count += 1

    def update_from_quiz(self, signals: dict):
        """
        从一道 quiz 题的答案更新多个维度

        Args:
            signals: { dimension_name: { "value": float, "confidence": float } }
        """
        for dim_name, signal in signals.items():
            self.update(
                dimension=dim_name,
                value=signal["value"],
                confidence=signal["confidence"],
                source=SignalSource.QUIZ,
            )

    def update_from_conversation(self, signals: dict):
        """
        从对话推断更新（LLM 提取的信号）

        Args:
            signals: { dimension_name: { "value": float, "confidence": float } }
        """
        for dim_name, signal in signals.items():
            self.update(
                dimension=dim_name,
                value=signal["value"],
                confidence=signal["confidence"],
                source=SignalSource.CHOICE_INFERENCE,
            )

    def needs_more_data(self) -> bool:
        """画像是否还需要更多数据"""
        return self.profile.mean_confidence() < 0.50

    def get_biggest_gaps(self, n: int = 2) -> list[str]:
        """返回最需要补充数据的维度"""
        return self.profile.least_confident_dimensions(n)

    def get_snapshot(self) -> dict:
        """导出当前画像快照"""
        return {
            "dimensions": self.profile.to_dict(),
            "mean_confidence": round(self.profile.mean_confidence(), 3),
            "confident_count": self.profile.confident_dimensions_count(),
            "needs_more_data": self.needs_more_data(),
        }
