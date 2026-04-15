"""
Core module - 数据模型和共享类型
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class UserState(Enum):
    """用户当前的交互状态"""
    ENGAGED_PLAYFUL = "engaged_playful"
    TASK_ORIENTED = "task_oriented"
    TENTATIVE_EXPLORING = "tentative_exploring"
    GUARDED = "guarded"
    DISENGAGING = "disengaging"
    UNKNOWN = "unknown"


class AgentAction(Enum):
    """Agent 的行动类型"""
    ASK_PLAYFUL = "ask_playful"
    ASK_DIRECT = "ask_direct"
    OFFER_CHOICE = "offer_choice"
    OBSERVE_REACTION = "observe_reaction"
    SELF_DISCLOSE = "self_disclose"
    GIVE_VALUE = "give_value"
    EVALUATE_USER = "evaluate_user"
    DO_NOTHING = "do_nothing"
    SHOW_ARCHETYPE = "show_archetype"


class SignalSource(Enum):
    """信号来源类型"""
    QUIZ = "quiz"
    DIRECT_PREFERENCE = "direct_preference"
    CHOICE_INFERENCE = "choice_inference"
    REACTION_OBSERVATION = "reaction_observation"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class ProfilingMode(Enum):
    """画像采集模式（单向降级：active → passive → off）"""
    ACTIVE = "active"      # 主动出题 + 对话推断
    PASSIVE = "passive"    # 不主动出题，但继续从对话推断
    OFF = "off"            # 完全关闭采集


@dataclass
class DimensionState:
    """单个维度的状态"""
    value: float = 0.0        # -1.0 ~ +1.0
    confidence: float = 0.0    # 0.0 ~ 1.0
    signal_count: int = 0      # 收到过多少个信号

    @property
    def is_active(self) -> bool:
        return self.confidence >= 0.55

    @property
    def is_tentative(self) -> bool:
        return 0.30 <= self.confidence < 0.55

    @property
    def pole(self) -> Optional[str]:
        """返回当前偏向的极，None 表示信号不足"""
        if self.confidence < 0.30:
            return None
        return "positive" if self.value > 0 else "negative"


@dataclass
class ProfileState:
    """完整的 5 维用户画像"""
    novelty_appetite: DimensionState = field(default_factory=DimensionState)
    decision_tempo: DimensionState = field(default_factory=DimensionState)
    social_energy: DimensionState = field(default_factory=DimensionState)
    sensory_cerebral: DimensionState = field(default_factory=DimensionState)
    control_flow: DimensionState = field(default_factory=DimensionState)

    def get_dimension(self, name: str) -> DimensionState:
        return getattr(self, name)

    def mean_confidence(self) -> float:
        dims = [self.novelty_appetite, self.decision_tempo, self.social_energy,
                self.sensory_cerebral, self.control_flow]
        return sum(d.confidence for d in dims) / len(dims)

    def needs_more_data(self) -> bool:
        """画像是否还需要更多数据"""
        return self.mean_confidence() < 0.50

    def confident_dimensions_count(self, threshold: float = 0.45) -> int:
        dims = [self.novelty_appetite, self.decision_tempo, self.social_energy,
                self.sensory_cerebral, self.control_flow]
        return sum(1 for d in dims if d.confidence >= threshold)

    def least_confident_dimensions(self, n: int = 2) -> list[str]:
        """返回 confidence 最低的 n 个维度名"""
        dims = {
            "novelty_appetite": self.novelty_appetite.confidence,
            "decision_tempo": self.decision_tempo.confidence,
            "social_energy": self.social_energy.confidence,
            "sensory_cerebral": self.sensory_cerebral.confidence,
            "control_flow": self.control_flow.confidence,
        }
        sorted_dims = sorted(dims.items(), key=lambda x: x[1])
        return [name for name, _ in sorted_dims[:n]]

    def to_dict(self) -> dict:
        return {
            name: {"value": round(getattr(self, name).value, 3),
                   "confidence": round(getattr(self, name).confidence, 3)}
            for name in ["novelty_appetite", "decision_tempo", "social_energy",
                         "sensory_cerebral", "control_flow"]
        }


@dataclass
class ConversationTurn:
    """一轮对话"""
    role: str            # "user" or "agent"
    content: str
    action_type: Optional[AgentAction] = None
    signals_extracted: Optional[dict] = None
    user_state_at_turn: Optional[UserState] = None


@dataclass
class AgentState:
    """Agent 的完整内部状态"""
    profile: ProfileState = field(default_factory=ProfileState)
    user_state: UserState = UserState.UNKNOWN
    rapport: float = 0.25
    fatigue: float = 0.0
    turn_count: int = 0
    questions_asked: int = 0
    last_action: Optional[AgentAction] = None
    asked_question_ids: list = field(default_factory=list)
    asked_choice_keys: set = field(default_factory=set)
    asked_fallback_keys: set = field(default_factory=set)
    meh_count: int = 0
    last_fatigue_level: Optional[str] = None
    last_user_state_for_meta: Optional[str] = None
    guarded_streak: int = 0
    meta_transitions_fired: set = field(default_factory=set)
    turns_since_last_evaluate: int = 99
    last_meta_prefix: Optional[str] = None
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    onboarding_complete: bool = False
    onboarding_session_active: bool = False
    onboarding_questions_answered: int = 0
    onboarding_questions_target: int = 4
    archetype_revealed: bool = False
    active_task: Optional[str] = None
    profiling_mode: ProfilingMode = ProfilingMode.ACTIVE

    def add_turn(self, turn: ConversationTurn):
        self.conversation_history.append(turn)
        if turn.role == "user":
            self.turn_count += 1
