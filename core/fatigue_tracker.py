"""
FatigueTracker - 追踪"用户是不是开始烦了"

和 rapport 对称但独立：用户可以信任 Agent 但仍被问烦。
fatigue ∈ [0, 1]，分三档行为影响（由 ConversationManager 读阈值）：
  >= 0.3  软缩短——问题更轻、选项更少、少追问
  >= 0.6  自动切到 profiling_mode = passive（停止主动出题）
  >= 0.8  切到 off + 道歉收尾

信号来源：
- 规则同步：连续提问、用户回复长度下降、敷衍词、显式负面、极短回复
- LLM 异步：更精细的"被烦到"语气判别
"""

import json
from typing import Optional

from core import AgentAction, UserState


FATIGUE_SIGNAL_PROMPT = """你是一个对话疲劳分析器。判断用户最新的回复里是否透出"被问烦了/不耐烦/想快点结束"的信号。

信号列表（可多选）：
疲劳信号（会让 fatigue 上升）:
- user_stalling: 敷衍回答、"随便/都行/你定"等回避
- user_shortening: 回复比之前显著变短
- user_annoyance: 显式不耐烦（"够了/烦/别问了/能不能直接"）
- user_skip_request: 想跳过/想直接开始/想结束
- user_terse_negative: 短回复 + 消极意味（"不想说"、"算了"）
复苏信号（会让 fatigue 回落）:
- user_recovering: 再次投入、追问、主动分享、笑/语气放松

上一轮 Agent 说了：{agent_message}
用户最新回复：{user_message}

输出 JSON，只输出，不要其他：
{{"signals": ["signal_name", ...]}}
"""


class FatigueTracker:
    """追踪用户被问烦的程度。"""

    SIGNAL_DELTAS = {
        "user_stalling": 0.08,
        "user_shortening": 0.06,
        "user_annoyance": 0.22,
        "user_skip_request": 0.25,
        "user_terse_negative": 0.12,
        "user_consecutive_questions": 0.05,   # Agent 端：连续被 Agent 追问
        "user_long_gap_between_engagement": 0.04,
        "user_recovering": -0.15,
    }

    DECAY_PER_TURN = 0.02  # 任何信号都没有时自然衰减

    SOFT_SHORTEN_THRESHOLD = 0.30
    AUTO_PASSIVE_THRESHOLD = 0.60
    AUTO_OFF_THRESHOLD = 0.80

    def __init__(self, initial_fatigue: float = 0.0):
        self.fatigue = initial_fatigue
        self._last_user_msg_len: Optional[int] = None
        self._consecutive_agent_questions: int = 0
        self.signal_log: list[dict] = []

    def update_from_signals(self, signals: list[str]):
        total = 0.0
        for sig in signals:
            delta = self.SIGNAL_DELTAS.get(sig, 0.0)
            total += delta
            self.signal_log.append({"signal": sig, "delta": delta})
        self.fatigue = max(0.0, min(1.0, self.fatigue + total))

    def apply_decay(self):
        self.fatigue = max(0.0, self.fatigue - self.DECAY_PER_TURN)

    def update_rule_based(
        self,
        user_message: str,
        user_state: UserState,
        agent_last_action: Optional[AgentAction],
        via_button: bool = False,
    ) -> list[str]:
        """
        via_button=True 表示这条用户消息来自前端"点了 A/B 按钮"
        而不是在输入框里打的字。按钮点击在 UX 上是 engaged 行为
        （只是因为选项现成所以回复短），不应被当成"敷衍/疲劳"。
        """
        msg = user_message.strip()
        msg_len = len(msg)
        signals: list[str] = []

        # 连续 Agent 提问计数
        # 按钮点击 = 主动答题，重置计数（不累积疲劳）
        if via_button:
            self._consecutive_agent_questions = 0
        elif agent_last_action in (AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT, AgentAction.OFFER_CHOICE):
            self._consecutive_agent_questions += 1
        else:
            self._consecutive_agent_questions = 0

        if self._consecutive_agent_questions >= 3:
            signals.append("user_consecutive_questions")

        # 显式不耐烦 / 跳过
        annoyance_tokens = ["别问了", "烦", "够了", "能不能直接", "先别", "不要再问"]
        if any(t in msg for t in annoyance_tokens):
            signals.append("user_annoyance")

        skip_tokens = ["跳过", "直接开始", "先这样", "算了", "不测了", "不想测", "不答了"]
        if any(t in msg for t in skip_tokens):
            signals.append("user_skip_request")

        # 敷衍
        stalling_tokens = ["随便", "都行", "都可以", "无所谓", "你定", "你说了算", "不知道", "没想法"]
        quiz_tokens = {"A", "B", "C", "D", "1", "2", "3", "4", "一", "二", "三", "四"}
        is_quiz_answer = msg.upper() in quiz_tokens
        if not is_quiz_answer and any(t in msg for t in stalling_tokens):
            signals.append("user_stalling")

        # 短化 & terse_negative：按钮点击不计
        if not via_button:
            if self._last_user_msg_len is not None and msg_len > 0:
                if msg_len <= 4 and self._last_user_msg_len >= 15 and not is_quiz_answer:
                    signals.append("user_shortening")
            if msg_len <= 4 and user_state in (UserState.GUARDED, UserState.DISENGAGING) and not is_quiz_answer:
                signals.append("user_terse_negative")

        # 复苏：文本主动语气 / 长回复 / 按钮点选都算
        recovering_tokens = ["哈哈", "有意思", "好玩", "继续", "然后呢", "还有吗", "😄", "😂"]
        if via_button or any(t in msg for t in recovering_tokens) or msg_len >= 30:
            signals.append("user_recovering")

        self._last_user_msg_len = msg_len
        self.update_from_signals(signals)
        if not signals:
            self.apply_decay()
        return signals

    async def update_llm_based(
        self,
        user_message: str,
        agent_message: str,
        llm_client,
    ) -> list[str]:
        prompt = FATIGUE_SIGNAL_PROMPT.format(
            agent_message=agent_message[:300] or "(无)",
            user_message=user_message[:300],
        )
        try:
            response = await llm_client.complete(prompt)
            result = json.loads(response)
            signals = result.get("signals", [])
            valid = [s for s in signals if s in self.SIGNAL_DELTAS]
            self.update_from_signals(valid)
            return valid
        except (json.JSONDecodeError, Exception):
            return []

    @property
    def level(self) -> str:
        if self.fatigue >= self.AUTO_OFF_THRESHOLD:
            return "critical"
        if self.fatigue >= self.AUTO_PASSIVE_THRESHOLD:
            return "high"
        if self.fatigue >= self.SOFT_SHORTEN_THRESHOLD:
            return "elevated"
        return "low"

    def should_shorten(self) -> bool:
        return self.fatigue >= self.SOFT_SHORTEN_THRESHOLD

    def should_auto_passive(self) -> bool:
        return self.fatigue >= self.AUTO_PASSIVE_THRESHOLD

    def should_auto_off(self) -> bool:
        return self.fatigue >= self.AUTO_OFF_THRESHOLD
