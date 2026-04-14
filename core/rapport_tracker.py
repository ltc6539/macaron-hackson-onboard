"""
RapportTracker - 追踪 Agent 与用户之间的信任关系

信任不是一个感性概念，而是一个可量化的状态：
- 正向信号（用户投入、分享、回应）增加信任
- 负向信号（用户回避、敷衍、拒绝）降低信任
- 每轮有微小的自然衰减（用户的耐心是有限的）

Agent 的行为选择直接依赖于 rapport 水平。
"""

import json
from typing import Optional

from core import UserState, AgentAction


# 用于 LLM 调用的 prompt
RAPPORT_SIGNAL_PROMPT = """你是一个对话关系分析器。根据用户的最新回复，判断以下哪些信号出现了。

信号列表（可多选）：
正向信号:
- user_answers_open_question: 用户认真回答了一个开放问题
- user_asks_followup: 用户主动追问
- user_shares_personal_info: 用户分享了个人信息/偏好/经历
- user_uses_emoji_or_humor: 用户使用了 emoji 或幽默
- user_accepts_recommendation: 用户接受了推荐/建议
- user_says_thanks: 用户表达了感谢
- user_long_response: 用户给了较长的回复（>30字）

负向信号:
- user_ignores_question: 用户无视了 Agent 的问题
- user_gives_minimal_response: 用户给了极简回复（<5字）
- user_rejects_recommendation: 用户拒绝了推荐
- user_expresses_annoyance: 用户表达了不耐烦
- user_says_skip_or_stop: 用户要求跳过/停止

上一轮 Agent 说了: {agent_message}
用户回复: {user_message}

输出 JSON（只输出信号名列表，不要其他内容）：
{{"signals": ["signal_name_1", "signal_name_2"]}}
"""


class RapportTracker:
    """追踪和更新用户信任度"""

    # 各信号对应的 rapport 变化量
    SIGNAL_DELTAS = {
        # 正向
        "user_answers_open_question": 0.08,
        "user_asks_followup": 0.10,
        "user_shares_personal_info": 0.12,
        "user_uses_emoji_or_humor": 0.05,
        "user_accepts_recommendation": 0.06,
        "user_says_thanks": 0.04,
        "user_long_response": 0.05,
        # 负向
        "user_ignores_question": -0.10,
        "user_gives_minimal_response": -0.05,
        "user_rejects_recommendation": -0.03,
        "user_expresses_annoyance": -0.15,
        "user_says_skip_or_stop": -0.20,
    }

    DECAY_PER_TURN = 0.01

    def __init__(self, initial_rapport: float = 0.25):
        self.rapport = initial_rapport
        self.signal_log: list[dict] = []

    def update_from_signals(self, signals: list[str]):
        """
        根据检测到的信号更新 rapport

        Args:
            signals: 信号名列表
        """
        total_delta = 0.0
        for signal in signals:
            delta = self.SIGNAL_DELTAS.get(signal, 0.0)
            total_delta += delta
            self.signal_log.append({"signal": signal, "delta": delta})

        self.rapport = max(0.0, min(1.0, self.rapport + total_delta))

    def apply_decay(self):
        """每轮自然衰减"""
        self.rapport = max(0.0, self.rapport - self.DECAY_PER_TURN)

    def update_rule_based(
        self,
        user_message: str,
        user_state: UserState,
        agent_last_action: Optional[AgentAction],
    ):
        """
        基于规则的快速 rapport 更新（不需要 LLM）

        这是 LLM 信号检测的补充/兜底
        """
        msg = user_message.strip()
        signals = []
        quiz_tokens = {
            "A", "B", "C", "D",
            "1", "2", "3", "4",
            "一", "二", "三", "四",
            "第一", "第二", "第三", "第四",
        }
        is_quiz_answer = (
            agent_last_action in (AgentAction.ASK_PLAYFUL, AgentAction.OFFER_CHOICE)
            and msg.upper() in quiz_tokens
        )

        # 消息长度信号
        if len(msg) > 30:
            signals.append("user_long_response")
        elif len(msg) <= 4 and not is_quiz_answer:
            signals.append("user_gives_minimal_response")

        # Emoji / 语气词
        humor_indicators = ["哈哈", "😄", "😂", "🤣", "有意思", "好玩", "笑死"]
        if any(ind in msg for ind in humor_indicators):
            signals.append("user_uses_emoji_or_humor")

        # 感谢
        if any(w in msg for w in ["谢谢", "感谢", "thanks", "thx", "谢啦"]):
            signals.append("user_says_thanks")

        # 追问
        if any(w in msg for w in ["还有吗", "然后呢", "继续", "多说说", "展开讲讲"]):
            signals.append("user_asks_followup")

        # 跳过/停止
        if any(w in msg for w in ["跳过", "不用了", "算了", "停", "不想"]):
            signals.append("user_says_skip_or_stop")

        # 主动接受互动邀请（如测试）→ 非常强的正向信号
        quiz_accept = ["好啊", "来吧", "做题", "冲", "来",
                        "好呀", "走起", "好的呀", "好嘞", "整",
                        "测试", "测一下", "做个测试", "试试看"]
        is_task = any(msg.startswith(p) for p in ["帮我", "我想", "我要", "找", "搜"])
        if any(phrase in msg for phrase in quiz_accept) and not is_task:
            signals.append("user_answers_open_question")
            signals.append("user_shares_personal_info")  # 愿意参与 ≈ 愿意暴露

        # 不耐烦
        if any(w in msg for w in ["别问了", "烦", "够了", "能不能直接"]):
            signals.append("user_expresses_annoyance")

        # 如果上一轮 Agent 问了问题，检查用户是否回答了
        if agent_last_action in (AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT):
            if is_quiz_answer:
                signals.append("user_answers_open_question")
            elif len(msg) > 10 and user_state != UserState.GUARDED:
                signals.append("user_answers_open_question")
            elif user_state == UserState.GUARDED:
                signals.append("user_ignores_question")

        self.update_from_signals(signals)
        self.apply_decay()

        return signals

    async def update_llm_based(self, user_message: str, agent_message: str, llm_client):
        """
        用 LLM 做更精细的信号检测
        """
        prompt = RAPPORT_SIGNAL_PROMPT.format(
            agent_message=agent_message[:300],
            user_message=user_message[:300],
        )

        try:
            response = await llm_client.complete(prompt)
            result = json.loads(response)
            signals = result.get("signals", [])
            # 过滤无效信号
            valid_signals = [s for s in signals if s in self.SIGNAL_DELTAS]
            self.update_from_signals(valid_signals)
            self.apply_decay()
            return valid_signals
        except (json.JSONDecodeError, Exception):
            return []

    @property
    def level(self) -> str:
        """返回 rapport 等级（用于策略查询）"""
        if self.rapport < 0.25:
            return "low"
        elif self.rapport < 0.6:
            return "medium"
        else:
            return "high"
