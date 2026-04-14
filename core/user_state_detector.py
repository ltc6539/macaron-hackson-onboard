"""
UserStateDetector - 实时推断用户的交互状态

不是情感分析，而是基于对话行为特征的状态推断。
Agent 的所有决策都依赖于这个状态判断。

核心思路：用 LLM 做精细判断，用规则做兜底和校验。
"""

import json
from typing import Optional

from core import UserState, ConversationTurn


# 用于 LLM 调用的 prompt
USER_STATE_DETECTION_PROMPT = """你是一个对话状态分析器。根据用户最近的消息，判断用户当前处于以下哪种状态：

1. engaged_playful - 用户投入且愉快，愿意互动
   信号：回复较长、有 emoji/语气词、主动追问、表达好奇、开玩笑
   
2. task_oriented - 用户有明确目的，想高效完成
   信号：开头就说需求、回复简洁直接、忽略闲聊、信息密度高
   
3. tentative_exploring - 用户在试探了解产品
   信号：问"你能做什么"、试探性小需求、不投入太多个人信息
   
4. guarded - 用户有戒备心或不太想聊
   信号：回复极短（"嗯"/"好"/"行"）、不回答开放问题、模糊回答（"都行"/"随便"）
   
5. disengaging - 用户在失去兴趣准备离开
   信号：回复越来越短、出现终结性回复（"好的知道了"/"嗯嗯"）、不再追问

对话上下文（最近几轮）：
{conversation_context}

用户最新消息：
{latest_message}

请输出 JSON，格式如下（不要输出其他内容）：
{{"state": "状态名", "confidence": 0.0-1.0, "reasoning": "简短理由"}}
"""


class UserStateDetector:
    """推断用户当前的交互状态"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._state_history: list[UserState] = []

    async def detect(
        self,
        latest_message: str,
        conversation_history: list[ConversationTurn],
    ) -> UserState:
        """
        综合 LLM 判断和规则兜底来推断用户状态

        Args:
            latest_message: 用户最新的消息
            conversation_history: 对话历史

        Returns:
            推断的用户状态
        """
        # Layer 1: 规则快速判断（覆盖高确信场景）
        rule_state = self._rule_based_detection(latest_message, conversation_history)
        if rule_state is not None:
            self._state_history.append(rule_state)
            return rule_state

        # Layer 2: LLM 精细判断
        if self.llm_client:
            llm_state = await self._llm_detection(latest_message, conversation_history)
            if llm_state is not None:
                self._state_history.append(llm_state)
                return llm_state

        # Layer 3: 基于历史趋势的兜底
        fallback = self._trend_fallback()
        self._state_history.append(fallback)
        return fallback

    def _rule_based_detection(
        self,
        message: str,
        history: list[ConversationTurn],
    ) -> Optional[UserState]:
        """规则引擎：处理高确信的简单场景"""

        msg = message.strip()
        msg_len = len(msg)

        # ---------- Disengaging 信号 ----------
        terminal_phrases = ["好的", "知道了", "嗯嗯", "ok", "行吧", "好吧", "再见", "拜拜", "bye"]
        if msg_len < 10 and any(msg.lower().startswith(p) for p in terminal_phrases):
            # 检查趋势：如果最近几轮也很短，大概率在 disengage
            if self._recent_messages_short(history, n=2):
                return UserState.DISENGAGING

        # ---------- Guarded 信号 ----------
        guarded_phrases = ["都行", "随便", "都可以", "无所谓", "你说了算", "不知道", "没想法"]
        # 简短选项回复可能是在回答 quiz，不算 guarded
        quiz_tokens = {
            "A", "B", "C", "D",
            "1", "2", "3", "4",
            "一", "二", "三", "四",
        }
        is_quiz_answer = msg_len <= 2 and msg.upper() in quiz_tokens
        if not is_quiz_answer:
            if msg in guarded_phrases or (msg_len <= 4 and msg in ["嗯", "好", "行", "哦", "ok"]):
                return UserState.GUARDED

        # ---------- Task-oriented 信号 ----------
        task_indicators = ["帮我", "我想", "我要", "找一个", "推荐", "订", "搜", "查"]
        if any(msg.startswith(ind) for ind in task_indicators):
            return UserState.TASK_ORIENTED

        # ---------- Tentative 信号 ----------
        exploring_phrases = ["你能做什么", "你会什么", "比如呢", "怎么用", "有什么功能", "你是谁"]
        if any(phrase in msg for phrase in exploring_phrases):
            return UserState.TENTATIVE_EXPLORING

        # ---------- Engaged 信号 ----------
        # 用户主动接受测试/互动邀请 → 强 engaged 信号
        # 用户主动接受测试/互动邀请 → 强 engaged 信号
        # 注意：这些词必须足够特定，不能误匹配任务意图
        quiz_acceptance = ["好啊", "来吧", "做题", "冲", "来",
                           "好呀", "走起", "好的呀", "好嘞", "整",
                           "测试", "测一下", "做个测试", "试试看"]
        if any(phrase in msg for phrase in quiz_acceptance) and msg_len < 20:
            # 排除任务意图（"帮我试试XXX"不是接受测试）
            task_prefixes = ["帮我", "我想", "我要", "找", "搜"]
            if not any(msg.startswith(p) for p in task_prefixes):
                return UserState.ENGAGED_PLAYFUL

        engagement_indicators = [
            msg_len > 30,
            any(c in msg for c in ["😄", "😂", "🤣", "❤️", "👍", "哈哈", "有意思", "好玩", "！"]),
            msg.endswith("？") or msg.endswith("?"),  # 主动追问
            "还有" in msg or "然后呢" in msg or "继续" in msg,
        ]
        if sum(engagement_indicators) >= 1:
            return UserState.ENGAGED_PLAYFUL

        # 没有明确信号，交给 LLM
        return None

    async def _llm_detection(
        self,
        message: str,
        history: list[ConversationTurn],
    ) -> Optional[UserState]:
        """LLM 精细判断"""
        # 构建上下文（最近 4 轮）
        recent = history[-4:] if len(history) > 4 else history
        context_lines = []
        for turn in recent:
            role = "用户" if turn.role == "user" else "Agent"
            context_lines.append(f"{role}: {turn.content[:200]}")
        context = "\n".join(context_lines) if context_lines else "(这是第一轮对话)"

        prompt = USER_STATE_DETECTION_PROMPT.format(
            conversation_context=context,
            latest_message=message,
        )

        try:
            response = await self.llm_client.complete(prompt)
            result = json.loads(response)
            state_name = result.get("state", "unknown")
            return UserState(state_name)
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    def _trend_fallback(self) -> UserState:
        """基于历史趋势的兜底判断"""
        if not self._state_history:
            return UserState.UNKNOWN
        # 返回最近一次已知状态
        for state in reversed(self._state_history):
            if state != UserState.UNKNOWN:
                return state
        return UserState.UNKNOWN

    def _recent_messages_short(
        self, history: list[ConversationTurn], n: int = 2, threshold: int = 10
    ) -> bool:
        """检查最近 n 条用户消息是否都很短"""
        user_messages = [t for t in history if t.role == "user"]
        recent = user_messages[-n:] if len(user_messages) >= n else user_messages
        if not recent:
            return False
        return all(len(m.content.strip()) < threshold for m in recent)
