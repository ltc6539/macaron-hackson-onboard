"""
SignalExtractor - 从自然对话中提取画像维度信号

这是"最好的问题不像问题"的核心实现。
Agent 在和用户聊天、执行任务的过程中，
这个模块在后台持续分析对话内容，提取隐式的画像信号。
"""

import json


SIGNAL_EXTRACTION_PROMPT = """你是一个用户偏好分析器。从用户的自然对话中提取行为倾向信号。

我们追踪 5 个维度（每个维度 -1.0 到 +1.0）：

1. novelty_appetite（新鲜感渴求度）
   -1.0 = 极度偏好熟悉的事物、固定路线、常去的店
   +1.0 = 极度追求新鲜、未知、冒险、没尝试过的

2. decision_tempo（决策节奏）
   -1.0 = 深思熟虑型，收集信息、对比分析后再决定
   +1.0 = 直觉快决策，跟着感觉走、不纠结

3. social_energy（社交能量）
   -1.0 = 独处型，享受一个人的活动、安静的空间
   +1.0 = 社交型，喜欢热闹、约人、组局

4. sensory_cerebral（感性-理性光谱）
   -1.0 = 理性型，看数据、评分、价格，分析导向
   +1.0 = 感性型，看氛围、感觉、视觉、体验导向

5. control_flow（掌控感偏好）
   -1.0 = 随遇而安，喜欢被安排、接受惊喜、"你来定"
   +1.0 = 掌控型，喜欢自己规划、选择、掌握全局

对话上下文：
{conversation_context}

用户最新消息：
{user_message}

Agent 上一条消息（如有）：
{agent_message}

请分析用户消息中隐含的维度信号。
规则：
- 只提取有实际信号的维度，没有信号就不要输出
- confidence 反映信号的强度（0.05-0.30，对话推断通常比直接问答低）
- 不要过度解读，宁可遗漏也不要误判
- "随便"/"都行"如果是回避性的，不提取信号；如果是真的随意，提取 control_flow 信号

输出 JSON（只输出 JSON，不要其他内容）：
{{
  "signals": {{
    "dimension_name": {{ "value": 0.0, "confidence": 0.0, "evidence": "引用的原文片段" }}
  }},
  "has_task_intent": true/false,
  "task_description": "如果有任务意图，描述是什么"
}}

如果没有任何有意义的信号，输出：
{{"signals": {{}}, "has_task_intent": false, "task_description": ""}}
"""


VALID_DIMENSIONS = {
    "novelty_appetite", "decision_tempo", "social_energy",
    "sensory_cerebral", "control_flow",
}


class SignalExtractor:
    """从对话中提取画像维度信号"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def extract(
        self,
        user_message: str,
        agent_message: str = "",
        conversation_context: str = "",
    ) -> dict:
        """
        从用户消息中提取画像信号

        Returns:
            {
                "signals": { dim_name: { "value": float, "confidence": float } },
                "has_task_intent": bool,
                "task_description": str,
            }
        """
        # 先用规则引擎做快速提取
        rule_signals = self._rule_based_extract(user_message, agent_message)

        # 如果有 LLM client，做更深入的分析
        if self.llm_client:
            llm_result = await self._llm_extract(
                user_message, agent_message, conversation_context
            )
            # 合并：LLM 信号覆盖规则信号（LLM 更精细）
            merged_signals = {**rule_signals, **llm_result.get("signals", {})}
            return {
                "signals": merged_signals,
                "has_task_intent": llm_result.get("has_task_intent", False),
                "task_description": llm_result.get("task_description", ""),
            }

        has_task_intent = self._detect_task_intent(user_message)
        return {
            "signals": rule_signals,
            "has_task_intent": has_task_intent,
            "task_description": self._infer_task_description(user_message) if has_task_intent else "",
        }

    def _rule_based_extract(self, user_message: str, agent_message: str) -> dict:
        """规则引擎：快速提取高确信信号"""
        msg = user_message.strip()
        signals = {}

        # ---------- Novelty 信号 ----------
        novelty_positive = ["新开的", "没吃过", "试试", "换个", "新鲜", "没去过", "尝尝"]
        novelty_negative = ["老地方", "上次那家", "还是去", "老样子", "常去的", "还是吃"]

        if any(w in msg for w in novelty_positive):
            signals["novelty_appetite"] = {"value": 0.4, "confidence": 0.20}
        elif any(w in msg for w in novelty_negative):
            signals["novelty_appetite"] = {"value": -0.4, "confidence": 0.20}

        # ---------- Decision Tempo 信号 ----------
        deliberate_indicators = ["对比一下", "哪个好", "评分", "看看评价", "多找几个", "有什么区别"]
        impulsive_indicators = ["就这个", "直接", "不纠结", "随便哪个", "第一个就行"]

        if any(w in msg for w in deliberate_indicators):
            signals["decision_tempo"] = {"value": -0.3, "confidence": 0.20}
        elif any(w in msg for w in impulsive_indicators):
            signals["decision_tempo"] = {"value": 0.3, "confidence": 0.20}

        # ---------- Social Energy 信号 ----------
        social_indicators = ["聚餐", "朋友", "约", "一起", "几个人", "包间", "大桌"]
        solo_indicators = ["一个人", "自己", "安静", "独处", "不想约人"]

        if any(w in msg for w in social_indicators):
            signals["social_energy"] = {"value": 0.3, "confidence": 0.15}
        elif any(w in msg for w in solo_indicators):
            signals["social_energy"] = {"value": -0.3, "confidence": 0.15}

        # ---------- Sensory vs Cerebral 信号 ----------
        # 用户要求看图 vs 看数据
        sensory_indicators = ["看看图", "有照片吗", "环境怎么样", "氛围", "感觉", "好看"]
        cerebral_indicators = ["评分多少", "人均多少", "距离", "排名", "对比", "性价比"]

        if any(w in msg for w in sensory_indicators):
            signals["sensory_cerebral"] = {"value": 0.3, "confidence": 0.20}
        elif any(w in msg for w in cerebral_indicators):
            signals["sensory_cerebral"] = {"value": -0.3, "confidence": 0.20}

        # ---------- Control vs Flow 信号 ----------
        control_indicators = ["我自己选", "给我列个表", "我来决定", "我看看", "让我想想"]
        flow_indicators = ["你帮我选", "你决定", "你推荐", "直接帮我订", "你来"]

        if any(w in msg for w in control_indicators):
            signals["control_flow"] = {"value": 0.4, "confidence": 0.20}
        elif any(w in msg for w in flow_indicators):
            signals["control_flow"] = {"value": -0.4, "confidence": 0.20}

        return signals

    async def _llm_extract(
        self,
        user_message: str,
        agent_message: str,
        conversation_context: str,
    ) -> dict:
        """LLM 深度分析"""
        prompt = SIGNAL_EXTRACTION_PROMPT.format(
            conversation_context=conversation_context or "(无上下文)",
            user_message=user_message,
            agent_message=agent_message or "(无)",
        )

        try:
            response = await self.llm_client.complete(prompt)
            result = json.loads(response)

            # 验证和清理信号
            cleaned_signals = {}
            for dim, signal in result.get("signals", {}).items():
                if dim in VALID_DIMENSIONS:
                    value = max(-1.0, min(1.0, float(signal.get("value", 0))))
                    confidence = max(0.0, min(0.3, float(signal.get("confidence", 0))))
                    if abs(value) > 0.05 and confidence > 0.03:
                        cleaned_signals[dim] = {
                            "value": value,
                            "confidence": confidence,
                        }

            return {
                "signals": cleaned_signals,
                "has_task_intent": result.get("has_task_intent", False),
                "task_description": result.get("task_description", ""),
            }
        except (json.JSONDecodeError, Exception):
            return {"signals": {}, "has_task_intent": False, "task_description": ""}

    def _detect_task_intent(self, message: str) -> bool:
        """简单规则判断用户是否有任务意图"""
        task_phrases = [
            "帮我", "我想", "我要", "找一个", "推荐", "订", "搜",
            "有没有", "哪里有", "去哪", "吃什么", "做什么",
        ]
        return any(phrase in message for phrase in task_phrases)

    def _infer_task_description(self, message: str) -> str:
        """无 LLM 模式下，尽量保留用户原始任务描述"""
        text = message.strip()
        prefixes = ["帮我", "我想", "我要"]
        for prefix in prefixes:
            if text.startswith(prefix):
                return text[len(prefix):].strip() or text
        return text
