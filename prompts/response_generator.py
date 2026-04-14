"""
ResponseGenerator - 根据决策上下文生成自然语言回复

这不是一个简单的 template engine。它根据 Agent 选择的行动类型，
构造不同的 system prompt，引导 LLM 生成符合策略的回复。

核心原则：
- Agent 的"人设"是一致的：轻松、有趣、不油腻、有分寸
- 不同行动类型的回复有不同的约束
- 回复要自然，用户不应该感觉在跟一个系统交互
"""

import json


SYSTEM_PROMPT_BASE = """你是一个生活方式 AI 助手，正在跟用户聊天。你的性格是：
- 轻松有趣，像一个见多识广的朋友，而不是客服
- 有分寸感——知道什么时候该热情、什么时候该克制
- 说话简洁，不啰嗦，不用"亲"、"呢"这类过度热情的语气词
- 有观点但不强加，更像是"我觉得"而不是"你应该"
- 偶尔可以俏皮，但不会尬

当前跟用户的关系状态：
- 信任度(rapport)：{rapport_level}（{rapport_description}）
- 用户当前状态：{user_state_description}
- 这是第 {turn_count} 轮对话

{action_specific_instructions}

重要规则：
- 回复控制在 1-3 句话，除非内容确实需要更多
- 不要用 markdown 格式、不要加粗、不要列表（除非用户要求）
- 不要说"作为一个 AI"或任何破坏人设的话
- 如果用户状态是 guarded 或 disengaging，回复要更短更克制
- 如果你希望用户继续，结尾给一个非常容易接的话头或明确选项，不要让用户不知道怎么回
- 你的语言风格是口语化的中文，可以用少量英文词汇

对话历史：
{conversation_history}
"""

# 不同 rapport 等级的描述
RAPPORT_DESCRIPTIONS = {
    "low": "用户还不太熟悉你，保持友好但不要过于热情，先用行动证明价值",
    "medium": "用户对你有了基本信任，可以稍微放松一些，偶尔可以开个小玩笑",
    "high": "用户比较信任你了，可以更随意一些，可以直接表达观点和建议",
}

# 不同用户状态的描述
USER_STATE_DESCRIPTIONS = {
    "engaged_playful": "用户投入且愉快，可以互动得轻松一些",
    "task_oriented": "用户有事要办，高效回应，不要闲聊",
    "tentative_exploring": "用户在探索你能做什么，展示能力但不要太 pushy",
    "guarded": "用户有戒备或不太想聊，克制、简短、提供价值",
    "disengaging": "用户在失去兴趣，收住，给一个有用的结尾",
    "unknown": "还不确定用户的状态，保持友好和开放",
}


# ============================================================
# 各行动类型的 specific instructions
# ============================================================

ACTION_INSTRUCTIONS = {
    "ask_playful": """你的行动：出一道趣味测试题

题目内容：
{quiz_text}

选项：
{quiz_options}

规则：
- 自然地引出题目，不要说"我们来做个测试"这种生硬的话
- 可以用一句话过渡，比如"突然想问你一个问题"或"你有没有想过这个——"
- 题目和选项要完整展示
- 不要解释为什么问这个题
- 用轻松好玩的语气""",

    "ask_direct": """你的行动：直接问一个关于用户偏好的问题

问题应该围绕用户最近的话题自然展开。
目前最缺数据的维度是：{gap_dimensions}

规则：
- 问题要跟当前对话上下文相关，不能突兀
- 不要让用户觉得在被"采集信息"
- 一次只问一个问题""",

    "offer_choice": """你的行动：给用户一个二选一，从他的选择里推断偏好

系统已经为这一轮准备好了一个 A/B 题（用户会在 UI 上看到按钮），你的任务是用轻松自然的语气把它问出来：

问题：{choice_prompt}
A. {choice_option_a}
B. {choice_option_b}

规则：
- **必须保留 A/B 选项结构**，并且标签就用 A / B（前端按钮上是 A 和 B）
- 选项文字可以微调措辞但语义不能变
- 开头可以给一句很自然的引入（"顺手问一个超简单的"、"这个也很快"之类），然后自然地贴出题目和两个选项
- 不要解释这是测试、不要说"帮助我了解你"这种话
- 整体 3–4 行""",

    "observe_reaction": """你的行动：给一个推荐或信息，然后观察用户的反应

主动提供一些有用的东西（推荐、建议、信息），
用户对你的反应本身就是信号。

规则：
- 推荐要具体，不要泛泛而谈
- 不要问"你觉得怎么样"——让用户自己决定是否回应
- 一个推荐就够了，不要给太多""",

    "self_disclose": """你的行动：先分享一个观察或有趣的事实，邀请用户产生共鸣

比如："我发现很多人周五其实不想自己选餐厅，就想有人直接告诉他去哪——你也是吗？"
比如："最近好像很多人都在关注 XX 这家店，不知道你听说过没有"

规则：
- 分享的内容要有趣、有价值
- 以开放式的方式结尾，但不要直接提问
- 给用户选择要不要回应的空间""",

    "give_value": """你的行动：纯粹给用户价值，不做任何采集

这是在"存信任"。给用户一个有用的信息、一个贴心的建议、或者一个有趣的发现。

规则：
- 不要问任何问题
- 不要试图引导对话方向
- 纯粹对用户有用就好
- 可以很短，一句话也行""",

    "do_nothing": """你的行动：回应用户但不主动引导

简短、友好地回应用户，不引入新话题，不问问题。
让用户掌握对话的主动权。

规则：
- 回复要短
- 不要问问题
- 不要推荐或建议
- 只是礼貌地回应""",

    "evaluate_user": """你的行动：主动"评价"用户——用一两句话说出你对他的判断

你现在根据聊天积累出的 top 维度信号是：
{top_traits}

规则：
- 用你观察到的特征做一次"我看到的是..."的总结
- 语气是轻松不装逼，像朋友随口点评（不是心理学家）
- 最后加一句"这个对吗？"或"我判断得准不准？"——留空间给用户 push back
- 整体 2–3 句话，不要列表
- 不要泄露维度名（novelty_appetite 之类），用生活化语言""",

    "show_archetype": """你的行动：揭示用户的 archetype 结果

用户画像：
类型名：{archetype_emoji} {archetype_name}
描述：{archetype_description}
Agent 承诺：{archetype_promise}

规则：
- 用兴奋但不过度的语气揭示结果
- 先说类型名和 emoji
- 然后用自己的话（而不是模板）描述这个类型
- 最后告诉用户"我以后会怎么配合你"
- 整体不要超过 5 句话
- 可以让用户觉得"你怎么这么了解我"
- 结尾自然过渡到"好了我们可以正式开始了——有什么想吃的吗？"这类""",
}

# ============================================================
# 小型模板 pool（用 asked_fallback_keys 去重，避免复读同一条）
# 每条是 (key, text)。key 稳定，用于 dedupe。
# ============================================================

SELF_DISCLOSE_POOL = [
    ("self_friday", "我发现很多人周五晚上其实不想自己选餐厅，就想有人直接告诉他去哪——你也是这样吗？"),
    ("self_decision_fatigue", "我经常看到用户一到下班就对选择很累，宁愿我直接给答案——你是这种人吗？"),
    ("self_old_vs_new", "我发现一类人去餐厅只看'熟不熟'，另一类只在意'没去过'——你偏哪边？"),
]

GIVE_VALUE_POOL = [
    ("gv_anchor", "我先不多问，想吃什么直接说，我帮你接。"),
    ("gv_open", "你这边想干嘛都可以说，我都能接。"),
    ("gv_hint", "给你一个冷知识：工作日晚餐 19:30 之后最容易排不到位，早一点出门更稳。"),
]

GIVE_VALUE_GUARDED_POOL = [
    ("gv_guarded_quiet", "那先这样，需要的时候说一声就行。"),
    ("gv_guarded_offer", "你附近最近新开了几家不错的店，想看看再说。"),
    ("gv_guarded_chill", "不急，随时喊我都在。"),
]

OBSERVE_REACTION_POOL = [
    ("or_new_store", "对了，你附近有一家评价很好的新店，感兴趣的话可以看看。"),
    ("or_time_window", "如果这周想出去吃，周四晚上通常比周五人少、选择也多。"),
    ("or_quick_win", "其实你想不出来吃什么的时候，大多数人会选饺子或者寿司——要不我帮你从这俩开始？"),
]

OBSERVE_REACTION_GUARDED_POOL = [
    ("or_guarded_note", "你附近最近新开了几家不错的店，需要的话随时说。"),
    ("or_guarded_soft", "没想好也正常，我先在这儿。"),
]

# 所有 pool 全用完的最终兜底；也按 asked_fallback_keys 轮一遍，
# 这样不至于连续 4 条"需要我的时候随时说"。
FINAL_SILENCE_POOL = [
    ("fs_wait", "需要我的时候随时说。"),
    ("fs_stay", "我在这儿，想聊就开口。"),
    ("fs_go", "你想动一下就说 'go'，我帮你接下一步。"),
    ("fs_chill", "不急，慢慢来。"),
]


# Greeting 的特殊 instructions
GREETING_INSTRUCTIONS = """你的行动：首次见面打招呼

这是你们的第一次对话。你需要：
1. 简短有趣地自我介绍（你是谁、你能做什么——用大白话）
2. 给用户一个轻松的入口选择：
   - 可以直接开始用（"你现在想吃点什么？"）
   - 可以先做个小测试让你更了解他（"花两分钟做个小测试？"）
   - 可以先了解你能做什么（"想先看看我能帮你什么？"）

不要列出功能清单。用一两句话概括你的能力就好。
语气是轻松的朋友，不是产品说明书。

注意：这三个选项不要用 1/2/3 或 A/B/C 的格式列出来，
而是自然地在一段话里融入这几个方向，让用户觉得怎么回复都行。"""


class ResponseGenerator:
    """根据 action context 生成自然语言回复"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    async def generate(self, context: dict) -> str:
        """
        生成 Agent 回复

        Args:
            context: 包含 action、user_message、各种状态的完整上下文

        Returns:
            Agent 的自然语言回复
        """
        system_prompt = self._build_system_prompt(context)
        user_message = self._build_user_prompt(context)
        meta_prefix = (context.get("meta_prefix") or "").strip()

        if self.llm_client:
            raw = await self.llm_client.chat(system_prompt, user_message)
            # LLM 模式：meta_prefix 已在 system prompt 里要求作为开头（见 _build_system_prompt）
            return raw
        # 模板模式：直接把元句拼在回复前
        body = self._template_fallback(context)
        if meta_prefix and body:
            return f"{meta_prefix}\n\n{body}"
        return meta_prefix or body

    async def generate_stream(self, context: dict):
        """
        流式版本。yield text chunks。

        LLM 模式：直接透传 LLM 的流。
        模板模式：生成整串后一次 yield（保持接口一致，UI 无需分支）。
        """
        meta_prefix = (context.get("meta_prefix") or "").strip()

        if self.llm_client:
            system_prompt = self._build_system_prompt(context)
            user_message = self._build_user_prompt(context)
            async for chunk in self.llm_client.chat_stream(system_prompt, user_message):
                if chunk:
                    yield chunk
            return

        # 模板模式：一次性吐出
        body = self._template_fallback(context)
        if meta_prefix and body:
            yield f"{meta_prefix}\n\n{body}"
        elif meta_prefix:
            yield meta_prefix
        elif body:
            yield body

    def _build_system_prompt(self, context: dict) -> str:
        """构建 system prompt"""
        action = context.get("action", "do_nothing")
        rapport_level = context.get("rapport_level", "low")
        user_state = context.get("user_state", "unknown")
        turn_count = context.get("turn_count", 0)

        # 获取 action specific instructions
        if context.get("is_greeting"):
            action_instructions = GREETING_INSTRUCTIONS
        else:
            template = ACTION_INSTRUCTIONS.get(action, ACTION_INSTRUCTIONS["do_nothing"])
            action_instructions = self._fill_action_template(template, context)

        # 对话历史
        history = context.get("conversation_history", [])
        history_str = "\n".join(
            f"{'用户' if h['role'] == 'user' else 'Agent'}: {h['content']}"
            for h in history[-4:]
        ) or "(第一轮对话)"

        # 状态转变元句（一次性）：要求 LLM 把这句话的意思放在回复开头
        meta_prefix = (context.get("meta_prefix") or "").strip()
        if meta_prefix:
            action_instructions += (
                f"\n\n【额外要求】请用下面这句话的意思作为回复的第一句（可以改措辞，但意思要到）："
                f"\n"
                f"「{meta_prefix}」"
            )

        return SYSTEM_PROMPT_BASE.format(
            rapport_level=rapport_level,
            rapport_description=RAPPORT_DESCRIPTIONS.get(rapport_level, ""),
            user_state_description=USER_STATE_DESCRIPTIONS.get(user_state, ""),
            turn_count=turn_count,
            action_specific_instructions=action_instructions,
            conversation_history=history_str,
        )

    def _build_user_prompt(self, context: dict) -> str:
        """构建 user prompt（送给 LLM 的用户消息）"""
        parts = [f"用户说：{context.get('user_message', '')}"]

        # 如果刚回答了 quiz，加入 quiz 回应
        quiz_response = context.get("quiz_response_text", "")
        if quiz_response:
            parts.append(f"\n（用户刚回答了一道测试题，你的个性化回应参考：{quiz_response}）")
            if context.get("action") == "ask_playful":
                parts.append("请用这个回应作为灵感，用你自己的话回复用户，并自然接上下一题，让对话一环扣一环。")
            else:
                parts.append("请用这个回应作为灵感，用你自己的话回复用户。回复完 quiz 反馈后，自然过渡到一个非常好接的话头。")

        # 如果用户有任务意图或正在延续一个任务
        if context.get("task_intent") or context.get("active_task"):
            task_desc = context.get("task_description") or context.get("active_task", "")
            parts.append(f"\n（用户似乎有一个任务需求：{task_desc}。优先回应这个需求，并在结尾给一个用户容易接的明确下一问或两个可选项。）")

        parts.append("\n请生成你的回复（只输出回复内容，不要输出任何其他东西）：")

        return "\n".join(parts)

    def _fill_action_template(self, template: str, context: dict) -> str:
        """填充 action template 中的变量"""
        quiz = context.get("quiz", {})
        archetype = context.get("archetype", {})
        profile = context.get("profile_snapshot", {})

        # Gap dimensions
        dims = profile.get("dimensions", {})
        gap_dims = sorted(dims.items(), key=lambda x: x[1].get("confidence", 0))[:2]
        gap_str = ", ".join(d[0] for d in gap_dims)

        # Quiz
        quiz_text = quiz.get("text", "")
        quiz_options = ""
        if quiz.get("options"):
            quiz_options = "\n".join(f"{k}. {v}" for k, v in quiz["options"].items())

        # 给 evaluate_user 准备 top_traits 字符串
        top_traits = self._format_top_traits(dims)

        # 给 offer_choice 准备 prompt / A / B 字段
        choice = context.get("choice", {}) or {}
        choice_prompt = choice.get("prompt", "")
        choice_options = choice.get("options", {}) or {}
        choice_option_a = choice_options.get("A", "")
        choice_option_b = choice_options.get("B", "")

        return template.format(
            quiz_text=quiz_text,
            quiz_options=quiz_options,
            gap_dimensions=gap_str,
            top_traits=top_traits,
            choice_prompt=choice_prompt,
            choice_option_a=choice_option_a,
            choice_option_b=choice_option_b,
            archetype_name=archetype.get("name", ""),
            archetype_emoji=archetype.get("emoji", ""),
            archetype_description=archetype.get("description", ""),
            archetype_promise=archetype.get("agent_promise", ""),
        )

    def _format_top_traits(self, dims: dict) -> str:
        """取置信度 top 2–3 维，转成自然语言的特征描述。"""
        ranked = sorted(
            dims.items(),
            key=lambda kv: kv[1].get("confidence", 0),
            reverse=True,
        )
        labels = {
            "novelty_appetite":  ("你更吃新鲜感，偏探索",     "你更吃确定感，偏忠诚"),
            "decision_tempo":    ("你偏直觉、快决策",         "你偏审慎、会多比一轮"),
            "social_energy":     ("你社交能量高、爱热闹",     "你偏独处、安静更舒服"),
            "sensory_cerebral":  ("你重感觉、在意氛围",       "你重理性、看数据"),
            "control_flow":      ("你想自己掌舵、拍板",       "你更想被安排、享受省心"),
        }
        lines = []
        for name, data in ranked:
            if data.get("confidence", 0) < 0.30:
                continue
            pole_pair = labels.get(name)
            if not pole_pair:
                continue
            lines.append(pole_pair[0] if data.get("value", 0) >= 0 else pole_pair[1])
            if len(lines) >= 3:
                break
        return "；".join(lines) if lines else "（信号还不够强）"

    def _template_fallback(self, context: dict) -> str:
        """无 LLM 时的模板 fallback（demo 用途）"""
        action = context.get("action", "do_nothing")
        user_msg = context.get("user_message", "")
        user_state = context.get("user_state", "unknown")

        # ---- Greeting ----
        if context.get("is_greeting"):
            return (
                "嘿！我是你的生活搭子 🙌\n"
                "我能帮你找好吃的、订位、做饭、安排周末——基本上跟「吃喝玩乐」有关的我都管。\n"
                "你可以直接告诉我你想吃什么，或者花两分钟做个小测试让我更快了解你——随你。"
            )

        # ---- Quiz response (刚回答了一道题) ----
        quiz_response = context.get("quiz_response_text", "")
        if quiz_response:
            if action == "ask_playful":
                return self._combine_segments(
                    quiz_response,
                    self._render_quiz_prompt(
                        context,
                        prefix="顺着这个感觉，我再追问你一个更像你的小问题——",
                    ),
                )
            if action == "show_archetype":
                return self._combine_segments(
                    quiz_response,
                    self._render_archetype(context, lead_in="我大概摸到你的路数了——"),
                )
            if action == "offer_choice":
                next_step = self._render_task_reply(context) if (context.get("task_intent") or context.get("active_task")) else self._render_offer_choice(context, intro="我顺手再收一个小偏好，这样后面会更贴你：")
                return self._combine_segments(quiz_response, next_step)
            if action == "observe_reaction" and (context.get("task_intent") or context.get("active_task")):
                return self._combine_segments(quiz_response, self._render_task_reply(context))
            if action == "give_value":
                return self._combine_segments(
                    quiz_response,
                    self._render_offer_choice(context, intro="我再顺着接一个超容易答的小问题："),
                )
            if context.get("needs_more_data") and not context.get("onboarding_complete"):
                chained_question = self._render_offer_choice(context) or self._render_quiz_prompt(context)
                if chained_question:
                    return self._combine_segments(quiz_response, chained_question)
            return quiz_response

        # ---- Evaluate user (Agent 主动评价) ----
        if action == "evaluate_user":
            return self._render_evaluate(context)

        # ---- Show archetype ----
        if action == "show_archetype":
            return self._render_archetype(context)

        # ---- Task-oriented response ----
        if user_state == "task_oriented" or context.get("task_intent") or context.get("active_task"):
            return self._render_task_reply(context)

        # ---- Ask playful (出趣味题) ----
        if action == "ask_playful":
            return self._render_quiz_prompt(context)

        # ---- Offer choice ----
        if action == "offer_choice":
            return self._render_offer_choice(context)

        # ---- Self disclose ----
        if action == "self_disclose":
            line = self._pick_fallback(context, "self_disclose", SELF_DISCLOSE_POOL)
            return line or self._final_silence(context)

        # ---- Give value ----
        if action == "give_value":
            structured = self._render_offer_choice(context, intro="我先不绕，直接接一个特别好回的小问题：")
            if structured:
                return structured
            pool = GIVE_VALUE_GUARDED_POOL if user_state == "guarded" else GIVE_VALUE_POOL
            line = self._pick_fallback(context, f"give_value_{user_state}", pool)
            return line or self._final_silence(context)

        # ---- Observe reaction ----
        if action == "observe_reaction":
            structured = self._render_offer_choice(context, intro="我先给你压个方向，再补一个小确认：")
            if structured:
                return structured
            pool = OBSERVE_REACTION_GUARDED_POOL if user_state == "guarded" else OBSERVE_REACTION_POOL
            line = self._pick_fallback(context, f"observe_{user_state}", pool)
            return line or self._final_silence(context)

        # ---- Do nothing / fallback ----
        if user_state == "disengaging":
            return "好的，需要我的时候随时说 👋"

        return "嗯嗯，需要帮忙随时说。"

    def _combine_segments(self, *parts: str) -> str:
        """把多段短回复拼成一个更顺的回应。"""
        cleaned = [part.strip() for part in parts if part and part.strip()]
        return "\n\n".join(cleaned)

    def _final_silence(self, context: dict) -> str:
        """所有 action pool 用完后的最后一兜：也做成 pool 避免连续重复。"""
        line = self._pick_fallback(context, "final_silence", FINAL_SILENCE_POOL)
        return line or "需要我的时候随时说。"

    def _pick_fallback(self, context: dict, scope: str, pool: list) -> str:
        """从一个小 pool 里取一条没用过的模板文案。

        context["asked_fallback_keys"] 是一个 set（由 ConversationManager 以 state
        的引用传入），被 mutate 后 state 也会看到。pool 元素是 (key, text)。
        全用完返回 ""。scope 用来给 key 加命名空间避免冲突。
        """
        asked = context.get("asked_fallback_keys")
        if asked is None:
            asked = set()
            context["asked_fallback_keys"] = asked
        for key, text in pool:
            full_key = f"{scope}:{key}"
            if full_key in asked:
                continue
            asked.add(full_key)
            return text
        return ""

    def _render_quiz_prompt(self, context: dict, prefix: str = "突然想问你一个问题——") -> str:
        quiz = context.get("quiz", {})
        if not quiz:
            return ""

        text = quiz.get("text", "").strip()
        options = quiz.get("options", {})
        option_lines = [f"{key}. {value}" for key, value in options.items()]
        return f"{prefix}\n\n{text}\n\n" + "\n".join(option_lines)

    def _render_evaluate(self, context: dict) -> str:
        """模板模式下的主动评价：lead-in + top traits + 确认尾巴。"""
        lead_ins = [
            ("eval_lead_chat",    "我跟你聊下来看到的是这样："),
            ("eval_lead_sense",   "按我现在对你的感觉："),
            ("eval_lead_observe", "观察下来你大概是这么个人："),
        ]
        lead = ""
        for key, text in lead_ins:
            full_key = f"evaluate:{key}"
            asked = context.get("asked_fallback_keys")
            if asked is None:
                asked = set()
                context["asked_fallback_keys"] = asked
            if full_key not in asked:
                asked.add(full_key)
                lead = text
                break
        if not lead:
            lead = "再补一个判断："

        profile = context.get("profile_snapshot", {})
        dims = profile.get("dimensions", {})
        traits = self._format_top_traits(dims)
        tail = "这个感觉对吗？觉得哪里不准可以直接说。"
        return f"{lead}\n\n{traits}。\n\n{tail}"

    def _render_archetype(self, context: dict, lead_in: str = "我看出来了——") -> str:
        arch = context.get("archetype", {})
        if not arch:
            return ""
        return (
            f"{lead_in}你是 {arch.get('emoji', '')} {arch.get('name', '')}\n\n"
            f"{arch.get('description', '')}\n\n"
            f"{arch.get('agent_promise', '')}\n\n"
            "好了，我们正式开始吧——有什么想吃的吗？"
        )

    def _render_offer_choice(self, context: dict, intro: str = "") -> str:
        """只渲染由 ConversationManager 生成的结构化 choice；无则返回 ""，
        让 _combine_segments 自然丢弃。避免掉回老的 dim→string 兜底字典
        那套会复读的逻辑（已移除）。"""
        choice = context.get("choice", {})
        if not choice:
            return ""
        prompt = choice.get("prompt", "")
        options = choice.get("options", {})
        option_lines = [f"{key}. {value}" for key, value in options.items()]
        body = self._combine_segments(prompt, "\n".join(option_lines))
        return self._combine_segments(intro, body) if intro else body

    def _render_task_reply(self, context: dict) -> str:
        user_msg = context.get("user_message", "")
        task_desc = context.get("task_description") or context.get("active_task") or user_msg
        last_action = context.get("last_action")
        task_label = self._task_label(task_desc)

        decision_phrases = ["你帮我选", "你决定", "直接", "就这个", "定吧", "别纠结"]
        rating_phrases = ["评分", "稳妥", "靠谱", "评价", "口碑"]
        cooking_phrases = ["做饭", "食谱", "冰箱", "在家"]
        location_phrases = ["附近", "近一点", "离我近", "商圈", "地铁", "公司附近"]
        ambiance_phrases = ["安静", "热闹", "氛围", "环境", "适合聊天", "约会"]

        if any(phrase in user_msg for phrase in decision_phrases):
            return f"好，那我就不让你来回选了。我先按「{task_label}」这条线直接收窄。你只要再回我两个字就行：安静，还是热闹？"

        if any(phrase in user_msg for phrase in rating_phrases):
            return f"收到，那我先走稳妥路线：按「{task_label}」这个目标，优先看评分稳定、踩雷率低的。你要我再偏安静一点，还是更有氛围一点？"

        if any(phrase in user_msg for phrase in ambiance_phrases):
            return f"好，我把这个感觉记上了。那我就按「{task_label}」继续收。下一步你更希望我优先看评分稳一点，还是离你近一点？"

        if any(phrase in user_msg for phrase in location_phrases):
            return f"收到，我先把范围压近一点。按「{task_label}」这个需求，我后面就优先给你离得近的答案。你可以直接回我一个地标，比如公司附近或家附近。"

        if any(phrase in user_msg for phrase in cooking_phrases) or any(phrase in task_desc for phrase in cooking_phrases):
            return f"收到，我先按「{task_label}」来想。你直接告诉我手头有什么食材，或者说你想吃热乎一点还是省事一点，我就能顺着往下接。"

        if last_action == "offer_choice":
            return f"我先按「{task_label}」继续往前走。你更想要评分稳一点，还是环境更有感觉一点？"

        if last_action == "observe_reaction":
            return f"那我先替你把路数压窄一点：围绕「{task_label}」，我会优先给你那种不用反复比较也能放心选的答案。你更想让我按『稳一点』还是『新一点』继续推？"

        if "餐厅" in task_desc or "吃饭" in task_desc or "吃" in task_desc:
            return f"收到，我先按「{task_label}」来想。你这次更想走评分稳一点的路线，还是想找一家氛围更对的？"

        return f"收到，我先把任务记成「{task_label}」。你可以继续补一句条件，我会顺着这个方向往下接。"

    def _task_label(self, task_desc: str) -> str:
        """把累积任务描述收成一句短标签，避免回复越聊越长。"""
        base = task_desc.split("；用户刚补充：", 1)[0].strip() or task_desc.strip()
        return base[:28] + ("..." if len(base) > 28 else "")
