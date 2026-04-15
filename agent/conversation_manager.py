"""
ConversationManager - 顶层编排器

这是 Agent 的主循环，串联 Perceive → Update → Decide → Act 四个阶段。
每一轮用户消息进来，经过完整的循环后，产生 Agent 的回复。

它不直接处理 LLM 的回复生成——那是 response generator 的事。
它只负责：分析输入、更新状态、选择行动、构造 LLM prompt 的上下文。
"""

import json
from typing import Optional

from core import (
    AgentState, UserState, AgentAction,
    ConversationTurn, SignalSource, ProfilingMode,
)
from core.profile_accumulator import ProfileAccumulator
from core.user_state_detector import UserStateDetector
from core.rapport_tracker import RapportTracker
from core.signal_extractor import SignalExtractor
from core.fatigue_tracker import FatigueTracker
from agent.action_selector import ActionSelector
from agent.archetype_mapper import ArchetypeMapper
from prompts.response_generator import ResponseGenerator


class ConversationManager:
    """
    Agent 的主循环编排器

    Usage:
        manager = ConversationManager(llm_client=my_client)
        response = await manager.process_message("你好！")
        response = await manager.process_message("帮我找个餐厅")
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

        # Core engines
        self.accumulator = ProfileAccumulator()
        self.state_detector = UserStateDetector(llm_client=llm_client)
        self.rapport_tracker = RapportTracker(initial_rapport=0.25)
        self.signal_extractor = SignalExtractor(llm_client=llm_client)
        self.fatigue_tracker = FatigueTracker()

        # Agent modules
        self.action_selector = ActionSelector()
        self.archetype_mapper = ArchetypeMapper()
        self.response_generator = ResponseGenerator(llm_client=llm_client)

        # State
        self.state = AgentState()

        # 等待 quiz / structured choice 回答
        self._pending_quiz: Optional[dict] = None
        self._pending_choice: Optional[dict] = None

    async def process_message(self, user_message: str, via_button: bool = False) -> str:
        """
        处理一条用户消息，返回 Agent 回复

        完整的 Perceive → Update → Decide → Act 循环

        Args:
            via_button: 该消息是否来自前端的 A/B 按钮点选。默认 False。
                按钮点击在 UX 上是 engaged 行为，不应累积 fatigue / guarded 信号。
        """

        # ==========================================
        # PHASE 1: PERCEIVE (感知)
        # ==========================================

        # 1a. 如果有 pending quiz / choice，先处理结构化回答
        quiz_result = None
        choice_result = None
        if self._pending_quiz:
            quiz_result = self._process_quiz_answer(user_message)
            if quiz_result:
                self._pending_quiz = None
        elif self._pending_choice:
            choice_result = self._process_choice_answer(user_message)
            if choice_result:
                self._pending_choice = None

        # 1b. 检测用户状态
        user_state = await self.state_detector.detect(
            user_message, self.state.conversation_history
        )

        # 1c. 提取对话中的隐式画像信号
        last_agent_msg = ""
        if self.state.conversation_history:
            for turn in reversed(self.state.conversation_history):
                if turn.role == "agent":
                    last_agent_msg = turn.content
                    break

        extraction = await self.signal_extractor.extract(
            user_message=user_message,
            agent_message=last_agent_msg,
            conversation_context=self._build_context_string(),
        )
        has_task_intent, task_description = self._resolve_task_context(
            user_message=user_message,
            extracted_has_task_intent=extraction.get("has_task_intent", False),
            extracted_task_description=extraction.get("task_description", ""),
        )

        # ==========================================
        # PHASE 2: UPDATE (更新状态)
        # ==========================================

        # 2a. 更新用户状态
        self.state.user_state = user_state

        # 2b. 更新 rapport
        self.rapport_tracker.update_rule_based(
            user_message, user_state, self.state.last_action
        )
        self.state.rapport = self.rapport_tracker.rapport

        # 2b'. 更新 fatigue（规则 + LLM）并按阈值自动降级 profiling_mode
        self.fatigue_tracker.update_rule_based(
            user_message, user_state, self.state.last_action, via_button=via_button
        )
        if self.llm_client and not via_button:
            # 按钮点选时信息太少，没必要再跑 LLM 疲劳分析，省 token 省时间
            await self.fatigue_tracker.update_llm_based(
                user_message, last_agent_msg, self.llm_client
            )
        self.state.fatigue = self.fatigue_tracker.fatigue
        self._apply_fatigue_auto_degrade()

        # 2c. 更新画像（来自 quiz / structured choice）
        if quiz_result and quiz_result.get("signals"):
            self.accumulator.update_from_quiz(quiz_result["signals"])
        if choice_result and choice_result.get("signals"):
            self.accumulator.update_from_conversation(choice_result["signals"])

        # 2d. 更新画像（来自对话推断）
        if extraction["signals"]:
            self.accumulator.update_from_conversation(extraction["signals"])

        self.state.profile = self.accumulator.profile

        # 2d'. 计算状态转变，给 generation_context 注入 meta_prefix
        meta_prefix = self._compute_meta_prefix(user_state)

        # 2e. 记录用户消息
        user_turn = ConversationTurn(
            role="user",
            content=user_message,
            user_state_at_turn=user_state,
            signals_extracted=extraction["signals"],
        )
        self.state.add_turn(user_turn)

        # ==========================================
        # PHASE 3: DECIDE (决策)
        # ==========================================

        action, action_context, onboarding_entry_mode = self._decide_next_action(
            quiz_result=quiz_result,
            choice_result=choice_result,
            has_task_intent=has_task_intent,
        )

        # ==========================================
        # PHASE 4: ACT (执行)
        # ==========================================

        exec_result = await self._execute_action(
            action=action,
            action_context=action_context,
            user_message=user_message,
            quiz_result=quiz_result,
            choice_result=choice_result,
            has_task_intent=has_task_intent,
            task_description=task_description,
            meta_prefix=meta_prefix,
            onboarding_entry_mode=onboarding_entry_mode,
        )
        response, action = exec_result

        # 更新 Agent 状态
        self.state.last_action = action
        # EVALUATE_USER 触发时把"距离上次 evaluate"计数归零，其他时候 +1
        if action == AgentAction.EVALUATE_USER:
            self.state.turns_since_last_evaluate = 0
        else:
            self.state.turns_since_last_evaluate += 1
        if action in (AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT):
            self.state.questions_asked += 1

        # 记录 Agent 回复
        agent_turn = ConversationTurn(
            role="agent",
            content=response,
            action_type=action,
        )
        self.state.add_turn(agent_turn)

        return response

    async def process_message_stream(self, user_message: str, via_button: bool = False):

        # ==========================================
        # PHASE 4: ACT (执行)
        # ==========================================

        exec_result = await self._execute_action(
            action=action,
            action_context=action_context,
            user_message=user_message,
            quiz_result=quiz_result,
            choice_result=choice_result,
            has_task_intent=has_task_intent,
            task_description=task_description,
            meta_prefix=meta_prefix,
        )
        response, action = exec_result

        # 更新 Agent 状态
        self.state.last_action = action
        # EVALUATE_USER 触发时把"距离上次 evaluate"计数归零，其他时候 +1
        if action == AgentAction.EVALUATE_USER:
            self.state.turns_since_last_evaluate = 0
        else:
            self.state.turns_since_last_evaluate += 1
        if action in (AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT):
            self.state.questions_asked += 1

        # 记录 Agent 回复
        agent_turn = ConversationTurn(
            role="agent",
            content=response,
            action_type=action,
        )
        self.state.add_turn(agent_turn)

        return response

    async def process_message_stream(self, user_message: str, via_button: bool = False):
        """
        process_message 的流式版本。异步生成器。yield 两类事件：

          {"type": "chunk", "text": "..."}  —— 每个 LLM 吐出的片段
          {"type": "done",  "reply": "完整字符串", "state": debug_state}  —— 最后一条

        Perceive/Update/Decide 和 process_message 完全一致；只有 Act 阶段改成流式。
        所有 state mutation（rapport / fatigue / pending / asked_*）都发生在 Phase 1-3，
        因此前端收到 "done" 时 get_debug_state() 已经包含了本轮所有状态（choices / archetype 等）。
        """
        # ----- Phase 1 perceive -----
        quiz_result = None
        choice_result = None
        if self._pending_quiz:
            quiz_result = self._process_quiz_answer(user_message)
            if quiz_result:
                self._pending_quiz = None
        elif self._pending_choice:
            choice_result = self._process_choice_answer(user_message)
            if choice_result:
                self._pending_choice = None

        user_state = await self.state_detector.detect(
            user_message, self.state.conversation_history
        )

        last_agent_msg = ""
        for turn in reversed(self.state.conversation_history):
            if turn.role == "agent":
                last_agent_msg = turn.content
                break

        extraction = await self.signal_extractor.extract(
            user_message=user_message,
            agent_message=last_agent_msg,
            conversation_context=self._build_context_string(),
        )
        has_task_intent, task_description = self._resolve_task_context(
            user_message=user_message,
            extracted_has_task_intent=extraction.get("has_task_intent", False),
            extracted_task_description=extraction.get("task_description", ""),
        )

        # ----- Phase 2 update -----
        self.state.user_state = user_state
        self.rapport_tracker.update_rule_based(
            user_message, user_state, self.state.last_action
        )
        self.state.rapport = self.rapport_tracker.rapport
        self.fatigue_tracker.update_rule_based(
            user_message, user_state, self.state.last_action, via_button=via_button
        )
        if self.llm_client and not via_button:
            await self.fatigue_tracker.update_llm_based(
                user_message, last_agent_msg, self.llm_client
            )
        self.state.fatigue = self.fatigue_tracker.fatigue
        self._apply_fatigue_auto_degrade()

        if quiz_result and quiz_result.get("signals"):
            self.accumulator.update_from_quiz(quiz_result["signals"])
        if choice_result and choice_result.get("signals"):
            self.accumulator.update_from_conversation(choice_result["signals"])
        if extraction["signals"]:
            self.accumulator.update_from_conversation(extraction["signals"])
        self.state.profile = self.accumulator.profile

        meta_prefix = self._compute_meta_prefix(user_state)

        user_turn = ConversationTurn(
            role="user",
            content=user_message,
            user_state_at_turn=user_state,
            signals_extracted=extraction["signals"],
        )
        self.state.add_turn(user_turn)

        # ----- Phase 3 decide -----
        action, action_context, onboarding_entry_mode = self._decide_next_action(
            quiz_result=quiz_result,
            choice_result=choice_result,
            has_task_intent=has_task_intent,
        )

        # ----- Phase 4 act (streaming) -----
        generation_context, effective_action = self._build_generation_context(
            action, action_context, user_message,
            quiz_result, choice_result, has_task_intent, task_description, meta_prefix,
            onboarding_entry_mode,
        )

        chunks = []
        # 先尝试流式；如果上游在中途断流，就在同一轮里回退到非流式重试，
        # 这样不会重复处理用户消息，也更容易把最终画像卡补出来。
        try:
            async for chunk in self.response_generator.generate_stream(generation_context):
                chunks.append(chunk)
                yield {"type": "chunk", "text": chunk}
            full_response = "".join(chunks)
        except Exception:
            full_response = await self.response_generator.generate(generation_context)

        # ----- finalize state -----
        self.state.last_action = effective_action
        if effective_action == AgentAction.EVALUATE_USER:
            self.state.turns_since_last_evaluate = 0
        else:
            self.state.turns_since_last_evaluate += 1
        if effective_action in (AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT):
            self.state.questions_asked += 1

        agent_turn = ConversationTurn(
            role="agent",
            content=full_response,
            action_type=effective_action,
        )
        self.state.add_turn(agent_turn)

        yield {"type": "done", "reply": full_response}

    async def _execute_action(
        self,
        action: AgentAction,
        action_context: dict,
        user_message: str,
        quiz_result: Optional[dict],
        choice_result: Optional[dict],
        has_task_intent: bool,
        task_description: str,
        meta_prefix: str = "",
        onboarding_entry_mode: Optional[str] = None,
    ) -> tuple:
        """执行选定的行动，生成回复。返回 (response_text, effective_action)。"""
        generation_context, effective_action = self._build_generation_context(
            action, action_context, user_message,
            quiz_result, choice_result, has_task_intent, task_description, meta_prefix,
            onboarding_entry_mode,
        )
        response = await self.response_generator.generate(generation_context)
        return response, effective_action

    def _build_generation_context(
        self,
        action: AgentAction,
        action_context: dict,
        user_message: str,
        quiz_result: Optional[dict],
        choice_result: Optional[dict],
        has_task_intent: bool,
        task_description: str,
        meta_prefix: str,
        onboarding_entry_mode: Optional[str] = None,
    ) -> tuple:
        """从 action + state 组装 generation_context。会 mutate 相关状态
        （_pending_quiz / _pending_choice / asked_choice_keys / asked_question_ids）。
        返回 (context, effective_action)。effective_action 可能因 offer demote 改变。"""
        generation_context = {
            "action": action.value,
            "user_message": user_message,
            "user_state": self.state.user_state.value,
            "rapport": self.state.rapport,
            "rapport_level": self.rapport_tracker.level,
            "profile_snapshot": self.accumulator.get_snapshot(),
            "turn_count": self.state.turn_count,
            "last_action": self.state.last_action.value if self.state.last_action else None,
            "active_task": self.state.active_task,
            "needs_more_data": self.state.profile.needs_more_data(),
            "onboarding_complete": self.state.onboarding_complete,
            # 传引用：ResponseGenerator 挑 pool 文案时会 mutate 这个 set
            "asked_fallback_keys": self.state.asked_fallback_keys,
            # 状态转变元句（若有）：要拼到回复开头
            "meta_prefix": meta_prefix,
            "onboarding_entry_mode": onboarding_entry_mode,
            "conversation_history": [
                {"role": t.role, "content": t.content[:200]}
                for t in self.state.conversation_history[-6:]
            ],
        }

        # 根据不同行动添加额外上下文
        if action == AgentAction.ASK_PLAYFUL:
            quiz = action_context.get("quiz")
            if quiz:
                self._pending_quiz = quiz
                self._pending_choice = None
                self.state.asked_question_ids.append(quiz["id"])
                generation_context["quiz"] = {
                    "text": quiz["text"],
                    "options": {
                        k: v["text"] for k, v in quiz["options"].items()
                    },
                }

        elif action == AgentAction.SHOW_ARCHETYPE:
            archetype = self.archetype_mapper.match(
                self.state.profile, self._behavioral_snapshot()
            )
            self.state.archetype_revealed = True
            self.state.onboarding_complete = True
            generation_context["archetype"] = archetype

        elif action == AgentAction.GIVE_VALUE and action_context.get("type") == "greeting":
            generation_context["is_greeting"] = True
            entry_choice = self._build_onboarding_entry_choice()
            self._pending_choice = entry_choice
            generation_context["entry_choice"] = {
                "prompt": entry_choice["prompt"],
                "options": {
                    key: value["text"] for key, value in entry_choice["options"].items()
                },
            }

        # 结构化 A/B 选项：
        # - OFFER_CHOICE：不论 LLM 还是模板模式都挂一个 choice，前端才有按钮、signals 才能回写
        # - GIVE_VALUE / OBSERVE_REACTION：只在模板模式下夹带（LLM 模式这两个动作就该是"纯给价值/纯观察"，不该再问）
        if self.llm_client is None:
            build_choice_for = (AgentAction.OFFER_CHOICE, AgentAction.GIVE_VALUE, AgentAction.OBSERVE_REACTION)
        else:
            build_choice_for = (AgentAction.OFFER_CHOICE,)
        if (
            action in build_choice_for
            and action_context.get("type") not in ("greeting", "task_kickoff")
            and not has_task_intent
            and not self.state.active_task
            and self.state.profiling_mode == ProfilingMode.ACTIVE
        ):
            choice = self._build_offer_choice()
            if choice:
                self._pending_choice = choice
                if choice.get("key"):
                    self.state.asked_choice_keys.add(choice["key"])
                generation_context["choice"] = {
                    "prompt": choice["prompt"],
                    "options": {
                        key: value["text"] for key, value in choice["options"].items()
                    },
                }
            elif action == AgentAction.OFFER_CHOICE:
                # 没有可用 choice 了，demote 到 give_value，避免掉到文字兜底复读
                action = AgentAction.GIVE_VALUE
                generation_context["action"] = action.value

        # 如果 quiz_result / choice_result 有内容，加入上下文
        if quiz_result:
            generation_context["quiz_response_text"] = quiz_result.get("agent_response", "")
        elif choice_result and not choice_result.get("entry_action"):
            generation_context["quiz_response_text"] = choice_result.get("agent_response", "")

        # 如果用户有任务意图，优先处理任务
        if has_task_intent:
            self._pending_choice = None
            generation_context["task_intent"] = True
            generation_context["task_description"] = task_description

        return generation_context, action

    def _decide_next_action(
        self,
        quiz_result: Optional[dict],
        choice_result: Optional[dict],
        has_task_intent: bool,
    ) -> tuple:
        """决定本轮 action；入口按钮和 onboarding flow 可覆盖常规策略。"""
        if self.state.turn_count == 1:
            return AgentAction.GIVE_VALUE, {"type": "greeting"}, None

        if choice_result and choice_result.get("entry_action") == "start_quiz":
            self.state.onboarding_session_active = True
            self.state.onboarding_questions_answered = 0
            quiz = self.action_selector._select_quiz(self.state)
            if quiz:
                return AgentAction.ASK_PLAYFUL, {"quiz": quiz}, "start_quiz"
            self.state.onboarding_session_active = False
            return AgentAction.SHOW_ARCHETYPE, {"type": "onboarding_reveal"}, "onboarding_reveal"

        if choice_result and choice_result.get("entry_action") == "start_task":
            self.state.onboarding_session_active = False
            self.state.onboarding_questions_answered = 0
            self.set_profiling_mode(ProfilingMode.PASSIVE)
            return AgentAction.GIVE_VALUE, {"type": "task_kickoff"}, "start_task"

        if self.state.onboarding_session_active:
            if has_task_intent:
                self.state.onboarding_session_active = False
                self.set_profiling_mode(ProfilingMode.PASSIVE)
                return AgentAction.GIVE_VALUE, {"type": "task_kickoff"}, "start_task"

            if quiz_result and not quiz_result.get("is_meh"):
                self.state.onboarding_questions_answered += 1

            if self._should_finish_onboarding_flow():
                self.state.onboarding_session_active = False
                return AgentAction.SHOW_ARCHETYPE, {"type": "onboarding_reveal"}, "onboarding_reveal"

            if self.state.profiling_mode == ProfilingMode.ACTIVE:
                quiz = self.action_selector._select_quiz(self.state)
                if quiz:
                    return AgentAction.ASK_PLAYFUL, {"quiz": quiz}, "continue_onboarding"

            self.state.onboarding_session_active = False
            return AgentAction.SHOW_ARCHETYPE, {"type": "onboarding_reveal"}, "onboarding_reveal"

        action, action_context = self.action_selector.select_action(self.state)
        return action, action_context, None

    def _should_finish_onboarding_flow(self) -> bool:
        """onboarding-first 入口下，至少完成 4 题再给第一版卡。"""
        answered = self.state.onboarding_questions_answered
        return answered >= self.state.onboarding_questions_target

    def _build_onboarding_entry_choice(self) -> dict:
        """首次欢迎后的明确入口：先 onboarding，或先办事。"""
        return {
            "kind": "entry",
            "prompt": "我们先怎么开始？",
            "options": {
                "A": {
                    "text": "先用两个小问题快速了解我",
                    "entry_action": "start_quiz",
                },
                "B": {
                    "text": "先干活，画像慢慢来",
                    "entry_action": "start_task",
                },
            },
        }

    def _compute_meta_prefix(self, current_user_state) -> str:
        """
        算出一条状态转变的"元句"，一次性（通过 meta_transitions_fired 去重）。

        关键转变：
          - fatigue: low → elevated（用户开始烦了）
          - fatigue: elevated/high → auto passive（已经切到被动模式）
          - user_state: → guarded 连续 2 轮
          - mean_confidence: <0.5 → >=0.5（首次"我摸到你路数了"）
        """
        prefix = ""
        fired = self.state.meta_transitions_fired

        # 最高优先级：用户连续点了 ≥2 次"对这题没感觉" → 自动切 passive
        if self.state.meh_count >= 2 and "meta_meh_out" not in fired:
            fired.add("meta_meh_out")
            if self.state.profiling_mode == ProfilingMode.ACTIVE:
                self.set_profiling_mode(ProfilingMode.PASSIVE)
            prefix = "这些题你好像都没感觉，我不测了，直接聊就行。"
            self.state.last_meta_prefix = prefix
            return prefix

        # fatigue level 转变
        current_fat_level = self.fatigue_tracker.level
        prev_fat_level = self.state.last_fatigue_level
        if prev_fat_level is not None:
            if prev_fat_level == "low" and current_fat_level in ("elevated", "high", "critical"):
                tag = "meta_fatigue_1"
                if tag not in fired:
                    fired.add(tag)
                    prefix = "你好像有点累了，我缩一下，不多追问。"
            if current_fat_level in ("high", "critical") and self.state.profiling_mode != ProfilingMode.ACTIVE:
                tag = "meta_fatigue_2"
                if tag not in fired:
                    fired.add(tag)
                    prefix = "收到，我接下来不主动问问题了，你想怎么聊都行。"
        self.state.last_fatigue_level = current_fat_level

        # guarded 连续 2 轮
        cur_state_val = current_user_state.value if hasattr(current_user_state, "value") else str(current_user_state)
        if cur_state_val == "guarded":
            self.state.guarded_streak += 1
        else:
            self.state.guarded_streak = 0
        if self.state.guarded_streak >= 2 and "meta_guarded" not in fired:
            fired.add("meta_guarded")
            if not prefix:
                prefix = "那先不聊这个，慢慢来。"
        self.state.last_user_state_for_meta = cur_state_val

        # confidence 首次越过 0.5
        mean_conf = self.accumulator.profile.mean_confidence()
        if mean_conf >= 0.5 and "meta_confidence_unlock" not in fired:
            fired.add("meta_confidence_unlock")
            if not prefix:
                prefix = "我大概摸到你的路数了——"

        self.state.last_meta_prefix = prefix
        return prefix

    def _behavioral_snapshot(self) -> dict:
        """打包给 ArchetypeMapper 的行为快照，用于行为型 archetype 判定。"""
        user_msgs = [
            t.content.strip()
            for t in self.state.conversation_history
            if t.role == "user" and t.content.strip()
        ]
        if user_msgs:
            avg_len = sum(len(m) for m in user_msgs) / len(user_msgs)
        else:
            avg_len = 0.0
        return {
            "profiling_mode": self.state.profiling_mode.value,
            "turn_count": self.state.turn_count,
            "avg_user_msg_len": avg_len,
            "rapport": self.state.rapport,
            "fatigue": self.state.fatigue,
        }

    def force_reveal_archetype(self) -> dict:
        """无视阈值直接揭示当前 archetype（给 UI 的'查看我的画像'按钮用）。"""
        archetype = self.archetype_mapper.match(
            self.state.profile, self._behavioral_snapshot()
        )
        self.state.archetype_revealed = True
        if not archetype.get("is_fallback"):
            self.state.onboarding_complete = True
        return archetype

    def build_macaron_promises(self, limit: int = 3) -> list:
        """根据 top 维度从 archetypes.yaml 里的 macaron_promises_by_dim 取能力承诺。"""
        promises_map = self.archetype_mapper.config.get("macaron_promises_by_dim", {})
        if not promises_map:
            return []

        dims = self.state.profile.to_dict()
        # 按 confidence 从高到低选 top 维度；confidence 太低不出承诺
        ranked = sorted(
            dims.items(),
            key=lambda kv: kv[1].get("confidence", 0),
            reverse=True,
        )
        out: list = []
        for dim_name, data in ranked:
            # 下限放松到 0.15：软匹配卡片也能有话可说（配合 soft_match UI 软语气）
            if data.get("confidence", 0) < 0.15:
                continue
            entry = promises_map.get(dim_name, {})
            pole = "positive" if data.get("value", 0) >= 0 else "negative"
            line = entry.get(pole)
            if line and line not in out:
                out.append(line)
            if len(out) >= limit:
                break
        return out

    def set_profiling_mode(self, mode) -> ProfilingMode:
        """单向降级：active → passive → off。不允许反向开启。"""
        if isinstance(mode, str):
            mode = ProfilingMode(mode)
        order = {ProfilingMode.ACTIVE: 0, ProfilingMode.PASSIVE: 1, ProfilingMode.OFF: 2}
        if order[mode] <= order[self.state.profiling_mode]:
            return self.state.profiling_mode
        self.state.profiling_mode = mode
        # 降级后立即清掉待答题，避免前端继续显示按钮
        if mode != ProfilingMode.ACTIVE:
            self._pending_quiz = None
            self._pending_choice = None
        return self.state.profiling_mode

    def _apply_fatigue_auto_degrade(self):
        """fatigue 过阈值自动降级 profiling_mode。"""
        if self.fatigue_tracker.should_auto_off():
            self.set_profiling_mode(ProfilingMode.OFF)
        elif self.fatigue_tracker.should_auto_passive():
            self.set_profiling_mode(ProfilingMode.PASSIVE)

    def get_pending_choices(self) -> Optional[dict]:
        """返回当前等待用户回答的结构化选项（给前端渲染按钮）。
        每个 pending 的 options 末尾都会追加一个 "meh"（对这题没感觉）选项 —— 这是
        UI 出口而不是真的维度选项，不进 YAML。用户选 meh 时不回写画像信号，累积
        多次后 Agent 会自动切到 passive 模式。"""
        meh_option = {"value": "meh", "label": "对这题没感觉"}
        if self._pending_quiz:
            q = self._pending_quiz
            base = [
                {"value": key, "label": opt.get("text", "") if isinstance(opt, dict) else str(opt)}
                for key, opt in q.get("options", {}).items()
            ]
            return {
                "kind": "quiz",
                "prompt": q.get("text", "").strip(),
                "options": base + [meh_option],
            }
        if self._pending_choice:
            c = self._pending_choice
            base = [
                {"value": key, "label": opt.get("text", "") if isinstance(opt, dict) else str(opt)}
                for key, opt in c.get("options", {}).items()
            ]
            if c.get("kind") == "entry":
                return {
                    "kind": "entry",
                    "prompt": c.get("prompt", ""),
                    "options": base,
                }
            return {
                "kind": "choice",
                "prompt": c.get("prompt", ""),
                "options": base + [meh_option],
            }
        return None

    def _process_quiz_answer(self, user_message: str) -> Optional[dict]:
        """处理用户对 quiz 的回答"""
        quiz = self._pending_quiz
        if not quiz:
            return None

        # "meh" = 对这题没感觉：不回写任何信号，累计 meh_count
        if user_message.strip().lower() == "meh":
            self.state.meh_count += 1
            return {
                "quiz_id": quiz.get("id"),
                "selected_option": "meh",
                "signals": {},
                "agent_response": "",
                "is_meh": True,
            }

        msg = user_message.strip().upper()

        # 尝试匹配选项
        matched_option = None

        # 直接匹配 A/B/C/D
        if msg in quiz.get("options", {}):
            matched_option = msg

        # 模糊匹配：看用户的回复中是否包含某个选项的关键词
        if matched_option is None:
            for key, opt in quiz.get("options", {}).items():
                opt_text = opt.get("text", "") if isinstance(opt, dict) else str(opt)
                # 取选项文本的前 6 个字作为关键词
                keyword = opt_text[:6]
                if keyword in user_message:
                    matched_option = key
                    break

        # 如果还是没匹配到，尝试数字匹配
        if matched_option is None:
            option_keys = list(quiz.get("options", {}).keys())
            number_map = {"1": 0, "一": 0, "第一": 0,
                          "2": 1, "二": 1, "第二": 1,
                          "3": 2, "三": 2, "第三": 2,
                          "4": 3, "四": 3, "第四": 3}
            for num_str, idx in number_map.items():
                if num_str in user_message and idx < len(option_keys):
                    matched_option = option_keys[idx]
                    break

        if matched_option is None:
            return None

        # 提取信号
        option_data = quiz["options"][matched_option]
        signals = option_data.get("signals", {})

        # 获取 agent 回应
        agent_responses = quiz.get("agent_responses", {})
        agent_response = agent_responses.get(matched_option, "")

        return {
            "quiz_id": quiz["id"],
            "selected_option": matched_option,
            "signals": signals,
            "agent_response": agent_response,
        }

    def _build_offer_choice(self) -> Optional[dict]:
        """给模板模式生成一个可直接回答的 A/B 选择题。"""
        profile = self.accumulator.get_snapshot()
        dims = profile.get("dimensions", {})
        if not dims:
            return None

        choice_bank = {
            "novelty_appetite": {
                "key": "choice_novelty_v1",
                "prompt": "周末要约朋友吃饭，你翻出收藏夹——",
                "options": {
                    "A": {
                        "text": "点开那家去过 3 次的老地方",
                        "signals": {"novelty_appetite": {"value": -0.35, "confidence": 0.18}},
                        "agent_response": "懂，你更吃确定感那一边。",
                    },
                    "B": {
                        "text": "滑过它，去看最近加进来的那几家新店",
                        "signals": {"novelty_appetite": {"value": 0.35, "confidence": 0.18}},
                        "agent_response": "懂，你对新鲜感是有点 appetite 的。",
                    },
                },
            },
            "decision_tempo": {
                "key": "choice_tempo_v1",
                "prompt": "菜单递上来那一刻——",
                "options": {
                    "A": {
                        "text": "前两页就锁定了，先吃上再说",
                        "signals": {"decision_tempo": {"value": 0.3, "confidence": 0.18}},
                        "agent_response": "好，快决策型，我记住了。",
                    },
                    "B": {
                        "text": "一页一页翻到最后，先比完再决定",
                        "signals": {"decision_tempo": {"value": -0.3, "confidence": 0.18}},
                        "agent_response": "好，你会先过一轮脑子再下决定。",
                    },
                },
            },
            "social_energy": {
                "key": "choice_social_v1",
                "prompt": "理想中的\"好好吃一顿\"画面里——",
                "options": {
                    "A": {
                        "text": "一个人戴着耳机慢慢吃",
                        "signals": {"social_energy": {"value": -0.3, "confidence": 0.16}},
                        "agent_response": "收到，你更像是安静充电型。",
                    },
                    "B": {
                        "text": "一桌朋友边吵边夹菜",
                        "signals": {"social_energy": {"value": 0.3, "confidence": 0.16}},
                        "agent_response": "收到，你更容易在热闹里来电。",
                    },
                },
            },
            "sensory_cerebral": {
                "key": "choice_sensory_v1",
                "prompt": "挑店的 30 秒里你先盯哪个——",
                "options": {
                    "A": {
                        "text": "门头照片和菜品图，氛围对不对",
                        "signals": {"sensory_cerebral": {"value": 0.3, "confidence": 0.18}},
                        "agent_response": "懂，你是先看感觉的那种。",
                    },
                    "B": {
                        "text": "评分和人均，先确认不踩雷",
                        "signals": {"sensory_cerebral": {"value": -0.3, "confidence": 0.18}},
                        "agent_response": "懂，你会先拿事实做锚点。",
                    },
                },
            },
            "control_flow": {
                "key": "choice_control_v1",
                "prompt": "周末前一晚，你更踏实的状态是——",
                "options": {
                    "A": {
                        "text": "已经订好明天几点去哪，省心",
                        "signals": {"control_flow": {"value": 0.35, "confidence": 0.18}},
                        "agent_response": "好，你还是喜欢自己掌舵。",
                    },
                    "B": {
                        "text": "明天起床再看心情，不想提前被排满",
                        "signals": {"control_flow": {"value": -0.35, "confidence": 0.18}},
                        "agent_response": "好，你不想把精力花在提前规划上。",
                    },
                },
            },
        }

        # 按 confidence 从低到高遍历维度，挑第一个 key 还没问过的
        gaps = sorted(dims.items(), key=lambda item: item[1].get("confidence", 0))
        for gap_dim, _ in gaps:
            choice = choice_bank.get(gap_dim)
            if not choice:
                continue
            if choice.get("key") in self.state.asked_choice_keys:
                continue
            return choice
        # 全问过了——不再塞 offer_choice，让上层选别的 action
        return None

    def _process_choice_answer(self, user_message: str) -> Optional[dict]:
        """处理用户对 A/B 结构化选择题的回答。"""
        choice = self._pending_choice
        if not choice:
            return None

        # "meh" = 对这题没感觉
        if user_message.strip().lower() == "meh":
            self.state.meh_count += 1
            return {
                "selected_option": "meh",
                "signals": {},
                "agent_response": "",
                "is_meh": True,
            }

        msg = user_message.strip().upper()
        matched_option = None

        if msg in choice.get("options", {}):
            matched_option = msg

        if matched_option is None:
            option_keys = list(choice.get("options", {}).keys())
            number_map = {
                "1": 0, "一": 0, "第一": 0,
                "2": 1, "二": 1, "第二": 1,
                "3": 2, "三": 2, "第三": 2,
                "4": 3, "四": 3, "第四": 3,
            }
            for token, idx in number_map.items():
                if token in user_message and idx < len(option_keys):
                    matched_option = option_keys[idx]
                    break

        if matched_option is None:
            keyword_hints = [
                "老地方", "新店", "先吃", "先比", "锁定",
                "一个人", "朋友", "氛围", "评分",
                "省心", "看心情", "起床",
            ]
            for key, option in choice.get("options", {}).items():
                option_text = option.get("text", "")
                if option_text[:8] in user_message:
                    matched_option = key
                    break
                if any(keyword in user_message and keyword in option_text for keyword in keyword_hints):
                    matched_option = key
                    break

        if matched_option is None:
            return None

        option_data = choice["options"][matched_option]
        return {
            "selected_option": matched_option,
            "signals": option_data.get("signals", {}),
            "agent_response": option_data.get("agent_response", ""),
            "entry_action": option_data.get("entry_action"),
        }

    def _resolve_task_context(
        self,
        user_message: str,
        extracted_has_task_intent: bool,
        extracted_task_description: str,
    ) -> tuple[bool, str]:
        """把显式任务和简短 follow-up 串成一个连续任务上下文。"""
        message = user_message.strip()

        stop_phrases = ["不用了", "算了", "先这样", "先到这", "没事了"]
        if any(phrase in message for phrase in stop_phrases):
            self.state.active_task = None
            return False, ""

        if self.state.active_task and self._looks_like_task_followup(message) and not message.startswith(("帮我", "我想", "我要")):
            task_description = f"{self.state.active_task}；用户刚补充：{message}"
            self.state.active_task = task_description
            return True, task_description

        if extracted_has_task_intent:
            task_description = extracted_task_description.strip() or message
            self.state.active_task = task_description
            return True, task_description

        if self.state.active_task and self._looks_like_task_followup(message):
            task_description = f"{self.state.active_task}；用户刚补充：{message}"
            self.state.active_task = task_description
            return True, task_description

        return False, ""

    def _looks_like_task_followup(self, user_message: str) -> bool:
        """识别像“评分高一点”“你帮我选”这类延续任务的短回复。"""
        if not user_message:
            return False

        followup_keywords = [
            "评分", "高一点", "低一点", "预算", "人均", "附近", "近一点", "远一点",
            "安静", "热闹", "两个人", "一个人", "几个人", "包间", "环境", "氛围",
            "新开的", "老地方", "稳妥", "不要辣", "清淡", "快一点",
            "你帮我选", "你决定", "直接", "就这个", "第一个", "都可以但", "不要这个",
        ]
        if any(keyword in user_message for keyword in followup_keywords):
            return True

        if len(user_message) <= 16 and self.state.user_state == UserState.TASK_ORIENTED:
            return True

        return False

    def _build_context_string(self) -> str:
        """构建对话上下文字符串（给 LLM 用）"""
        recent = self.state.conversation_history[-4:]
        lines = []
        for turn in recent:
            role = "用户" if turn.role == "user" else "Agent"
            lines.append(f"{role}: {turn.content[:200]}")
        return "\n".join(lines) if lines else ""

    def get_debug_state(self) -> dict:
        """导出 Agent 当前状态（用于调试和可视化）"""
        return {
            "profile": self.accumulator.get_snapshot(),
            "user_state": self.state.user_state.value,
            "rapport": round(self.state.rapport, 3),
            "rapport_level": self.rapport_tracker.level,
            "fatigue": round(self.state.fatigue, 3),
            "fatigue_level": self.fatigue_tracker.level,
            "profiling_mode": self.state.profiling_mode.value,
            "turn_count": self.state.turn_count,
            "questions_asked": self.state.questions_asked,
            "onboarding_session_active": self.state.onboarding_session_active,
            "onboarding_questions_answered": self.state.onboarding_questions_answered,
            "last_action": self.state.last_action.value if self.state.last_action else None,
            "onboarding_complete": self.state.onboarding_complete,
            "archetype_revealed": self.state.archetype_revealed,
            "active_task": self.state.active_task,
            "pending_quiz": self._pending_quiz["id"] if self._pending_quiz else None,
            "pending_choice": self._pending_choice["prompt"] if self._pending_choice else None,
            "choices": self.get_pending_choices(),
            "macaron_promises": self.build_macaron_promises() if self.state.archetype_revealed else [],
            "meta_prefix_last_turn": self.state.last_meta_prefix,
        }
