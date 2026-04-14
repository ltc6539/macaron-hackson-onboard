"""
ActionSelector - Agent 的"大脑"

每一轮对话，Agent 需要从行动空间中选择最优行动。
这个决策综合了三层信息：
1. Profile State → 我还需要知道什么？
2. User State → 我现在能做什么？
3. Rapport → 用户信任我到什么程度？

核心原则：
- 永远不连续问两个问题
- 信任不够时用行动而不是语言
- 见好就收
"""

import yaml
import random
from pathlib import Path
from typing import Optional

from core import AgentState, UserState, AgentAction, ProfilingMode


class ActionSelector:
    """根据 Agent 内部状态选择下一步行动"""

    # Rapport 等级对应的允许行动
    RAPPORT_GATES = {
        "low": {AgentAction.GIVE_VALUE, AgentAction.OBSERVE_REACTION, AgentAction.DO_NOTHING},
        "medium": {
            AgentAction.ASK_PLAYFUL, AgentAction.OFFER_CHOICE, AgentAction.SELF_DISCLOSE,
            AgentAction.OBSERVE_REACTION, AgentAction.GIVE_VALUE, AgentAction.EVALUATE_USER,
        },
        "high": {
            AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT,
            AgentAction.OFFER_CHOICE, AgentAction.SELF_DISCLOSE,
            AgentAction.OBSERVE_REACTION, AgentAction.GIVE_VALUE, AgentAction.EVALUATE_USER,
        },
    }

    # User State 对应的允许行动
    STATE_GATES = {
        UserState.ENGAGED_PLAYFUL: {
            AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT,
            AgentAction.OFFER_CHOICE, AgentAction.SELF_DISCLOSE, AgentAction.GIVE_VALUE,
            AgentAction.EVALUATE_USER,
        },
        UserState.TASK_ORIENTED: {
            AgentAction.OFFER_CHOICE, AgentAction.OBSERVE_REACTION, AgentAction.GIVE_VALUE,
        },
        UserState.TENTATIVE_EXPLORING: {
            AgentAction.SELF_DISCLOSE, AgentAction.GIVE_VALUE, AgentAction.OFFER_CHOICE,
            AgentAction.EVALUATE_USER,
        },
        UserState.GUARDED: {
            AgentAction.GIVE_VALUE, AgentAction.OBSERVE_REACTION, AgentAction.DO_NOTHING,
        },
        UserState.DISENGAGING: {
            AgentAction.GIVE_VALUE, AgentAction.DO_NOTHING,
        },
        UserState.UNKNOWN: {
            AgentAction.GIVE_VALUE, AgentAction.SELF_DISCLOSE, AgentAction.OFFER_CHOICE,
        },
    }

    def __init__(self, quiz_bank_path: str = None):
        self.quiz_bank = self._load_quiz_bank(quiz_bank_path)

    def select_action(self, state: AgentState) -> tuple[AgentAction, dict]:
        """
        选择下一步行动

        Args:
            state: Agent 的完整内部状态

        Returns:
            (action, context) - 行动类型和执行上下文
            context 可能包含 quiz_id, question_text 等具体内容
        """
        # Step 0: 检查是否可以展示 archetype 结果
        if self._should_reveal_archetype(state):
            return AgentAction.SHOW_ARCHETYPE, {}

        # Step 1: 计算允许的行动集合
        allowed = self._compute_allowed_actions(state)

        # Step 2: 如果画像已经足够，停止主动采集
        if not state.profile.needs_more_data() or state.onboarding_complete:
            allowed -= {AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT}

        # Step 3: 如果已经问了太多题，停止
        question_budget = self._question_budget(state)
        if state.questions_asked >= question_budget:
            allowed -= {AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT}
        elif state.questions_asked >= max(question_budget - 2, 1) and state.user_state != UserState.ENGAGED_PLAYFUL:
            allowed -= {AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT}

        # Step 4: profiling_mode 的硬闸门（passive/off 不再主动问或夹带选择）
        if state.profiling_mode != ProfilingMode.ACTIVE:
            allowed -= {AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT, AgentAction.OFFER_CHOICE}

        # Step 4b: 5 条结构化 A/B choice 问完后不再出 OFFER_CHOICE（防止掉到文字兜底复读）
        if len(state.asked_choice_keys) >= 5:
            allowed -= {AgentAction.OFFER_CHOICE}

        # Step 5: 从允许的行动中选最优
        if not allowed:
            allowed = {AgentAction.DO_NOTHING}

        action, context = self._pick_best_action(allowed, state)
        return action, context

    def _question_budget(self, state: AgentState) -> int:
        """根据 fatigue 动态收紧最大追问数：>=0.3 时减 2，>=0.6 时不再主动问。"""
        base = 5
        if state.fatigue >= 0.60:
            return state.questions_asked  # 已问数就是上限，等同于禁
        if state.fatigue >= 0.30:
            return max(base - 2, 1)
        return base

    def _compute_allowed_actions(self, state: AgentState) -> set[AgentAction]:
        """根据 rapport + user_state + 历史计算当前允许的行动"""

        # Rapport gate
        rapport_level = "low" if state.rapport < 0.3 else ("medium" if state.rapport < 0.6 else "high")
        rapport_allowed = self.RAPPORT_GATES[rapport_level]

        # User state gate
        state_allowed = self.STATE_GATES.get(state.user_state, self.STATE_GATES[UserState.UNKNOWN])

        # 取交集
        allowed = rapport_allowed & state_allowed

        # 规则 1：不能连续 ask
        if state.last_action in (AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT):
            allowed -= {AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT}

        return allowed

    def _should_reveal_archetype(self, state: AgentState) -> bool:
        """判断是否应该揭示 archetype 结果"""
        if state.archetype_revealed:
            return False
        # 硬匹配路径：3 维 conf >= 0.45
        if state.profile.confident_dimensions_count(threshold=0.45) >= 3:
            if state.user_state == UserState.ENGAGED_PLAYFUL:
                return True
            if state.questions_asked >= 2:
                return True
        # 行为型档：淡人
        if (
            state.profiling_mode != ProfilingMode.ACTIVE
            and state.turn_count >= 3
            and state.rapport < 0.35
        ):
            return True
        # 软匹配路径：问够 4 道 + 2 维 conf >= 0.35 就收尾
        if (
            state.questions_asked >= 4
            and state.profile.confident_dimensions_count(threshold=0.35) >= 2
        ):
            return True
        # 时间兜底：聊了够多轮且 profile 有基本的 shape
        if state.turn_count >= 12 and state.profile.mean_confidence() >= 0.25:
            return True
        return False

    def _pick_best_action(
        self,
        allowed: set[AgentAction],
        state: AgentState,
    ) -> tuple[AgentAction, dict]:
        """从允许的行动中选最优的"""

        # EVALUATE_USER 条件：rapport 够 + 问过题 + 有 2 维软画像 + 离上次 evaluate 够远
        can_evaluate = (
            state.rapport >= 0.4
            and state.questions_asked >= 2
            and state.profile.confident_dimensions_count(threshold=0.35) >= 2
            and state.turns_since_last_evaluate >= 3
            and not state.archetype_revealed
        )

        # 优先级逻辑
        priority_order = [
            # 到了合适时机，先主动评价一次用户（比"再问一题"更像 Agent）
            (AgentAction.EVALUATE_USER, can_evaluate),
            # 用户 engaged 且需要数据 → 问趣味题
            (AgentAction.ASK_PLAYFUL, self._can_ask_quiz(state)),
            # 如果有信任且需要数据 → 在对话中夹带选择
            (AgentAction.OFFER_CHOICE, state.profile.needs_more_data()),
            # 用自我暴露引发用户分享
            (AgentAction.SELF_DISCLOSE, state.rapport < 0.5),
            # 给推荐，观察反应
            (AgentAction.OBSERVE_REACTION, True),
            # 纯给价值
            (AgentAction.GIVE_VALUE, True),
            # 不做什么
            (AgentAction.DO_NOTHING, True),
        ]

        for action, condition in priority_order:
            if action in allowed and condition:
                context = {}
                if action == AgentAction.ASK_PLAYFUL:
                    quiz = self._select_quiz(state)
                    if quiz:
                        context = {"quiz": quiz}
                    else:
                        continue  # 没有合适的题了，跳过
                return action, context

        return AgentAction.DO_NOTHING, {}

    def _can_ask_quiz(self, state: AgentState) -> bool:
        """是否还有合适的 quiz 可以问"""
        available = self._get_available_quizzes(state)
        return len(available) > 0

    def _get_available_quizzes(self, state: AgentState) -> list[dict]:
        """获取当前可用的题目（未问过 + 满足前置条件）"""
        available = []
        for q in self.quiz_bank:
            # 已经问过的跳过
            if q["id"] in state.asked_question_ids:
                continue

            # 检查 tier 前置条件
            tier = q.get("tier", 1)
            if tier == 2 and state.rapport < 0.4:
                continue
            if tier == 3:
                if state.rapport < 0.6:
                    continue
                if state.profile.confident_dimensions_count(0.3) < 2:
                    continue

            available.append(q)
        return available

    def _select_quiz(self, state: AgentState) -> Optional[dict]:
        """
        选择信息增益最高的题目

        策略：找当前 confidence 最低的维度，选对应的题
        """
        available = self._get_available_quizzes(state)
        if not available:
            return None

        # 找最缺数据的维度
        gaps = state.profile.least_confident_dimensions(n=3)

        # 按题目与 gap 维度的匹配度排序
        def quiz_score(q):
            primary = q.get("primary_dimensions", [])
            secondary = q.get("secondary_dimensions", [])
            score = 0
            for i, gap in enumerate(gaps):
                weight = len(gaps) - i  # 最缺的权重最高
                if gap in primary:
                    score += weight * 3
                elif gap in secondary:
                    score += weight * 1
            return score

        available.sort(key=quiz_score, reverse=True)

        # 如果最高分的有多个，随机选一个（增加趣味性）
        top_score = quiz_score(available[0])
        top_quizzes = [q for q in available if quiz_score(q) == top_score]

        return random.choice(top_quizzes)

    def _load_quiz_bank(self, path: str = None) -> list[dict]:
        """加载题库"""
        if path is None:
            path = str(Path(__file__).parent.parent / "config" / "quiz_bank.yaml")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            quizzes = []
            for key, q in data.get("questions", {}).items():
                q["key"] = key
                quizzes.append(q)
            return quizzes
        except FileNotFoundError:
            return []
