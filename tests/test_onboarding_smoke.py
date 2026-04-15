import http.client
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

from agent.conversation_manager import ConversationManager
from agent.archetype_mapper import ArchetypeMapper
from core import AgentAction, ProfilingMode, ProfileState
from core.persistence import PersistenceStore


class OnboardingSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_greeting_exposes_onboarding_entry(self):
        manager = ConversationManager()

        greeting = await manager.process_message("你好")
        self.assertIn("生活搭子", greeting)
        self.assertIn("先用两个小问题快速了解我", greeting)

        choices = manager.get_pending_choices()
        self.assertIsNotNone(choices)
        assert choices is not None
        self.assertEqual(choices["kind"], "entry")
        self.assertEqual([o["value"] for o in choices["options"]], ["A", "B"])

    async def test_entry_choice_starts_first_quiz(self):
        manager = ConversationManager()
        await manager.process_message("你好")

        quiz = await manager.process_message("A", via_button=True)
        self.assertIn("A.", quiz)
        self.assertIn("B.", quiz)
        choices = manager.get_pending_choices()
        self.assertIsNotNone(choices)
        assert choices is not None
        self.assertEqual(choices["kind"], "quiz")

    async def test_quiz_flow_keeps_onboarding_alive(self):
        manager = ConversationManager()

        greeting = await manager.process_message("你好")
        self.assertIn("生活搭子", greeting)

        quiz = await manager.process_message("来个测试")
        self.assertIn("A.", quiz)
        self.assertIn("B.", quiz)

        feedback = await manager.process_message("B")
        self.assertIn("A.", feedback)
        self.assertIn("B.", feedback)
        self.assertGreater(manager.state.rapport, 0.3)
        any_dim_confident = any(
            getattr(manager.state.profile, dim).confidence > 0.0
            for dim in (
                "novelty_appetite", "control_flow", "decision_tempo",
                "social_energy", "sensory_cerebral",
            )
        )
        self.assertTrue(any_dim_confident, "quiz B 应至少推高一个维度的 confidence")

    async def test_onboarding_entry_locks_into_four_questions_then_card(self):
        manager = ConversationManager()
        await manager.process_message("你好")

        first_quiz = await manager.process_message("A", via_button=True)
        self.assertEqual(manager.state.last_action, AgentAction.ASK_PLAYFUL)
        self.assertTrue(manager.state.onboarding_session_active)
        self.assertIn("A.", first_quiz)

        second_quiz = await manager.process_message("B", via_button=True)
        self.assertEqual(manager.state.last_action, AgentAction.ASK_PLAYFUL)
        self.assertTrue(manager.state.onboarding_session_active)
        self.assertIn("A.", second_quiz)

        third_quiz = await manager.process_message("B", via_button=True)
        self.assertEqual(manager.state.last_action, AgentAction.ASK_PLAYFUL)
        self.assertTrue(manager.state.onboarding_session_active)
        self.assertIn("A.", third_quiz)

        fourth_quiz = await manager.process_message("B", via_button=True)
        self.assertEqual(manager.state.last_action, AgentAction.ASK_PLAYFUL)
        self.assertTrue(manager.state.onboarding_session_active)
        self.assertIn("A.", fourth_quiz)

        card = await manager.process_message("B", via_button=True)
        self.assertEqual(manager.state.last_action, AgentAction.SHOW_ARCHETYPE)
        self.assertTrue(manager.state.archetype_revealed)
        self.assertFalse(manager.state.onboarding_session_active)
        self.assertEqual(manager.state.onboarding_questions_answered, 4)
        self.assertTrue(
            "你是" in card or "我看出来了" in card or "画像" in card,
            f"期望第四题后直接出第一版卡，实际：{card[:80]}",
        )

    async def test_task_request_gets_task_style_response(self):
        manager = ConversationManager()
        await manager.process_message("你好")

        reply = await manager.process_message("帮我找个适合两个人吃饭的地方")
        self.assertIn("评分稳一点", reply)
        self.assertIn("氛围更对", reply)

    async def test_guarded_user_gets_low_pressure_reply(self):
        manager = ConversationManager()
        await manager.process_message("你好")

        reply = await manager.process_message("嗯")
        self.assertIn("老地方", reply)
        self.assertIn("新店", reply)

        chained = await manager.process_message("新店")
        self.assertIn("A.", chained)
        self.assertIn("B.", chained)

    async def test_skip_profiling_blocks_future_questions(self):
        manager = ConversationManager()
        await manager.process_message("你好")

        manager.set_profiling_mode("passive")
        self.assertEqual(manager.state.profiling_mode, ProfilingMode.PASSIVE)

        for msg in ["嗯", "新店", "一个人", "再说吧"]:
            await manager.process_message(msg)
            self.assertNotIn(
                manager.state.last_action,
                (AgentAction.ASK_PLAYFUL, AgentAction.ASK_DIRECT, AgentAction.OFFER_CHOICE),
                f"profiling passive 后不应再主动提问，却触发了 {manager.state.last_action}",
            )
        self.assertIsNone(manager.get_pending_choices())

    async def test_skip_profiling_is_one_way(self):
        manager = ConversationManager()
        await manager.process_message("你好")
        manager.set_profiling_mode(ProfilingMode.PASSIVE)
        manager.set_profiling_mode(ProfilingMode.ACTIVE)  # 应被拒绝
        self.assertEqual(manager.state.profiling_mode, ProfilingMode.PASSIVE)
        manager.set_profiling_mode(ProfilingMode.OFF)     # 可以继续降级
        self.assertEqual(manager.state.profiling_mode, ProfilingMode.OFF)

    async def test_fatigue_auto_downgrades_to_passive(self):
        manager = ConversationManager()
        await manager.process_message("你好")
        # 直接把 fatigue 推高（模拟一连串不耐烦信号累积到阈值）
        manager.fatigue_tracker.fatigue = 0.7
        await manager.process_message("嗯")
        self.assertIn(
            manager.state.profiling_mode,
            (ProfilingMode.PASSIVE, ProfilingMode.OFF),
        )

    async def test_pending_choices_exposed_for_frontend(self):
        manager = ConversationManager()
        await manager.process_message("你好")
        await manager.process_message("来个测试")
        choices = manager.get_pending_choices()
        self.assertIsNotNone(choices)
        assert choices is not None
        self.assertEqual(choices["kind"], "quiz")
        values = [o["value"] for o in choices["options"]]
        self.assertIn("A", values)
        self.assertIn("B", values)

    async def test_stream_retry_falls_back_to_non_stream_once(self):
        manager = ConversationManager()

        async def broken_stream(_context):
            yield "半截"
            raise RuntimeError("stream broken")

        async def fallback_generate(_context):
            return "完整回复"

        manager.response_generator.generate_stream = broken_stream
        manager.response_generator.generate = fallback_generate

        events = []
        async for event in manager.process_message_stream("你好"):
            events.append(event)

        self.assertEqual(events[0]["type"], "chunk")
        self.assertEqual(events[0]["text"], "半截")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["reply"], "完整回复")
        self.assertEqual(manager.state.last_action, AgentAction.GIVE_VALUE)
        self.assertEqual(manager.state.conversation_history[-1].content, "完整回复")

    async def test_dan_ren_archetype_on_skip_and_short_msgs(self):
        manager = ConversationManager()
        await manager.process_message("你好")
        manager.set_profiling_mode(ProfilingMode.PASSIVE)
        for msg in ["嗯", "好", "行"]:
            await manager.process_message(msg)
        result = manager.force_reveal_archetype()
        self.assertEqual(result["key"], "dan_ren")
        self.assertFalse(result["is_fallback"])

    async def test_offer_choice_does_not_repeat(self):
        """guarded 用户连续触发 offer_choice 时，同一条 A/B 不应被反复问。"""
        manager = ConversationManager()
        await manager.process_message("你好")
        seen_keys = []
        for _ in range(6):
            await manager.process_message("嗯")
            pending = manager._pending_choice
            if pending and pending.get("key"):
                seen_keys.append(pending["key"])
        # 每个 key 最多出现一次
        self.assertEqual(len(seen_keys), len(set(seen_keys)),
                         f"choice 重复了: {seen_keys}")

    async def test_offer_choice_fallback_text_does_not_repeat(self):
        """10 条敷衍后，任何硬编码问题不会复读 ≥2 次（P0 核心修复）。"""
        repeaty_lines = [
            "选吃的时候你是直觉型",
            "老地方，还是新店",
            "选餐厅的时候，你更看重氛围",
            "一个人吃，还是跟朋友",
            "自己做选择，还是更希望我直接帮你安排",
        ]
        manager = ConversationManager()
        await manager.process_message("你好")
        all_replies = []
        for _ in range(10):
            reply = await manager.process_message("嗯")
            all_replies.append(reply)
        joined = "\n".join(all_replies)
        for line in repeaty_lines:
            count = joined.count(line)
            self.assertLessEqual(
                count, 1,
                f"回复里 '{line[:20]}...' 出现了 {count} 次，应该 ≤ 1",
            )

    async def test_action_demotes_when_offer_exhausted(self):
        """asked_choice_keys 塞满 5 个后，不应再触发 OFFER_CHOICE。"""
        manager = ConversationManager()
        await manager.process_message("你好")
        # 手动灌满 5 个 key，模拟已经问完
        manager.state.asked_choice_keys = {
            "choice_novelty_v1", "choice_tempo_v1", "choice_social_v1",
            "choice_sensory_v1", "choice_control_v1",
        }
        await manager.process_message("嗯")
        self.assertNotEqual(manager.state.last_action, AgentAction.OFFER_CHOICE)

    async def test_archetype_reveals_via_soft_threshold(self):
        """松阈值路径：questions_asked >= 4 + 2 维 conf >= 0.35 → 应当触发解锁。"""
        manager = ConversationManager()
        await manager.process_message("你好")
        # 手动构造满足松阈值的状态
        manager.state.questions_asked = 4
        manager.state.profile.novelty_appetite.value = 0.5
        manager.state.profile.novelty_appetite.confidence = 0.4
        manager.state.profile.control_flow.value = 0.5
        manager.state.profile.control_flow.confidence = 0.4
        manager.accumulator.profile = manager.state.profile
        # 再发一条消息，走一遍决策
        await manager.process_message("那你说呢")
        self.assertTrue(
            manager.state.archetype_revealed,
            f"松阈值应该已满足，但 archetype 未解锁；last_action={manager.state.last_action}",
        )

    def test_soft_match_when_two_dims_moderate(self):
        """2 维 conf=0.40 的 profile 应该通过软匹配返回非 fallback 结果。"""
        profile = ProfileState()
        profile.novelty_appetite.value = 0.5
        profile.novelty_appetite.confidence = 0.40
        profile.control_flow.value = 0.5
        profile.control_flow.confidence = 0.40
        mapper = ArchetypeMapper()
        result = mapper.match(profile)
        self.assertFalse(result["is_fallback"],
                         "2 维 conf=0.4 应该软匹配成功")
        self.assertTrue(result.get("soft_match"),
                        "应该标记 soft_match=True")

    async def test_evaluate_user_fires_when_conditions_met(self):
        """rapport 够、2 维软画像、问过题 → 应该选 EVALUATE_USER。"""
        manager = ConversationManager()
        await manager.process_message("你好")
        # 手动把状态推到 evaluate 条件满足的位置
        manager.state.rapport = 0.5
        manager.rapport_tracker.rapport = 0.5
        manager.state.questions_asked = 2
        manager.state.turns_since_last_evaluate = 5
        manager.state.profile.novelty_appetite.confidence = 0.4
        manager.state.profile.novelty_appetite.value = 0.5
        manager.state.profile.control_flow.confidence = 0.4
        manager.state.profile.control_flow.value = 0.5
        manager.accumulator.profile = manager.state.profile

        await manager.process_message("哈哈有意思")
        self.assertEqual(
            manager.state.last_action,
            AgentAction.EVALUATE_USER,
            f"期望 EVALUATE_USER，实际 {manager.state.last_action}",
        )

    async def test_meh_two_picks_auto_switch_to_passive(self):
        """连续 2 次 'meh' 选项应触发自动切 passive + meta_prefix。"""
        manager = ConversationManager()
        await manager.process_message("你好")
        await manager.process_message("来个测试")
        # 第 1 次 meh
        await manager.process_message("meh", via_button=True)
        self.assertEqual(manager.state.meh_count, 1)
        self.assertEqual(manager.state.profiling_mode, ProfilingMode.ACTIVE)
        # 第 2 次 meh：需要再有 pending quiz，Agent 此轮应该又出了新题或 choice
        # 直接再发一次 meh（模拟用户点了 meh 按钮）
        reply = await manager.process_message("meh", via_button=True)
        self.assertEqual(manager.state.meh_count, 2)
        self.assertEqual(manager.state.profiling_mode, ProfilingMode.PASSIVE)
        self.assertTrue(
            "没感觉" in reply or "不测了" in reply,
            f"期望 meh 切 passive 的元句，实际：{reply[:80]}",
        )

    async def test_meh_does_not_update_profile_signals(self):
        """点 meh 不应回写任何画像信号。"""
        manager = ConversationManager()
        await manager.process_message("你好")
        await manager.process_message("来个测试")
        before = manager.accumulator.profile.to_dict()
        await manager.process_message("meh", via_button=True)
        after = manager.accumulator.profile.to_dict()
        # meh 这轮不应产生任何 quiz / choice 信号；confidence 不应增加
        for dim in before:
            self.assertAlmostEqual(
                before[dim]["confidence"], after[dim]["confidence"], places=2,
                msg=f"{dim} 的 confidence 不应因 meh 变化",
            )

    async def test_meta_prefix_on_fatigue_transition(self):
        """fatigue 从 low 跳到 elevated 时，下一条回复应带元句前缀。"""
        manager = ConversationManager()
        await manager.process_message("你好")
        # 手动让 fatigue 跨过 0.3 阈值
        manager.fatigue_tracker.fatigue = 0.35
        reply = await manager.process_message("嗯")
        self.assertTrue(
            "累了" in reply or "缩一下" in reply,
            f"期望元句前缀（累了/缩一下），实际回复：{reply[:80]}",
        )

    async def test_macaron_capability_quizzes_reachable(self):
        manager = ConversationManager()
        quiz_ids = {q["id"] for q in manager.action_selector.quiz_bank}
        for expected in ("q_macaron_friday", "q_macaron_group", "q_macaron_cook", "q_macaron_plan"):
            self.assertIn(expected, quiz_ids)


class PersistenceTests(unittest.TestCase):
    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test.db")
            store = PersistenceStore(db_path)
            store.upsert_user("u1", "阿禾")
            store.register_session("s1", "u1", "active")
            store.append_message("s1", 1, "user", "你好", None, "engaged_playful")
            store.append_message("s1", 1, "agent", "嘿", "give_value", None)
            store.save_final_profile(
                session_id="s1",
                user_id="u1",
                nickname="阿禾",
                archetype={
                    "key": "precision_hedonist",
                    "name": "精准享乐家",
                    "emoji": "🎯",
                    "description": "desc",
                    "agent_promise": "promise",
                },
                dims={"novelty_appetite": {"value": 0.5, "confidence": 0.6}},
                macaron_promises=["找餐厅我会先给你 3 张图"],
            )
            user = store.get_user("u1")
            assert user is not None
            self.assertEqual(user["nickname"], "阿禾")
            profile = store.get_final_profile("s1")
            assert profile is not None
            self.assertEqual(profile["archetype_name"], "精准享乐家")
            self.assertEqual(profile["macaron_promises"][0], "找餐厅我会先给你 3 张图")
            msgs = store.get_messages("s1")
            self.assertEqual(len(msgs), 2)
            store.close()


class WebNicknameGateTests(unittest.TestCase):
    """小集成测试：无 cookie 不能开 session；register 后可以。"""

    @classmethod
    def setUpClass(cls):
        from web_demo import DemoHandler, SessionStore

        cls.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(cls.tmpdir, "web.db")
        cls._store = SessionStore(llm_type=None, db=PersistenceStore(db_path))
        DemoHandler.store = cls._store
        cls._server = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
        cls._port = cls._server.server_address[1]
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls._server.shutdown()
        cls._server.server_close()
        cls._store.db.close()

    def _request(self, method, path, body=None, cookie=None):
        conn = http.client.HTTPConnection("127.0.0.1", self._port, timeout=5)
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        conn.request(method, path, body=json.dumps(body or {}), headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        set_cookie = resp.getheader("Set-Cookie")
        conn.close()
        return resp.status, (json.loads(data) if data else {}), set_cookie

    def test_session_requires_nickname_without_cookie(self):
        status, body, _ = self._request("POST", "/api/session")
        self.assertEqual(status, 401)
        self.assertTrue(body.get("requires_nickname"))

    def test_register_then_session_ok(self):
        status, body, set_cookie = self._request(
            "POST", "/api/register", {"nickname": "测试用户"}
        )
        self.assertEqual(status, 200)
        self.assertIn("user_id", body)
        self.assertIsNotNone(set_cookie)
        assert set_cookie is not None
        cookie_value = set_cookie.split(";", 1)[0]
        status, payload, _ = self._request("POST", "/api/session", cookie=cookie_value)
        self.assertEqual(status, 201)
        self.assertIn("session_id", payload)
        self.assertEqual(payload.get("nickname"), "测试用户")


if __name__ == "__main__":
    unittest.main()
