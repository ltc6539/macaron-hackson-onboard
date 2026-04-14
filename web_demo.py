import argparse
import asyncio
import json
import re
import uuid
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from agent.conversation_manager import ConversationManager
from core.persistence import PersistenceStore
from main import build_llm_client, resolve_llm_type

ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"

COOKIE_NAME = "macaron_uid"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365   # 1 year
NICKNAME_MAX_LEN = 20

def _sanitize_nickname(raw: str) -> str:
    cleaned = (raw or "").strip()
    # 去掉控制字符
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)
    return cleaned[:NICKNAME_MAX_LEN]


class SessionStore:
    def __init__(self, llm_type: Optional[str] = None, db: Optional[PersistenceStore] = None):
        self.llm_type = llm_type
        self.db = db or PersistenceStore()
        self.sessions: dict[str, ConversationManager] = {}
        # session_id → (user_id, nickname) 反查，用于落最终画像
        self._session_owner: dict[str, tuple[str, str]] = {}
        # 记录哪些 session 已经把 archetype 揭示结果写入 final_profiles
        self._profiles_saved: set[str] = set()

    def _new_manager(self) -> ConversationManager:
        return ConversationManager(llm_client=build_llm_client(self.llm_type))

    def _serialize_state(self, manager: ConversationManager) -> dict:
        state = manager.get_debug_state()
        state["archetype"] = manager.archetype_mapper.match(
            manager.state.profile, manager._behavioral_snapshot()
        )
        state["history_size"] = len(manager.state.conversation_history)
        return state

    # ---------- registration / identity ----------

    def register_user(self, nickname: str) -> dict:
        safe = _sanitize_nickname(nickname)
        if not safe:
            raise ValueError("nickname required")
        user_id = uuid.uuid4().hex
        self.db.upsert_user(user_id, safe)
        return {"user_id": user_id, "nickname": safe}

    def get_user(self, user_id: str) -> Optional[dict]:
        if not user_id:
            return None
        return self.db.get_user(user_id)

    # ---------- session lifecycle ----------

    async def create_session(self, user_id: str, nickname: str) -> dict:
        session_id = uuid.uuid4().hex
        manager = self._new_manager()
        self.sessions[session_id] = manager
        self._session_owner[session_id] = (user_id, nickname)
        self.db.register_session(
            session_id, user_id, manager.state.profiling_mode.value
        )
        self.db.touch_user(user_id)

        reply = await manager.process_message("你好")
        # 首轮对话（'你好' + Agent 回复）落库
        self._persist_last_exchange(session_id, manager, user_input="你好")
        return {
            "session_id": session_id,
            "reply": reply,
            "state": self._serialize_state(manager),
        }

    async def send_message(self, session_id: str, message: str, via_button: bool = False) -> dict:
        manager = self.sessions.get(session_id)
        if manager is None:
            raise KeyError("Unknown session")

        prev_mode = manager.state.profiling_mode.value
        reply = await manager.process_message(message, via_button=via_button)

        # profiling_mode 可能因 fatigue 被自动降级，同步到 DB
        if manager.state.profiling_mode.value != prev_mode:
            self.db.update_session_mode(session_id, manager.state.profiling_mode.value)

        self._persist_last_exchange(session_id, manager, user_input=message)
        self._persist_final_profile_if_ready(session_id, manager)

        return {
            "reply": reply,
            "state": self._serialize_state(manager),
        }

    async def send_message_stream(self, session_id: str, message: str, via_button: bool = False):
        """
        send_message 的流式版本。异步生成器 yield SSE 事件 dict:
          {"type": "chunk", "text": "..."}
          {"type": "done",  "reply": "...", "state": debug_state}
        """
        manager = self.sessions.get(session_id)
        if manager is None:
            raise KeyError("Unknown session")

        prev_mode = manager.state.profiling_mode.value
        full_reply = ""
        async for event in manager.process_message_stream(message, via_button=via_button):
            if event["type"] == "chunk":
                full_reply += event["text"]
                yield event
            elif event["type"] == "done":
                full_reply = event.get("reply", full_reply)
                # profiling_mode 自动降级同步 DB
                if manager.state.profiling_mode.value != prev_mode:
                    self.db.update_session_mode(session_id, manager.state.profiling_mode.value)
                # 落库：消息 + final_profile（如果刚解锁）
                self._persist_last_exchange(session_id, manager, user_input=message)
                self._persist_final_profile_if_ready(session_id, manager)
                yield {
                    "type": "done",
                    "reply": full_reply,
                    "state": self._serialize_state(manager),
                }

    def skip_profiling(self, session_id: str) -> dict:
        manager = self.sessions.get(session_id)
        if manager is None:
            raise KeyError("Unknown session")
        manager.set_profiling_mode("passive")
        self.db.update_session_mode(session_id, manager.state.profiling_mode.value)
        return {"state": self._serialize_state(manager)}

    def reveal_archetype(self, session_id: str) -> dict:
        manager = self.sessions.get(session_id)
        if manager is None:
            raise KeyError("Unknown session")
        archetype = manager.force_reveal_archetype()
        self._persist_final_profile_if_ready(session_id, manager)
        return {
            "archetype": archetype,
            "macaron_promises": manager.build_macaron_promises(),
            "state": self._serialize_state(manager),
        }

    # ---------- persistence helpers ----------

    def _persist_last_exchange(
        self,
        session_id: str,
        manager: ConversationManager,
        user_input: str,
    ):
        history = manager.state.conversation_history
        # 最后两条应该是刚刚的 user + agent
        if len(history) < 2:
            return
        user_turn = history[-2]
        agent_turn = history[-1]
        turn_idx = manager.state.turn_count
        if user_turn.role == "user":
            self.db.append_message(
                session_id=session_id,
                turn_index=turn_idx,
                role="user",
                content=user_turn.content,
                action_type=None,
                user_state=user_turn.user_state_at_turn.value if user_turn.user_state_at_turn else None,
            )
        if agent_turn.role == "agent":
            self.db.append_message(
                session_id=session_id,
                turn_index=turn_idx,
                role="agent",
                content=agent_turn.content,
                action_type=agent_turn.action_type.value if agent_turn.action_type else None,
                user_state=None,
            )

    def _persist_final_profile_if_ready(
        self,
        session_id: str,
        manager: ConversationManager,
    ):
        if not manager.state.archetype_revealed:
            return
        if session_id in self._profiles_saved:
            return
        owner = self._session_owner.get(session_id)
        if not owner:
            return
        user_id, nickname = owner
        archetype = manager.archetype_mapper.match(
            manager.state.profile, manager._behavioral_snapshot()
        )
        if archetype.get("is_fallback"):
            return
        dims = manager.state.profile.to_dict()
        promises = manager.build_macaron_promises()
        self.db.save_final_profile(
            session_id=session_id,
            user_id=user_id,
            nickname=nickname,
            archetype=archetype,
            dims=dims,
            macaron_promises=promises,
        )
        self._profiles_saved.add(session_id)


# ============================================================
# HTTP handler
# ============================================================


class DemoHandler(BaseHTTPRequestHandler):
    store: SessionStore = None  # type: ignore[assignment]

    # ---------- GET ----------

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        if self.path == "/styles.css":
            return self._serve_file(WEB_DIR / "styles.css", "text/css; charset=utf-8")
        if self.path == "/app.js":
            return self._serve_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
        if self.path == "/api/health":
            return self._send_json({"ok": True})
        if self.path == "/api/me":
            return self._handle_me()
        if self.path == "/api/profiles":
            # 团队查看所有画像（内部调试用，无鉴权）
            return self._send_json({"profiles": self.store.db.list_profiles()})
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    # ---------- POST ----------

    def do_POST(self):
        if self.path == "/api/register":
            return self._handle_register()

        if self.path == "/api/session":
            return self._handle_session_create()

        if self.path == "/api/message":
            return self._handle_message()

        if self.path == "/api/message/stream":
            return self._handle_message_stream()

        if self.path == "/api/skip-profiling":
            return self._handle_skip()

        if self.path == "/api/reveal":
            return self._handle_reveal()

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format, *args):
        return

    # ---------- handler impls ----------

    def _handle_me(self):
        user_id = self._read_cookie(COOKIE_NAME)
        user = self.store.get_user(user_id) if user_id else None
        if user:
            return self._send_json({"user": user})
        return self._send_json({"user": None, "requires_nickname": True})

    def _handle_register(self):
        body = self._read_json()
        nickname = body.get("nickname", "")
        try:
            info = self.store.register_user(nickname)
        except ValueError:
            return self._send_json(
                {"error": "nickname required"},
                status=HTTPStatus.BAD_REQUEST,
            )
        headers = [self._build_cookie_header(info["user_id"])]
        return self._send_json(info, extra_headers=headers)

    def _handle_session_create(self):
        user_id = self._read_cookie(COOKIE_NAME)
        user = self.store.get_user(user_id) if user_id else None
        if not user:
            # 引导前端先去 /api/register
            return self._send_json(
                {"requires_nickname": True},
                status=HTTPStatus.UNAUTHORIZED,
            )
        try:
            payload = asyncio.run(self.store.create_session(user["user_id"], user["nickname"]))
        except Exception as exc:
            return self._send_json(
                {"error": f"session init failed: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        payload["nickname"] = user["nickname"]
        return self._send_json(payload, status=HTTPStatus.CREATED)

    def _handle_message(self):
        body = self._read_json()
        session_id = body.get("session_id", "")
        message = str(body.get("message", "")).strip()
        via_button = bool(body.get("via_button", False))
        if not session_id or not message:
            return self._send_json(
                {"error": "session_id and message are required"},
                status=HTTPStatus.BAD_REQUEST,
            )
        try:
            payload = asyncio.run(self.store.send_message(session_id, message, via_button=via_button))
        except KeyError:
            return self._send_json({"error": "unknown session"}, status=HTTPStatus.NOT_FOUND)
        return self._send_json(payload)

    def _handle_message_stream(self):
        """Server-Sent Events 流式回复端点。"""
        body = self._read_json()
        session_id = body.get("session_id", "")
        message = str(body.get("message", "")).strip()
        via_button = bool(body.get("via_button", False))
        if not session_id or not message:
            return self._send_json(
                {"error": "session_id and message are required"},
                status=HTTPStatus.BAD_REQUEST,
            )
        if session_id not in self.store.sessions:
            return self._send_json({"error": "unknown session"}, status=HTTPStatus.NOT_FOUND)

        # SSE headers，Connection: close 避免需要 Content-Length
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        def write_event(ev: dict):
            line = "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        async def run():
            try:
                async for ev in self.store.send_message_stream(session_id, message, via_button=via_button):
                    write_event(ev)
            except Exception as e:
                write_event({"type": "error", "message": str(e)})

        try:
            asyncio.run(run())
        except BrokenPipeError:
            # 客户端提前断开，忽略
            pass

    def _handle_skip(self):
        body = self._read_json()
        session_id = body.get("session_id", "")
        if not session_id:
            return self._send_json(
                {"error": "session_id is required"},
                status=HTTPStatus.BAD_REQUEST,
            )
        try:
            payload = self.store.skip_profiling(session_id)
        except KeyError:
            return self._send_json({"error": "unknown session"}, status=HTTPStatus.NOT_FOUND)
        return self._send_json(payload)

    def _handle_reveal(self):
        body = self._read_json()
        session_id = body.get("session_id", "")
        if not session_id:
            return self._send_json(
                {"error": "session_id is required"},
                status=HTTPStatus.BAD_REQUEST,
            )
        try:
            payload = self.store.reveal_archetype(session_id)
        except KeyError:
            return self._send_json({"error": "unknown session"}, status=HTTPStatus.NOT_FOUND)
        return self._send_json(payload)

    # ---------- plumbing ----------

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _read_cookie(self, name: str) -> Optional[str]:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        jar = SimpleCookie()
        try:
            jar.load(raw)
        except Exception:
            return None
        morsel = jar.get(name)
        return morsel.value if morsel else None

    def _build_cookie_header(self, value: str) -> tuple[str, str]:
        parts = [
            f"{COOKIE_NAME}={value}",
            f"Max-Age={COOKIE_MAX_AGE}",
            "Path=/",
            "SameSite=Lax",
            "HttpOnly",
        ]
        return ("Set-Cookie", "; ".join(parts))

    def _send_json(
        self,
        payload: dict,
        status: int = HTTPStatus.OK,
        extra_headers: Optional[list] = None,
    ):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if extra_headers:
            for name, value in extra_headers:
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description="Onboarding Agent Web Demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--llm",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider；留空按环境变量自动探测（Anthropic 优先，其次 OpenAI-compatible）",
    )
    parser.add_argument("--db", default=None, help="SQLite path（默认 data/onboarding.db）")
    args = parser.parse_args()

    try:
        llm_type = resolve_llm_type(args.llm)
    except ValueError as exc:
        raise SystemExit(str(exc))

    db = PersistenceStore(Path(args.db)) if args.db else PersistenceStore()
    DemoHandler.store = SessionStore(llm_type=llm_type, db=db)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)

    print(f"Web demo running at http://{args.host}:{args.port}")
    print(f"DB at {db.db_path}")
    if llm_type == "anthropic":
        print("LLM mode: Anthropic Claude —— Agent 会现场生成回复、主动评价你。")
    else:
        print("LLM mode: OpenAI-compatible —— Agent 会现场生成回复、主动评价你。")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        db.close()


if __name__ == "__main__":
    main()
