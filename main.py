"""
Onboarding Agent - 可运行的交互式 demo

运行方式:
  1. 接入 OpenAI 兼容 API:
     OPENAI_API_KEY=xxx OPENAI_BASE_URL=xxx python main.py --llm openai

  2. 接入 Anthropic API:
     ANTHROPIC_API_KEY=xxx python main.py --llm anthropic

注意：当前项目只支持 LLM mode；未配置可用 provider 时会直接报错退出。
"""

import asyncio
import argparse
import json
import os
import sys

# 确保 import 路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.conversation_manager import ConversationManager


def load_local_env(env_name: str = ".env"):
    """从项目根目录加载 .env，并让项目本地配置优先。"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), env_name)
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


def resolve_llm_type(cli_choice=None):
    """解析当前要使用的 LLM provider；未配置时直接报错。"""
    load_local_env()

    if cli_choice in ("openai", "anthropic"):
        return cli_choice
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise ValueError(
        "未检测到可用的 LLM 配置。请设置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY，"
        "也可以先复制 .env.example 到 .env。"
    )


def build_llm_client(llm_type: str):
    if llm_type == "openai":
        return OpenAIClient()
    if llm_type == "anthropic":
        return AnthropicClient()
    if llm_type is None:
        return None
    raise ValueError(f"Unsupported llm_type: {llm_type}")


# ============================================================
# LLM Client 抽象层
# ============================================================

class BaseLLMClient:
    """LLM 客户端基类"""
    async def complete(self, prompt: str) -> str:
        raise NotImplementedError

    async def chat(self, system_prompt: str, user_message: str) -> str:
        raise NotImplementedError

    async def chat_stream(self, system_prompt: str, user_message: str):
        """默认实现：调用 chat() 一次性返回整段，作为 fallback。
        子类应覆盖为真正的流式。"""
        full = await self.chat(system_prompt, user_message)
        yield full


class OpenAIClient(BaseLLMClient):
    """OpenAI 兼容 API 客户端（支持 Novita / vLLM / 自建服务等）"""

    def __init__(self, model: str = None):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai --break-system-packages")

        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        # 允许通过 OPENAI_MODEL 环境变量覆盖（Novita / 自建服务要用）
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    async def complete(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    async def chat(self, system_prompt: str, user_message: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    async def chat_stream(self, system_prompt: str, user_message: str):
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=500,
            temperature=0.7,
            stream=True,
        )
        async for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                yield text


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API 客户端"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic --break-system-packages")

        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    async def complete(self, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    async def chat(self, system_prompt: str, user_message: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()

    async def chat_stream(self, system_prompt: str, user_message: str):
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text


# ============================================================
# 调试信息打印
# ============================================================

def print_debug(manager: ConversationManager):
    """打印 Agent 内部状态（调试用）"""
    state = manager.get_debug_state()

    print("\n" + "=" * 50)
    print("🔍 Agent Internal State")
    print("=" * 50)

    # Profile
    profile = state["profile"]
    print(f"\n📊 Profile (mean confidence: {profile['mean_confidence']:.2f}, "
          f"confident dims: {profile['confident_count']}/5)")

    dim_names = {
        "novelty_appetite": ("Loyalist", "Explorer"),
        "decision_tempo": ("Deliberate", "Impulsive"),
        "social_energy": ("Solo", "Social"),
        "sensory_cerebral": ("Cerebral", "Sensory"),
        "control_flow": ("Flow", "Control"),
    }

    for dim, (neg, pos) in dim_names.items():
        d = profile["dimensions"][dim]
        val = d["value"]
        conf = d["confidence"]
        bar_len = 20
        mid = bar_len // 2
        pos_in_bar = int((val + 1) / 2 * bar_len)
        bar = list("·" * bar_len)
        bar[pos_in_bar] = "●"
        bar_str = "".join(bar)
        conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        label = f"{neg:>10} [{bar_str}] {pos:<10}"
        print(f"  {label}  conf:[{conf_bar}] {conf:.2f}")

    # State
    print(f"\n🧠 User State: {state['user_state']}")
    print(f"💛 Rapport: {state['rapport']:.2f} ({state['rapport_level']})")
    print(f"💬 Turn: {state['turn_count']} | Questions: {state['questions_asked']}")
    print(f"🎬 Last Action: {state['last_action']}")
    print(f"📋 Onboarding: {'Complete ✅' if state['onboarding_complete'] else 'In progress...'}")

    if state.get("pending_quiz"):
        print(f"❓ Pending Quiz: {state['pending_quiz']}")

    print("=" * 50 + "\n")


# ============================================================
# 主交互循环
# ============================================================


async def main(llm_type=None, debug: bool = False):
    """主函数"""
    llm_type = resolve_llm_type(llm_type)
    llm_client = build_llm_client(llm_type)

    if llm_type == "openai":
        print("🔌 LLM: OpenAI-compatible —— 回复由大模型现场生成")
    else:
        print("🔌 LLM: Anthropic Claude —— 回复由大模型现场生成")

    # 初始化 ConversationManager
    manager = ConversationManager(llm_client=llm_client)

    print("╔" + "═" * 48 + "╗")
    print("║     🍜  Onboarding Agent Demo  🍜              ║")
    print("║                                                ║")
    print("║  输入消息开始对话，输入 /quit 退出             ║")
    print("║  输入 /debug 查看 Agent 内部状态               ║")
    print("║  输入 /profile 查看当前画像                    ║")
    print("╚" + "═" * 48 + "╝")
    print()

    # 开场白：模拟用户第一次打开
    first_response = await manager.process_message("你好")
    print(f"🤖 Agent: {first_response}\n")

    if debug:
        print_debug(manager)

    # 交互循环
    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！ 👋")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("再见！ 👋")
            break

        if user_input == "/debug":
            print_debug(manager)
            continue

        if user_input == "/profile":
            snapshot = manager.accumulator.get_snapshot()
            print(f"\n📊 Profile Snapshot:\n{json.dumps(snapshot, indent=2, ensure_ascii=False)}\n")
            # 尝试匹配 archetype
            archetype = manager.archetype_mapper.match(manager.state.profile)
            if not archetype["is_fallback"]:
                print(f"🏷️  Archetype: {archetype['emoji']} {archetype['name']}")
                print(f"   {archetype['description'][:100]}...\n")
            else:
                print(f"🏷️  Archetype: {archetype['emoji']} {archetype['name']} (还需要更多数据)\n")
            continue

        # 处理用户消息
        response = await manager.process_message(user_input)
        print(f"\n🤖 Agent: {response}\n")

        if debug:
            print_debug(manager)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Onboarding Agent Demo")
    parser.add_argument("--llm", choices=["openai", "anthropic"], default=None,
                        help="LLM provider；留空按环境变量自动探测（Anthropic 优先，其次 OpenAI-compatible）")
    parser.add_argument("--debug", action="store_true",
                        help="Show agent internal state after each turn")
    args = parser.parse_args()

    try:
        asyncio.run(main(llm_type=args.llm, debug=args.debug))
    except ValueError as exc:
        raise SystemExit(str(exc))
