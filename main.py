"""
Onboarding Agent - 可运行的交互式 demo

运行方式:
  1. 无 LLM 模式（纯规则 + 模板）:
     python main.py

  2. 接入 OpenAI 兼容 API:
     OPENAI_API_KEY=xxx OPENAI_BASE_URL=xxx python main.py --llm openai

  3. 接入 Anthropic API:
     ANTHROPIC_API_KEY=xxx python main.py --llm anthropic
"""

import asyncio
import argparse
import json
import os
import sys

# 确保 import 路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.conversation_manager import ConversationManager


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

def _resolve_llm_type(cli_choice):
    """未显式传 --llm 时按环境变量自动选；传 none 强制模板。"""
    if cli_choice == "none":
        return None
    if cli_choice in ("openai", "anthropic"):
        return cli_choice
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


async def main(llm_type=None, debug: bool = False):
    """主函数"""
    llm_type = _resolve_llm_type(llm_type)

    # 初始化 LLM Client
    llm_client = None
    if llm_type == "openai":
        llm_client = OpenAIClient()
        print("🔌 LLM: OpenAI —— 回复由大模型现场生成")
    elif llm_type == "anthropic":
        llm_client = AnthropicClient()
        print("🔌 LLM: Anthropic Claude —— 回复由大模型现场生成")
    else:
        print("🔌 Template mode —— 回复是硬编码模板，会更像问卷")
        print("   想要真 Agent 感：设置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY 再跑\n")

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
    parser.add_argument("--llm", choices=["openai", "anthropic", "none"], default=None,
                        help="LLM provider；留空按环境变量自动启；传 none 强制模板")
    parser.add_argument("--debug", action="store_true",
                        help="Show agent internal state after each turn")
    args = parser.parse_args()

    asyncio.run(main(llm_type=args.llm, debug=args.debug))
