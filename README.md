# Onboarding Agent - Embodied User Profiling System

一个本地可运行的 onboarding 原型：它不把用户拉进固定问卷，而是在自然对话里一边服务、一边观察、一边更新画像，最后给出一个可解释的 archetype（人格画像标签）。

这个项目的重点不是把题目包装得更好看，而是验证一套更像 Agent 的 onboarding 回路：

1. 判断用户当前状态
2. 提取显式/隐式偏好信号
3. 更新 rapport（信任）和 fatigue（被问烦程度）
4. 决定下一步是继续问、先给价值、还是收住
5. 生成一句自然回复，或者抛出一个低压力的小选择题

## 这和普通问卷有什么不同

核心原则：

- 知进退：先判断用户是投入、带任务、试探、戒备，还是准备 disengage
- 先给后拿：低 rapport 时先给价值，不急着采信息
- 最好的问题不像问题：能在任务里推断，就不做生硬问卷
- 见好就收：画像够用了就揭示 archetype，不无限追问

运行上，系统会在 `ConversationManager` 里走完整的：

`Perceive -> Update -> Decide -> Act`

- `Perceive`：识别 quiz/choice 回答、检测用户状态、抽取画像信号、识别任务意图
- `Update`：更新 `rapport` / `fatigue` / `profile` / `profiling_mode`
- `Decide`：选择 `ask_playful` / `offer_choice` / `give_value` / `evaluate_user` / `show_archetype` 等行动
- `Act`：生成回复、挂起下一题、或揭示画像

## 两种运行模式（非常重要）

项目有两种体验，差别很大：

### 1. Template mode

没有设置 API key 时，回复来自硬编码模板：

- 用户状态机仍然会跑
- `rapport`、`fatigue`、`profile` 仍然会更新
- quiz、A/B 选择题、跳过画像、最终 archetype 仍然能触发
- 但回复文案不会真正根据上下文灵活变化

所以它更适合：

- 本地离线开发
- 验证状态机和前端交互
- 跑 smoke tests

不适合用来证明“这个 onboarding 很像 Agent”。

### 2. LLM mode

设置 API key 后，Anthropic Claude 或 OpenAI 兼容接口会参与：

- 用户状态检测可走 LLM 精细判断
- 画像信号可做 LLM 抽取
- fatigue 可做更细的被问烦检测
- 回复由模型根据 action、rapport、profile、task context 现场生成

这才是更完整的体验。

自动检测规则：

- `ANTHROPIC_API_KEY` 优先
- 否则看 `OPENAI_API_KEY`
- 可显式传 `--llm none` 强制模板模式

## 快速开始

### 安装依赖

最小依赖：

```bash
python3 -m pip install pyyaml
```

如果要启用 LLM：

```bash
python3 -m pip install openai anthropic
```

### 跑 CLI demo

```bash
python3 main.py
```

可选参数：

```bash
python3 main.py --debug
python3 main.py --llm none
python3 main.py --llm openai
python3 main.py --llm anthropic
```

CLI 内置命令：

- `/debug`：打印内部状态
- `/profile`：打印当前 5 维画像快照
- `/quit`：退出

### 跑 Web demo

```bash
python3 web_demo.py --port 8123
```

可选参数：

```bash
python3 web_demo.py --host 127.0.0.1 --port 8123
python3 web_demo.py --llm none
python3 web_demo.py --llm openai
python3 web_demo.py --llm anthropic
python3 web_demo.py --db data/onboarding.db
```

打开浏览器访问：

```text
http://127.0.0.1:8123
```

### 启用 OpenAI 兼容接口

`main.py` 使用 `AsyncOpenAI`，并支持兼容接口：

```bash
export OPENAI_API_KEY=xxx
export OPENAI_BASE_URL=https://your-compatible-endpoint/v1
export OPENAI_MODEL=gpt-4o-mini
python3 web_demo.py --llm openai
```

### 启用 Anthropic

```bash
export ANTHROPIC_API_KEY=xxx
python3 web_demo.py --llm anthropic
```

默认模型是 `claude-sonnet-4-20250514`。

## Web demo 实际做了什么

Web 版不是静态页面，而是一个完整的本地 demo：

- 首次进入先要求输入昵称
- 后端为用户生成匿名 `user_id`
- 浏览器通过 `macaron_uid` cookie 维持身份
- 每次点“重开”都会新建一个 session
- 聊天回复优先走 SSE 流式接口，失败时回退到普通 JSON 接口
- 右侧画像卡实时展示五维信号、soft/hard archetype、以及 `Macaron promises`

前端还内置了三个关键按钮：

- `先跳过`：把 `profiling_mode` 从 `active` 降到 `passive`
- `查看我的画像`：强制 reveal 当前 archetype 快照
- `重开`：新建一个新 session

### 关于昵称和身份

昵称不是第三方登录，只是本地 demo 身份：

- 存在 SQLite 的 `users` 表里
- 通过 `macaron_uid` cookie 绑定浏览器
- 换浏览器/清 cookie 后会重新要求昵称

## 项目结构

```text
main.py                     CLI 入口；也定义了 OpenAI / Anthropic 客户端
web_demo.py                 本地 Web server + JSON API + SSE 接口
agent/
  action_selector.py        决策策略：当前该问、该收、还是该评价
  archetype_mapper.py       5 维画像 -> archetype，含软匹配和行为型淡人
  conversation_manager.py   顶层编排器，串起完整 turn loop
core/
  __init__.py               共享枚举和 dataclass
  profile_accumulator.py    Bayesian 风格画像累积
  user_state_detector.py    用户状态检测（规则优先，LLM 可选）
  rapport_tracker.py        信任更新
  fatigue_tracker.py        疲劳/被问烦更新
  signal_extractor.py       规则/LLM 画像信号提取 + task intent 检测
  persistence.py            SQLite 持久化（users / sessions / messages / final_profiles）
prompts/
  response_generator.py     system prompt + 模板 fallback + action-specific 回复逻辑
config/
  dimensions.yaml           五维模型说明（更多是设计文档）
  archetypes.yaml           archetype 定义、匹配配置、macaron promises
  quiz_bank.yaml            趣味题库
  agent_policy.yaml         策略说明（部分逻辑已硬编码进 Python）
web/
  index.html                Web shell
  app.js                    前端状态管理、聊天流、画像卡、按钮事件
  styles.css                本地 demo UI 样式
tests/
  test_onboarding_smoke.py  smoke + 行为回归 + 持久化 + Web gate 测试
```

## 核心运行机制

### 1. 用户状态不是情感分类，而是交互状态

系统会把用户粗分为：

- `engaged_playful`
- `task_oriented`
- `tentative_exploring`
- `guarded`
- `disengaging`
- `unknown`

这直接影响允许动作。例如：

- `guarded` 时更偏 `give_value` / `observe_reaction`
- `task_oriented` 时优先接任务，不闲聊
- `engaged_playful` 时才更适合出趣味题

### 2. 画像是 5 维连续值，不是离散 MBTI

每个维度都维护：

- `value`: `-1.0 ~ 1.0`
- `confidence`: `0.0 ~ 1.0`
- `signal_count`

五个维度是：

- `novelty_appetite`
- `decision_tempo`
- `social_energy`
- `sensory_cerebral`
- `control_flow`

更新方式是 Bayesian 风格融合：

- quiz 信号权重大
- 对话推断权重较低
- confidence 单调上升，但越往后越难提升

### 3. fatigue 会自动让系统收手

`fatigue` 是项目里很关键的一条线：

- `>= 0.30`：开始缩短、减少追问
- `>= 0.60`：自动切 `profiling_mode = passive`
- `>= 0.80`：自动切 `profiling_mode = off`

此外，前端所有 quiz / A/B 选择题都附带一个 `meh` 按钮：

- 文案是“对这题没感觉”
- 不回写画像信号
- 连续两次 `meh` 会自动切到 `passive`

### 4. profiling_mode 只允许单向降级

画像采集模式只有三档：

- `active`：可以主动出题、A/B choice、边聊边推断
- `passive`：不主动问，但仍继续从自然对话推断
- `off`：完全关闭采集

注意：这是单向的。

系统允许：

- `active -> passive -> off`

不允许：

- `passive -> active`
- `off -> active`

### 5. archetype 不只靠“问够三题”触发

目前有几条 reveal 路径：

- 硬匹配：至少 3 维 confidence 达阈值
- 软匹配：问够一定轮次后，2 维中等 confidence 也能先给 soft match
- 行为型匹配：跳过画像 + 极短回复 + 低 rapport 时命中 `dan_ren`
- 手动 reveal：前端可直接触发 `force_reveal_archetype()`

最终 archetype 定义在 `config/archetypes.yaml`。

## 当前已有的产品行为

这版原型已经包含一些很“Agent”的细节，而不是单纯问答：

- 连续两轮 guarded 会触发更克制的元句前缀
- fatigue 跨阈值时，回复开头会出现一次性 meta prefix
- `rapport` 足够高、画像初具轮廓时，会触发 `evaluate_user` 主动评价用户
- 任务型消息会被串成连续 `active_task` 上下文，而不是每轮都当新任务
- A/B choice 有去重，问完 5 条结构化小题后不会复读
- fallback 模板池也做了去重，避免“老地方还是新店”反复出现

## 数据持久化

Web demo 使用 SQLite，默认路径：

```text
data/onboarding.db
```

后端采用 stdlib `sqlite3` + WAL 模式，主要表：

- `users`
- `sessions`
- `messages`
- `final_profiles`

持久化策略：

- 注册昵称时写 `users`
- 新建会话时写 `sessions`
- 每一轮 user/agent 消息都写 `messages`
- archetype 真正揭示且不是 fallback 时，写 `final_profiles`

注意两点：

1. 持久化的是历史和最终画像，不是运行中的内存会话对象
2. 服务器重启后，旧 session message 还在，但活跃 `ConversationManager` 不会自动恢复

也就是说，这个 demo 目前是“持久化结果 + 内存会话”。

## HTTP API

Web demo 暴露的是本地 JSON/SSE API：

| Method | Path | 用途 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/me` | 查看当前 cookie 对应用户 |
| `GET` | `/api/profiles` | 拉取所有已保存最终画像，无鉴权，demo only |
| `POST` | `/api/register` | 注册昵称并写 cookie |
| `POST` | `/api/session` | 新建会话并返回首条欢迎语 |
| `POST` | `/api/message` | 非流式发消息 |
| `POST` | `/api/message/stream` | SSE 流式发消息 |
| `POST` | `/api/skip-profiling` | 切到 `passive` |
| `POST` | `/api/reveal` | 强制揭示当前 archetype |

### 示例：注册昵称

```bash
curl -i -X POST http://127.0.0.1:8123/api/register \
  -H 'Content-Type: application/json' \
  -d '{"nickname":"阿禾"}'
```

### 示例：创建 session

```bash
curl -i -X POST http://127.0.0.1:8123/api/session \
  -H 'Cookie: macaron_uid=YOUR_USER_ID'
```

### 示例：普通消息

```bash
curl -X POST http://127.0.0.1:8123/api/message \
  -H 'Content-Type: application/json' \
  -H 'Cookie: macaron_uid=YOUR_USER_ID' \
  -d '{"session_id":"SESSION_ID","message":"帮我找个适合两个人吃饭的地方"}'
```

### 示例：流式消息

```bash
curl -N -X POST http://127.0.0.1:8123/api/message/stream \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION_ID","message":"来个测试"}'
```

SSE 事件形态：

- `{"type":"chunk","text":"..."}`
- `{"type":"done","reply":"...","state":{...}}`

## 测试

运行测试：

```bash
python3 -m unittest discover -s tests -q
```

快速编译检查：

```bash
python3 -m compileall main.py web_demo.py core agent prompts tests
```

当前 smoke tests 覆盖的重点包括：

- quiz flow 是否持续推进 onboarding
- task-oriented 消息是否转成任务型回复
- `skip profiling` 后是否停止主动提问
- `profiling_mode` 是否只允许单向降级
- fatigue 是否会自动切到 passive
- 前端是否能拿到 pending choices
- `dan_ren` 行为型 archetype 是否可达
- offer choice 和 fallback 文案是否去重
- soft match / `evaluate_user` / `meh` 降级逻辑
- SQLite roundtrip
- Web 端昵称 gate

## 开发注意事项

- `config/quiz_bank.yaml` 和 `config/archetypes.yaml` 是运行时真配置
- `config/dimensions.yaml`、`config/agent_policy.yaml` 目前更偏设计说明；很多阈值实际写死在 Python 里
- Web demo 是本地原型，没有 auth、没有 session 恢复、没有生产级部署逻辑
- `GET /api/profiles` 没鉴权，只适合内网或本地 demo
- 如果改了 onboarding 行为，至少重新跑：
  - `python3 -m unittest discover -s tests -q`
  - 一次 quiz 流
  - 一次 task-oriented 流

## 推荐先看哪里

如果你第一次接手这个项目，建议按这个顺序读：

1. `agent/conversation_manager.py`
2. `agent/action_selector.py`
3. `prompts/response_generator.py`
4. `core/user_state_detector.py`
5. `core/signal_extractor.py`
6. `web_demo.py`
7. `tests/test_onboarding_smoke.py`

这样最容易建立“状态如何流动、动作如何决策、前端如何接入”的完整心智模型。