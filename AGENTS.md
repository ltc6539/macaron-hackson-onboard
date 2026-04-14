# PROJECT KNOWLEDGE BASE

Generated: 2026-04-14

## OVERVIEW
Project: Onboarding Agent - Embodied User Profiling System
Stack: Python 3.9+, stdlib `asyncio` + `http.server` + `sqlite3`, YAML config via `pyyaml`, optional `openai` / `anthropic` async clients, dependency-free browser UI (`web/`)

This is a conversational onboarding prototype, not a static questionnaire. The runtime loop is implemented in `agent/conversation_manager.py` and is effectively:

1. detect structured answer / infer task intent
2. detect user state
3. extract profile signals
4. update rapport + fatigue + profile + profiling mode
5. select an agent action
6. generate a reply / quiz / A/B choice / archetype reveal

Key product differentiator: the system tries to learn while being useful. It can switch from active profiling to passive observation when the user is guarded, tired, or repeatedly taps "meh".

## ENTRYPOINTS
- `main.py`: CLI demo entrypoint; also defines `BaseLLMClient`, `OpenAIClient`, and `AnthropicClient`.
- `web_demo.py`: local Web demo server using `ThreadingHTTPServer`; exposes JSON + SSE endpoints and serves `web/`.
- `agent/conversation_manager.py`: canonical turn loop; read this first when behavior seems surprising.
- `tests/test_onboarding_smoke.py`: best executable spec for expected behavior.

## RUNTIME REQUIREMENTS
- The product now runs in LLM-only mode; CLI/Web startup fails fast if no provider is configured.
- `main.py` and `web_demo.py` auto-load project-root `.env`, and local project config takes precedence over inherited shell variables.
- LLM mode uses `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, or explicit `--llm anthropic/openai`; user-state detection, signal extraction, fatigue analysis, and final response generation can all use the model.
- Auto-detect precedence in both `main.py` and `web_demo.py`: Anthropic first, then OpenAI.
- OpenAI path is OpenAI-compatible, not OpenAI-only: it honors `OPENAI_BASE_URL` and `OPENAI_MODEL`.
- `prompts/response_generator.py` still contains legacy fallback helpers, but current entrypoints do not expose template mode anymore.

## STRUCTURE
- `main.py`: terminal chat loop; supports `/debug`, `/profile`, `/quit`.
- `web_demo.py`: session store, cookie-based nickname gate, persistence plumbing, `/api/*` endpoints, SSE streaming.
- `agent/action_selector.py`: action policy and quiz selection; loads `config/quiz_bank.yaml` at runtime.
- `agent/archetype_mapper.py`: hard match, soft match, fallback match, and behavioral `dan_ren` special case.
- `agent/conversation_manager.py`: orchestrates perceive/update/decide/act, tracks pending quiz/choice, task continuity, meta-prefixes, and profiling degradation.
- `core/__init__.py`: enums + dataclasses (`UserState`, `AgentAction`, `ProfilingMode`, `ProfileState`, `AgentState`, etc.).
- `core/profile_accumulator.py`: Bayesian-style weighted merging for 5 dimensions.
- `core/user_state_detector.py`: rule-first state detection with LLM fallback.
- `core/rapport_tracker.py`: rapport deltas + decay.
- `core/fatigue_tracker.py`: fatigue deltas + thresholds for soft shorten / passive / off.
- `core/signal_extractor.py`: rule/LLM profile signal extraction and task-intent inference.
- `core/persistence.py`: SQLite WAL store for `users / sessions / messages / final_profiles`.
- `prompts/response_generator.py`: LLM system prompt construction plus template fallback pools and task reply helpers.
- `config/dimensions.yaml`: design reference for the 5 dimensions; not fully authoritative at runtime.
- `config/archetypes.yaml`: runtime archetype definitions + `macaron_promises_by_dim`.
- `config/quiz_bank.yaml`: runtime quiz bank, including `q_macaron_*` capability quizzes.
- `config/agent_policy.yaml`: strategy reference; several thresholds are duplicated or hardcoded in Python.
- `web/index.html`: shell layout with chat, portrait card, nickname modal.
- `web/app.js`: fetch/SSE client, choice buttons, portrait rendering, reveal/skip/reset wiring.
- `web/styles.css`: large local-only visual layer; no framework.
- `tests/test_onboarding_smoke.py`: async behavior tests, persistence tests, and Web nickname gate tests.

## COMMANDS
| Action | Command |
|--------|---------|
| Install minimal deps | `python3 -m pip install pyyaml` |
| Install optional LLM deps | `python3 -m pip install openai anthropic` |
| Run CLI demo | `python3 main.py` |
| Run CLI in debug | `python3 main.py --debug` |
| Run Web demo | `python3 web_demo.py --port 8123` |
| Run Web demo with custom DB | `python3 web_demo.py --port 8123 --db /tmp/onboarding.db` |
| Run tests | `python3 -m unittest discover -s tests -q` |
| Compile smoke check | `python3 -m compileall main.py web_demo.py core agent prompts tests` |

## WEB API
- `GET /api/health`: health check.
- `GET /api/me`: returns current cookie-bound user or `requires_nickname`.
- `GET /api/profiles`: returns all saved final profiles; no auth, demo-only.
- `POST /api/register`: body `{ "nickname": "..." }`; creates user and sets `macaron_uid` cookie.
- `POST /api/session`: requires cookie; creates a new in-memory session and returns greeting + initial state.
- `POST /api/message`: non-streaming message endpoint.
- `POST /api/message/stream`: SSE endpoint emitting `chunk` and `done` events.
- `POST /api/skip-profiling`: downgrades profiling mode to `passive`.
- `POST /api/reveal`: force-reveals current archetype snapshot.

Important: live sessions are in-memory only (`SessionStore.sessions`). DB persistence saves history/results, but does not restore active `ConversationManager` instances after restart.

## CORE STATE MODEL
- User states: `engaged_playful`, `task_oriented`, `tentative_exploring`, `guarded`, `disengaging`, `unknown`.
- Agent actions: `ask_playful`, `ask_direct`, `offer_choice`, `observe_reaction`, `self_disclose`, `give_value`, `evaluate_user`, `do_nothing`, `show_archetype`.
- Profiling modes are one-way only: `active -> passive -> off`. `ConversationManager.set_profiling_mode()` will refuse re-enabling.
- Five dimensions in `ProfileState`:
  - `novelty_appetite`
  - `decision_tempo`
  - `social_energy`
  - `sensory_cerebral`
  - `control_flow`

## BEHAVIORAL RULES THAT MATTER
- First turn is always treated as a greeting path (`AgentAction.GIVE_VALUE` with greeting context).
- `ActionSelector` never allows consecutive `ask_*` actions.
- If profile mean confidence is high enough, or question budget is exhausted, active questioning is suppressed.
- `profiling_mode != ACTIVE` blocks `ASK_PLAYFUL`, `ASK_DIRECT`, and `OFFER_CHOICE` entirely.
- Structured A/B choices are capped at 5 unique keys; once exhausted, `offer_choice` is demoted to `give_value`.
- Pending quiz/choice buttons always get an extra `meh` option in `get_pending_choices()`.
- Two `meh` picks trigger a one-time meta-prefix and force `ProfilingMode.PASSIVE`.
- Task follow-ups are sticky: short replies like `评分高一点`, `近一点`, `你帮我选` extend `active_task` instead of starting a fresh flow.
- Legacy fallback code still exists in a few modules for testing/internal use, but CLI/Web no longer expose template mode.

## FATIGUE / RAPPORT / REVEAL THRESHOLDS
- Rapport starts at `0.25` and decays each turn.
- Fatigue starts at `0.0`.
- Fatigue thresholds from `core/fatigue_tracker.py`:
  - `>= 0.30`: elevated, shorten / soften behavior
  - `>= 0.60`: auto-switch to `PASSIVE`
  - `>= 0.80`: auto-switch to `OFF`
- Archetype reveal paths:
  - hard path: at least 3 confident dimensions (`>= 0.45`) plus timing/user-state gates
  - behavioral path: `dan_ren` if profiling skipped + short replies + low rapport
  - soft path: 2 moderate dimensions (`>= 0.35`) after enough questioning
  - time fallback: long enough conversation with some profile shape
- `force_reveal_archetype()` is used by the Web UI even if thresholds are not met.

## ARCHETYPE / PERSISTENCE NOTES
- Final profiles are saved only when archetype is revealed and is not fallback.
- `final_profiles` includes archetype fields, dimension JSON, and `macaron_promises` derived from top confident dimensions.
- `GET /api/profiles` exposes those saved final profiles verbatim; treat it as internal-only.
- Behavioral archetype `dan_ren` is matched before dimension-based archetypes in `ArchetypeMapper.match()`.

## FRONTEND NOTES
- Identity is browser-cookie based; nickname gate is required before a session can start.
- Cookie name is `macaron_uid`, `HttpOnly`, `SameSite=Lax`, 1-year max-age.
- `web/app.js` prefers `/api/message/stream`; if SSE fails, it falls back to `/api/message`.
- Portrait card renders a custom SVG rose/radar-like chart from current dimension values/confidence.
- `skipButton` is only shown while profiling mode is still `active`.
- `revealButton` becomes enabled once an archetype snapshot exists, including soft/fallback cases.

## TEST COVERAGE TO TRUST
Tests already cover more than simple smoke:
- quiz flow progression
- task-oriented reply style
- guarded low-pressure responses
- skip profiling behavior
- one-way profiling downgrade
- fatigue auto downgrade
- pending choices for frontend
- behavioral `dan_ren`
- offer-choice dedupe and fallback-text dedupe
- soft archetype reveal
- `evaluate_user` action trigger
- `meh` non-update + passive switch
- persistence roundtrip
- Web nickname gate

If you change onboarding behavior, re-run tests and manually try:
1. one quiz-heavy playful flow
2. one task-oriented flow
3. one skip-profiling / low-engagement flow

## CODING STANDARDS
- Language: Python 3.9 compatible, type hints preferred, mostly class-based modules.
- Style: concise Chinese comments/docstrings; logic split into many small methods.
- Compatibility: avoid `X | None`; use `Optional[...]`.
- Dependencies: keep it stdlib-first unless a package is clearly justified.
- UI: keep Web demo dependency-free and local-first.
- Prompts/templates: keep tone conversational, short, and product-like; avoid sterile QA wording.

## WHERE TO LOOK FIRST
- Behavior bug or flow bug: `agent/conversation_manager.py`
- “Why did it ask this?”: `agent/action_selector.py`
- Archetype mismatch: `agent/archetype_mapper.py` + `config/archetypes.yaml`
- Weird wording / fallback repetition: `prompts/response_generator.py`
- Task-intent or implicit profile bug: `core/signal_extractor.py`
- Web identity/session bug: `web_demo.py` + `core/persistence.py`
- Regression expectations: `tests/test_onboarding_smoke.py`

## WORKING NOTES
- This folder is not currently a git repository; do not assume history/branches.
- `config/dimensions.yaml` and `config/agent_policy.yaml` are partly aspirational; runtime truth often lives in Python.
- `SessionStore` owns in-memory managers and a `_profiles_saved` set; restarting the process resets those ephemeral structures.
- `web_demo.py` silently swallows normal HTTP logging by overriding `log_message()`.
- The Web demo is local-only and intentionally lacks auth, CSRF protection, or production deployment hardening.
- User-facing docs and demos should assume LLM mode; if you touch legacy fallback code, treat it as internal/testing-only behavior.
- Existing SQLite files may already be present under `data/`; avoid assuming a fresh DB.
- If onboarding logic changes, update both docs (`README.md`, `AGENTS.md`) and tests together.