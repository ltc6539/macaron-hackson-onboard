const chatLog = document.getElementById('chatLog');
const composer = document.getElementById('composer');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const resetButton = document.getElementById('resetButton');
const skipButton = document.getElementById('skipButton');
const revealButton = document.getElementById('revealButton');
const choicePanel = document.getElementById('choicePanel');
const bubbleTemplate = document.getElementById('bubbleTemplate');
const portraitContent = document.getElementById('portraitContent');
const greetingLine = document.getElementById('greetingLine');

const nicknameModal = document.getElementById('nicknameModal');
const nicknameForm = document.getElementById('nicknameForm');
const nicknameInput = document.getElementById('nicknameInput');
const nicknameError = document.getElementById('nicknameError');

// 玫瑰图用的中文短标 + 极标签（正极 / 负极）
const ROSE_DIMS = [
  { key: 'novelty_appetite',  label: '新鲜', positive: '探索', negative: '忠诚' },
  { key: 'decision_tempo',    label: '决策', positive: '直觉', negative: '审慎' },
  { key: 'social_energy',     label: '社交', positive: '社交', negative: '独处' },
  { key: 'sensory_cerebral',  label: '感性', positive: '感性', negative: '理性' },
  { key: 'control_flow',      label: '掌控', positive: '掌控', negative: '随遇' },
];

let sessionId = null;
let nickname = '';
let pendingBubble = null;

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ============================================================
// 聊天 bubble + pending
// ============================================================

function appendBubble(role, text) {
  const fragment = bubbleTemplate.content.cloneNode(true);
  const bubble = fragment.querySelector('.bubble');
  bubble.classList.add(role);
  fragment.querySelector('.bubble-role').textContent = role === 'agent' ? 'Agent' : (nickname || 'You');
  fragment.querySelector('.bubble-body').innerHTML = escapeHtml(text).replace(/\n/g, '<br>');
  chatLog.appendChild(fragment);
  chatLog.scrollTop = chatLog.scrollHeight;
  return bubble;
}

function showTypingBubble() {
  pendingBubble = appendBubble('agent', '正在顺一下你的意思...');
  pendingBubble.classList.add('typing');
}

function clearTypingBubble() {
  if (pendingBubble) {
    pendingBubble.remove();
    pendingBubble = null;
  }
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  messageInput.disabled = isBusy;
  sendButton.textContent = isBusy ? '发送中...' : '发送';
  if (skipButton) skipButton.disabled = isBusy;
  if (revealButton && !revealButton.classList.contains('is-ready')) {
    revealButton.disabled = true;
  } else if (revealButton) {
    revealButton.disabled = isBusy;
  }
  choicePanel.querySelectorAll('button').forEach((btn) => {
    btn.disabled = isBusy;
  });
}

// ============================================================
// 选项按钮
// ============================================================

function renderChoices(choices) {
  choicePanel.innerHTML = '';
  if (!choices || !Array.isArray(choices.options) || choices.options.length === 0) {
    choicePanel.classList.add('hidden');
    return;
  }
  choicePanel.classList.remove('hidden');

  if (choices.prompt) {
    const hint = document.createElement('p');
    hint.className = 'choice-hint';
    hint.textContent = choices.kind === 'quiz' ? '点一个选项直接回答：' : '或者点一下就行：';
    choicePanel.appendChild(hint);
  }

  choices.options.forEach((opt) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    const isMeh = opt.value === 'meh';
    btn.className = isMeh ? 'choice-card choice-meh' : 'choice-card';
    if (isMeh) {
      btn.innerHTML = `<span class="meh-label">${escapeHtml(opt.label)}</span>`;
    } else {
      btn.innerHTML = `<strong>${escapeHtml(opt.value)}</strong><span>${escapeHtml(opt.label)}</span>`;
    }
    btn.addEventListener('click', async () => {
      if (sendButton.disabled) return;
      choicePanel.classList.add('hidden');
      choicePanel.innerHTML = '';
      await sendMessage(opt.value, { viaButton: true });
    });
    choicePanel.appendChild(btn);
  });
}

function updateSkipVisibility(state) {
  if (!skipButton) return;
  const mode = state && state.profiling_mode;
  if (!mode || mode === 'active') {
    skipButton.classList.remove('hidden');
  } else {
    skipButton.classList.add('hidden');
  }
}

// ============================================================
// 玫瑰图：纯 SVG，零依赖
// ============================================================

function buildRoseChartSvg(dimensions) {
  const W = 280;
  const H = 280;
  const CX = W / 2;
  const CY = H / 2;
  const R = 98;   // 最大半径
  const labelR = R + 20;

  // 5 个轴的角度（-90 起，顺时针 72 度间隔），新鲜朝正上
  const axisAngles = ROSE_DIMS.map((_, i) => -Math.PI / 2 + (i * 2 * Math.PI) / 5);

  const pt = (r, ang) => ({
    x: CX + r * Math.cos(ang),
    y: CY + r * Math.sin(ang),
  });

  // 背景圆环刻度（25/50/75/100%）
  const gridCircles = [0.25, 0.5, 0.75, 1].map(
    (k) => `<circle class="rose-grid" cx="${CX}" cy="${CY}" r="${R * k}" />`
  ).join('');

  // 轴线 + 轴标签
  const axesSvg = axisAngles.map((ang, i) => {
    const end = pt(R, ang);
    const lab = pt(labelR, ang);
    return `
      <line class="rose-axis" x1="${CX}" y1="${CY}" x2="${end.x}" y2="${end.y}"/>
      <text class="rose-axis-label" x="${lab.x}" y="${lab.y}"
            text-anchor="middle" dominant-baseline="middle">${escapeHtml(ROSE_DIMS[i].label)}</text>
    `;
  }).join('');

  // confidence 包络：每个维度轴上 confidence * R 处
  const confPoints = axisAngles.map((ang, i) => {
    const d = dimensions[ROSE_DIMS[i].key] || { confidence: 0 };
    const r = R * Math.max(0.02, Number(d.confidence) || 0);
    const p = pt(r, ang);
    return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
  }).join(' ');

  // value shape：半径 = ((value+1)/2) * (confidence * R)
  // 不 confidence 加权 value 会被夸大
  const valPoints = axisAngles.map((ang, i) => {
    const d = dimensions[ROSE_DIMS[i].key] || { value: 0, confidence: 0 };
    const conf = Math.max(0.02, Number(d.confidence) || 0);
    const valNorm = ((Number(d.value) || 0) + 1) / 2;
    const r = R * conf * valNorm;
    const p = pt(r, ang);
    return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
  }).join(' ');

  return `
    <svg id="roseChart" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" aria-label="Profile rose chart">
      ${gridCircles}
      ${axesSvg}
      <polygon class="rose-confidence" points="${confPoints}" />
      <polygon class="rose-value" points="${valPoints}" />
    </svg>
  `;
}

// 每个维度根据 value 符号 + 置信度生成一条短 trait
function buildTraitLines(dimensions) {
  return ROSE_DIMS.map(({ key, label, positive, negative }) => {
    const d = dimensions[key] || { value: 0, confidence: 0 };
    const conf = Number(d.confidence) || 0;
    const lowConf = conf < 0.30;
    const pole = lowConf
      ? '还没看清'
      : (Number(d.value) >= 0 ? positive : negative);
    const liClass = lowConf ? 'low-confidence' : '';
    return `
      <li class="${liClass}">
        <span class="dim-dot"></span>
        <span class="dim-label">${escapeHtml(label)}</span>
        <span class="pole">· ${escapeHtml(pole)}</span>
      </li>
    `;
  }).join('');
}

// ============================================================
// Archetype 卡片（升级版）
// ============================================================

function renderPortrait(state) {
  const archetype = (state && state.archetype) || null;
  const dimensions = (state && state.profile && state.profile.dimensions) || {};
  const rose = buildRoseChartSvg(dimensions);
  const traits = buildTraitLines(dimensions);
  const nickPrefix = nickname ? `${escapeHtml(nickname)}，` : '';

  const fallback = !archetype || !!archetype.is_fallback;
  const softMatch = !fallback && !!archetype.soft_match;

  // reveal 按钮：archetype 存在（哪怕 fallback 也能点看当前快照）
  if (revealButton) {
    revealButton.disabled = !archetype;
    if (archetype) revealButton.classList.add('is-ready');
    else revealButton.classList.remove('is-ready');
  }

  const promises = (state && state.macaron_promises) || [];
  const promisesHtml = promises.length
    ? `
      <div class="macaron-promises">
        <h4>${softMatch ? 'Macaron 可能会这样配合你' : 'Macaron 会怎么配合你'}</h4>
        <ul>
          ${promises.map((p) => `<li>${escapeHtml(p)}</li>`).join('')}
        </ul>
      </div>
    `
    : '';

  let headerLine;
  if (fallback) {
    headerLine = `<h3>${nickPrefix}画像还在显影中 🧭</h3>`;
  } else if (softMatch) {
    headerLine = `<h3><span class="nick-prefix">${nickPrefix}大概是</span> ${escapeHtml(archetype.emoji || '')} ${escapeHtml(archetype.name || '')}</h3>`;
  } else {
    headerLine = `<h3><span class="nick-prefix">${nickPrefix}你是</span> ${escapeHtml(archetype.emoji || '')} ${escapeHtml(archetype.name || '')}</h3>`;
  }

  const descText = archetype ? escapeHtml(archetype.description || '').replace(/\n/g, '<br>') : '';
  const promiseText = archetype ? escapeHtml(archetype.agent_promise || '') : '';
  const hint = fallback
    ? '<p class="soft-hint">继续聊几句我就能说出来你是什么样的。</p>'
    : (softMatch
        ? '<p class="soft-hint">再聊几句我会更确定 —— 这个判断会慢慢变。</p>'
        : '');

  const cardClass = fallback ? 'fallback' : (softMatch ? 'soft' : 'match');
  portraitContent.innerHTML = `
    <article class="archetype-card ${cardClass}">
      <div class="archetype-header">${headerLine}</div>
      <div class="rose-wrap">${rose}</div>
      <ul class="traits-list">${traits}</ul>
      ${descText ? `<p>${descText}</p>` : ''}
      ${promiseText ? `<div class="agent-promise-block">${promiseText}</div>` : ''}
      ${promisesHtml}
      ${hint}
    </article>
  `;
}

// ============================================================
// 侧栏 profile 条 + 顶部 chips
// ============================================================

function updateState(state) {
  renderPortrait(state);
  renderChoices(state.choices);
  updateSkipVisibility(state);
}

// ============================================================
// 昵称 modal
// ============================================================

function showNicknameModal() {
  nicknameModal.classList.remove('hidden');
  setTimeout(() => nicknameInput.focus(), 50);
}

function hideNicknameModal() {
  nicknameModal.classList.add('hidden');
}

function setGreetingLine(nick) {
  if (!greetingLine) return;
  if (nick) {
    greetingLine.textContent = `Hi, ${nick}`;
    greetingLine.classList.remove('hidden');
  } else {
    greetingLine.classList.add('hidden');
  }
}

// ============================================================
// 登录 / 会话流
// ============================================================

async function bootstrap() {
  // 先问 /api/me 看 cookie 里有没有 user
  const meResp = await fetch('/api/me', { credentials: 'include' });
  const meData = await meResp.json();
  if (meData && meData.user) {
    nickname = meData.user.nickname || '';
    setGreetingLine(nickname);
    await createSession();
    return;
  }
  showNicknameModal();
}

async function submitNickname(raw) {
  const cleaned = String(raw || '').trim();
  if (!cleaned) {
    showNicknameError('名字不能空着');
    return;
  }
  const resp = await fetch('/api/register', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nickname: cleaned }),
  });
  if (!resp.ok) {
    showNicknameError('保存失败，换个名字试试');
    return;
  }
  const data = await resp.json();
  nickname = data.nickname || cleaned;
  setGreetingLine(nickname);
  hideNicknameModal();
  await createSession();
}

function showNicknameError(msg) {
  if (!nicknameError) return;
  nicknameError.textContent = msg;
  nicknameError.classList.remove('hidden');
}

async function createSession() {
  setBusy(true);
  clearTypingBubble();
  chatLog.innerHTML = '';
  portraitContent.innerHTML = '';

  try {
    const response = await fetch('/api/session', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reset: true }),
    });
    if (response.status === 401) {
      showNicknameModal();
      return;
    }

    let payload = {};
    try {
      payload = await response.json();
    } catch {
      payload = {};
    }

    if (!response.ok) {
      const message = payload.error || '初始化会话失败，请检查后端日志或 LLM 配置。';
      appendBubble('agent', `启动失败：${message}`);
      return;
    }

    sessionId = payload.session_id;
    if (payload.nickname) {
      nickname = payload.nickname;
      setGreetingLine(nickname);
    }

    appendBubble('agent', payload.reply);
    updateState(payload.state);
    messageInput.focus();
  } catch (err) {
    appendBubble('agent', `启动失败：${err.message || err}`);
  } finally {
    setBusy(false);
  }
}

async function sendMessage(message, opts = {}) {
  if (!sessionId) return;
  const viaButton = !!opts.viaButton;

  appendBubble('user', message);
  setBusy(true);

  // 建一个 agent bubble，先用 thinking 占位动效；首个 chunk 到了就替换
  const bubble = appendBubble('agent', '');
  const body = bubble.querySelector('.bubble-body');
  bubble.classList.add('thinking');
  body.innerHTML = `
    <span class="thinking-indicator">
      <span class="thinking-label">思考中</span>
      <span class="thinking-dot"></span>
      <span class="thinking-dot"></span>
      <span class="thinking-dot"></span>
    </span>
  `;
  chatLog.scrollTop = chatLog.scrollHeight;

  let accumulated = '';
  let finalState = null;
  let firstChunkSeen = false;

  const swapToStreaming = () => {
    if (firstChunkSeen) return;
    firstChunkSeen = true;
    bubble.classList.remove('thinking');
    bubble.classList.add('streaming');
    body.innerHTML = '';
  };

  const renderAccumulated = () => {
    swapToStreaming();
    body.innerHTML = escapeHtml(accumulated).replace(/\n/g, '<br>');
    chatLog.scrollTop = chatLog.scrollHeight;
  };

  try {
    const response = await fetch('/api/message/stream', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message, via_button: viaButton }),
    });

    if (!response.ok || !response.body) {
      // 流接不上，降级到非流式端点
      const fallback = await fetch('/api/message', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message, via_button: viaButton }),
      });
      const payload = await fallback.json();
      accumulated = payload.reply || '';
      renderAccumulated();
      finalState = payload.state;
    } else {
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE: 按 \n\n 分割事件
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';
        for (const raw of events) {
          const line = raw.trim();
          if (!line.startsWith('data:')) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;
          let ev;
          try { ev = JSON.parse(jsonStr); } catch { continue; }
          if (ev.type === 'chunk') {
            accumulated += ev.text || '';
            renderAccumulated();
          } else if (ev.type === 'done') {
            if (ev.reply) {
              accumulated = ev.reply;
              renderAccumulated();
            }
            finalState = ev.state;
          } else if (ev.type === 'error') {
            accumulated += '\n(出错了：' + (ev.message || '') + ')';
            renderAccumulated();
          }
        }
      }
    }
  } catch (err) {
    accumulated = accumulated || `(流失败: ${err.message || err})`;
    renderAccumulated();
  } finally {
    // 如果整条流什么都没吐（例如远端 500），把 thinking 占位也清掉
    if (!firstChunkSeen) {
      bubble.classList.remove('thinking');
      bubble.classList.add('streaming');
      body.innerHTML = accumulated
        ? escapeHtml(accumulated).replace(/\n/g, '<br>')
        : '(没有收到回复)';
    }
    bubble.classList.remove('streaming');
    if (finalState) updateState(finalState);
    setBusy(false);
    messageInput.focus();
  }
}

// ============================================================
// Events
// ============================================================

composer.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  messageInput.value = '';
  await sendMessage(message);
});

messageInput.addEventListener('keydown', async (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (!message || sendButton.disabled) return;

    messageInput.value = '';
    await sendMessage(message);
  }
});

resetButton.addEventListener('click', async () => {
  await createSession();
});

if (skipButton) {
  skipButton.addEventListener('click', async () => {
    if (!sessionId || skipButton.disabled) return;
    setBusy(true);
    try {
      const response = await fetch('/api/skip-profiling', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (response.ok) {
        const payload = await response.json();
        appendBubble('agent', '好，那我们直接开始——有什么想吃的、想做的，直接说就行。');
        updateState(payload.state);
      }
    } finally {
      setBusy(false);
      messageInput.focus();
    }
  });
}

if (revealButton) {
  revealButton.addEventListener('click', async () => {
    if (!sessionId || revealButton.disabled) return;
    setBusy(true);
    try {
      const response = await fetch('/api/reveal', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (response.ok) {
        const payload = await response.json();
        updateState(payload.state);
      }
    } finally {
      setBusy(false);
    }
  });
}

if (nicknameForm) {
  nicknameForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    await submitNickname(nicknameInput.value);
  });
}

window.addEventListener('load', async () => {
  await bootstrap();
});
