// LUCID browser demo client. Protocol: docs/formats.md, "WebSocket protocol".

const BUTTON_BITS = { left: 0, right: 1, up: 2, down: 3, jump: 4, action_a: 5, action_b: 6 };

const KEY_MAP = {
  ArrowLeft: "left", KeyA: "left",
  ArrowRight: "right", KeyD: "right",
  ArrowUp: "up", KeyW: "up",
  ArrowDown: "down", KeyS: "down",
  Space: "jump",
  KeyZ: "action_a",
  KeyX: "action_b",
};

const SAMPLER_STEPS = [1, 2, 3, 4, 6, 8];
const SEND_INTERVAL_MS = 1000 / 20;
const CHANNEL_DREAM = 0;
const CHANNEL_REAL = 1;

const $ = (id) => document.getElementById(id);

const el = {
  status: $("status"),
  stage: $("stage"),
  dream: $("dream"),
  real: $("real"),
  realWrap: $("real-wrap"),
  dreamLabel: $("dream-label"),
  startScreen: $("start-screen"),
  connectingScreen: $("connecting-screen"),
  busyScreen: $("busy-screen"),
  busyMax: $("busy-max"),
  disconnectedScreen: $("disconnected-screen"),
  disconnectReason: $("disconnect-reason"),
  start: $("start"),
  busyRetry: $("busy-retry"),
  statFps: $("stat-fps"),
  statMs: $("stat-ms"),
  statSteps: $("stat-steps"),
  games: $("games"),
  steps: $("steps"),
  reset: $("reset"),
  sbsWrap: $("sbs-wrap"),
  sbs: $("sbs"),
  experiment: $("experiment"),
  blendOn: $("blend-on"),
  blendA: $("blend-a"),
  blendB: $("blend-b"),
  blendAlpha: $("blend-alpha"),
  switchLive: $("switch-live"),
  error: $("error"),
  touch: $("touch"),
};

const state = {
  ws: null,
  hello: null,
  playing: false,
  busy: false,
  game: null,
  chosenSteps: null,
  sideBySide: false,
  heldKeys: new Set(),
  touchHeld: new Map(), // button name -> Set of pointer ids
  lastSentButtons: -1,
  sendTimer: 0,
};

// ---------------------------------------------------------------- connection

function wsUrl() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}/ws`;
}

function setStatus(kind, text) {
  el.status.dataset.state = kind;
  el.status.textContent = text;
}

function showScreen(which) {
  el.startScreen.hidden = which !== "start";
  el.connectingScreen.hidden = which !== "connecting";
  el.busyScreen.hidden = which !== "busy";
  el.disconnectedScreen.hidden = which !== "disconnected";
}

function connect() {
  if (state.ws && state.ws.readyState <= WebSocket.OPEN) return;
  state.busy = false;
  state.hello = null;
  showScreen("connecting");
  setStatus("connecting", "connecting");

  const sock = new WebSocket(wsUrl());
  sock.binaryType = "arraybuffer";
  state.ws = sock;

  sock.onmessage = (event) => {
    if (sock !== state.ws) return;
    if (typeof event.data === "string") {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      handleMessage(msg);
    } else {
      handleFrame(event.data);
    }
  };

  sock.onclose = (event) => {
    if (sock !== state.ws) return;
    state.ws = null;
    updateSending();
    clearStats();
    if (state.busy) {
      setStatus("busy", "server busy");
      showScreen("busy");
    } else {
      setStatus("disconnected", "disconnected");
      el.disconnectReason.textContent = event.reason || "";
      showScreen("disconnected");
    }
  };
}

function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
  }
}

function handleMessage(msg) {
  switch (msg.t) {
    case "hello":
      onHello(msg);
      break;
    case "stats":
      el.statFps.textContent = Number(msg.fps).toFixed(1);
      el.statMs.textContent = Number(msg.frame_ms).toFixed(1);
      el.statSteps.textContent = String(msg.sampler_steps);
      markSteps(msg.sampler_steps);
      break;
    case "busy":
      state.busy = true;
      el.busyMax.textContent = String(msg.max_players);
      break;
    case "error":
      showError(msg.msg);
      break;
  }
}

function onHello(msg) {
  state.hello = msg;
  const games = msg.games || [];
  if (!games.length) {
    showError("The server has no games loaded.");
    return;
  }
  if (!games.some((g) => g.name === state.game)) state.game = games[0].name;

  buildGameButtons(games);
  buildBlendSelects(games);
  el.experiment.hidden = games.length < 2;

  const [w, h] = msg.resolution || [el.dream.width, el.dream.height];
  setCanvasSize(el.dream, w, h);
  setCanvasSize(el.real, w, h);

  el.sbsWrap.hidden = !msg.side_by_side_available;
  if (!msg.side_by_side_available) state.sideBySide = false;
  el.sbs.checked = state.sideBySide;
  applySideBySideLayout();

  markSteps(msg.sampler_steps);
  el.statSteps.textContent = String(msg.sampler_steps);

  setStatus("connected", "connected");
  showScreen("none");

  // Restore choices made before a reconnect, then start the dream.
  if (state.chosenSteps !== null && state.chosenSteps !== msg.sampler_steps) {
    send({ t: "config", sampler_steps: state.chosenSteps });
  }
  send({ t: "reset", game: state.game });
  if (state.sideBySide) send({ t: "side_by_side", on: true });
  if (el.blendOn.checked && !el.experiment.hidden) sendBlend();

  state.lastSentButtons = -1;
  updateSending();
}

// --------------------------------------------------------------------- input

function currentButtons() {
  const held = new Set();
  for (const code of state.heldKeys) held.add(KEY_MAP[code]);
  for (const [name, ids] of state.touchHeld) if (ids.size) held.add(name);
  if (held.has("left") && held.has("right")) {
    held.delete("left");
    held.delete("right");
  }
  if (held.has("up") && held.has("down")) {
    held.delete("up");
    held.delete("down");
  }
  let bits = 0;
  for (const name of held) bits |= 1 << BUTTON_BITS[name];
  return bits;
}

function canSend() {
  return (
    state.playing &&
    state.hello !== null &&
    state.ws !== null &&
    state.ws.readyState === WebSocket.OPEN &&
    !document.hidden
  );
}

function sendAct() {
  const buttons = currentButtons();
  state.lastSentButtons = buttons;
  send({ t: "act", buttons, look: [0, 0] });
}

function onInputChange() {
  if (canSend() && currentButtons() !== state.lastSentButtons) sendAct();
}

function updateSending() {
  if (canSend()) {
    if (!state.sendTimer) state.sendTimer = setInterval(sendAct, SEND_INTERVAL_MS);
  } else if (state.sendTimer) {
    clearInterval(state.sendTimer);
    state.sendTimer = 0;
  }
}

function releaseAllInput() {
  state.heldKeys.clear();
  for (const ids of state.touchHeld.values()) ids.clear();
  for (const b of el.touch.querySelectorAll("button.held")) b.classList.remove("held");
}

function isTextLikeTarget(target) {
  return target instanceof Element && target.matches("select, textarea, input[type=range], input[type=text]");
}

window.addEventListener("keydown", (e) => {
  const name = KEY_MAP[e.code];
  if (!name || !state.playing || isTextLikeTarget(e.target)) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  e.preventDefault();
  state.heldKeys.add(e.code);
  onInputChange();
});

window.addEventListener("keyup", (e) => {
  if (!KEY_MAP[e.code]) return;
  if (state.playing && !isTextLikeTarget(e.target)) e.preventDefault();
  state.heldKeys.delete(e.code);
  onInputChange();
});

window.addEventListener("blur", () => {
  releaseAllInput();
  onInputChange();
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    releaseAllInput();
    // Let go of every button before going quiet, so the dream is not left holding a key.
    if (state.playing && state.ws && state.ws.readyState === WebSocket.OPEN) {
      send({ t: "act", buttons: 0, look: [0, 0] });
      state.lastSentButtons = 0;
    }
  }
  updateSending();
});

// Touch buttons. Touch pointers are implicitly captured by the element they
// start on; releasing that capture lets a finger slide from one button to the next.
function setupTouch() {
  const coarse = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  if (coarse || "ontouchstart" in window) document.body.classList.add("has-touch");
  window.addEventListener("touchstart", () => document.body.classList.add("has-touch"), {
    once: true,
    passive: true,
  });

  for (const btn of el.touch.querySelectorAll("[data-btn]")) {
    const name = btn.dataset.btn;
    state.touchHeld.set(name, new Set());
    const press = (e) => {
      if (e.type === "pointerenter" && e.buttons === 0) return;
      e.preventDefault();
      if (e.type === "pointerdown" && btn.hasPointerCapture(e.pointerId)) {
        btn.releasePointerCapture(e.pointerId);
      }
      state.touchHeld.get(name).add(e.pointerId);
      btn.classList.add("held");
      onInputChange();
    };
    const release = (e) => {
      const ids = state.touchHeld.get(name);
      ids.delete(e.pointerId);
      if (!ids.size) btn.classList.remove("held");
      onInputChange();
    };
    btn.addEventListener("pointerdown", press);
    btn.addEventListener("pointerenter", press);
    btn.addEventListener("pointerup", release);
    btn.addEventListener("pointercancel", release);
    btn.addEventListener("pointerleave", release);
  }
  el.touch.addEventListener("contextmenu", (e) => e.preventDefault());
}

// -------------------------------------------------------------------- frames

const contexts = new Map();
const frameSeq = [0, 0];
const drawnSeq = [0, 0];
const pending = [null, null];
let rafId = 0;

function context(canvas) {
  let ctx = contexts.get(canvas);
  if (!ctx) {
    ctx = canvas.getContext("2d", { alpha: false });
    contexts.set(canvas, ctx);
  }
  ctx.imageSmoothingEnabled = false;
  return ctx;
}

// bytes: [channel, ...webp]. Decodes may finish out of order, so each frame
// carries a sequence number and anything older than what is already queued
// or drawn is dropped.
function handleFrame(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 2) return;
  const ch = bytes[0];
  if (ch !== CHANNEL_DREAM && ch !== CHANNEL_REAL) return;
  const seq = ++frameSeq[ch];
  createImageBitmap(new Blob([bytes.subarray(1)], { type: "image/webp" }))
    .then((bmp) => {
      if (seq <= drawnSeq[ch] || (pending[ch] && pending[ch].seq > seq)) {
        bmp.close();
        return;
      }
      if (pending[ch]) pending[ch].bmp.close();
      pending[ch] = { seq, bmp };
      if (!rafId) rafId = requestAnimationFrame(drawPending);
    })
    .catch(() => {});
}

function drawPending() {
  rafId = 0;
  for (const ch of [CHANNEL_DREAM, CHANNEL_REAL]) {
    const p = pending[ch];
    if (!p) continue;
    pending[ch] = null;
    drawnSeq[ch] = p.seq;
    const canvas = ch === CHANNEL_DREAM ? el.dream : el.real;
    if (canvas.width !== p.bmp.width || canvas.height !== p.bmp.height) {
      setCanvasSize(canvas, p.bmp.width, p.bmp.height);
    }
    context(canvas).drawImage(p.bmp, 0, 0);
    p.bmp.close();
  }
}

// -------------------------------------------------------------------- layout

function setCanvasSize(canvas, w, h) {
  if (canvas.width === w && canvas.height === h) return;
  canvas.width = w;
  canvas.height = h;
  layout();
}

// Largest whole-number scale that fits, so every model pixel becomes an
// exact block of screen pixels. Falls back to a fractional scale only when
// even 1x does not fit.
function layout() {
  const n = state.sideBySide ? 2 : 1;
  const style = getComputedStyle(el.stage);
  const padX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  const gap = 12;
  const captionH = state.sideBySide ? 24 : 0;
  const availW = (el.stage.clientWidth - padX - (n - 1) * gap) / n;
  const availH = Math.max(160, window.innerHeight * 0.62) - captionH;
  const w = el.dream.width;
  const h = el.dream.height;
  const fit = Math.min(availW / w, availH / h);
  const k = fit >= 1 ? Math.floor(fit) : fit;
  for (const c of [el.dream, el.real]) {
    c.style.width = `${Math.floor(w * k)}px`;
    c.style.height = `${Math.floor(h * k)}px`;
  }
}

function applySideBySideLayout() {
  el.realWrap.hidden = !state.sideBySide;
  el.dreamLabel.hidden = !state.sideBySide;
  layout();
}

window.addEventListener("resize", layout);

// ------------------------------------------------------------------ controls

function markRadio(container, value) {
  for (const b of container.querySelectorAll("button")) {
    b.setAttribute("aria-checked", String(b.dataset.value === String(value)));
  }
}

function markSteps(n) {
  markRadio(el.steps, n);
}

function buildGameButtons(games) {
  el.games.replaceChildren();
  for (const g of games) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.dataset.value = g.name;
    b.textContent = g.name;
    b.addEventListener("click", () => chooseGame(g.name));
    el.games.append(b);
  }
  markRadio(el.games, state.game);
}

function chooseGame(name) {
  state.game = name;
  markRadio(el.games, name);
  if (el.switchLive.checked) send({ t: "set_game", game: name });
  else send({ t: "reset", game: name });
}

function buildStepButtons() {
  for (const n of SAMPLER_STEPS) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.dataset.value = String(n);
    b.textContent = String(n);
    b.addEventListener("click", () => {
      state.chosenSteps = n;
      markSteps(n);
      send({ t: "config", sampler_steps: n });
    });
    el.steps.append(b);
  }
}

function fillSelect(select, games, preferred) {
  const keep = select.value;
  select.replaceChildren();
  for (const g of games) select.append(new Option(g.name, g.name));
  const names = games.map((g) => g.name);
  select.value = names.includes(keep) ? keep : preferred;
}

function buildBlendSelects(games) {
  fillSelect(el.blendA, games, games[0].name);
  fillSelect(el.blendB, games, (games[1] || games[0]).name);
}

let blendQueued = false;
function sendBlend() {
  if (blendQueued) return;
  blendQueued = true;
  requestAnimationFrame(() => {
    blendQueued = false;
    if (el.blendOn.checked) {
      send({
        t: "blend",
        a: el.blendA.value,
        b: el.blendB.value,
        alpha: Number(el.blendAlpha.value),
      });
    } else {
      send({ t: "blend", a: null });
    }
  });
}

let errorTimer = 0;
function showError(text) {
  el.error.textContent = text;
  el.error.hidden = false;
  clearTimeout(errorTimer);
  errorTimer = setTimeout(() => {
    el.error.hidden = true;
  }, 6000);
}

function clearStats() {
  el.statFps.textContent = "-";
  el.statMs.textContent = "-";
}

function setupControls() {
  buildStepButtons();

  el.start.addEventListener("click", () => {
    state.playing = true;
    connect();
  });
  el.busyRetry.addEventListener("click", connect);
  el.disconnectedScreen.addEventListener("click", connect);
  el.reset.addEventListener("click", () => {
    if (state.game) send({ t: "reset", game: state.game });
  });

  el.sbs.addEventListener("change", () => {
    state.sideBySide = el.sbs.checked;
    if (state.sideBySide) context(el.real).clearRect(0, 0, el.real.width, el.real.height);
    applySideBySideLayout();
    send({ t: "side_by_side", on: state.sideBySide });
  });

  el.blendOn.addEventListener("change", sendBlend);
  el.blendA.addEventListener("change", sendBlend);
  el.blendB.addEventListener("change", sendBlend);
  el.blendAlpha.addEventListener("input", () => {
    if (el.blendOn.checked) sendBlend();
  });

  // Give the keyboard back to the game after using a control, so Space and
  // the arrows do not keep operating the last button, checkbox or menu.
  const giveBack = (node) => {
    if (node instanceof HTMLElement && node.closest("#controls, .overlay")) node.blur();
  };
  document.addEventListener("click", (e) => {
    if (e.target instanceof Element) giveBack(e.target.closest("button, input[type=checkbox]"));
  });
  document.addEventListener("change", (e) => giveBack(e.target));
}

setupControls();
setupTouch();
showScreen("start");
layout();

// ?autostart skips the start screen, for a demo screen left running unattended.
if (new URLSearchParams(location.search).has("autostart")) el.start.click();
