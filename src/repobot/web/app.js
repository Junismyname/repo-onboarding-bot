// Repo Onboarding Bot web UI: chat with streamed answers, clickable citations, source viewer.

const CITATION_RE = /\[([^\[\]\s]+?):(\d+)-(\d+)\]/g;
const HISTORY_MESSAGES = 6; // the server trims further; this just bounds the request size
const HLJS_LANG = {
  python: "python", javascript: "javascript", typescript: "typescript", tsx: "typescript",
  markdown: "markdown", json: "json", yaml: "yaml", toml: "ini", ini: "ini",
  go: "go", rust: "rust", java: "java", dockerfile: "dockerfile",
};

const $ = (id) => document.getElementById(id);
const state = {
  repoId: null,
  messages: [], // {role, content, sources, query, error, el}
  busy: false,
};
const fileCache = new Map(); // `${repoId}:${path}` -> Promise<file>

// ---------- repos & indexing ----------

async function loadRepos(selectId) {
  const select = $("repo-select");
  let repos = [];
  try {
    repos = await (await fetch("/repos")).json();
  } catch {
    select.innerHTML = "<option>API unreachable</option>";
    return;
  }
  select.innerHTML = repos.length
    ? repos.map((r) => `<option value="${escapeHtml(r.repo_id)}">${escapeHtml(r.name)} @ ${escapeHtml(r.sha.slice(0, 7))}</option>`).join("")
    : "<option value=''>No repos indexed yet</option>";
  const wanted = selectId || localGet("repoId");
  if (wanted && repos.some((r) => r.repo_id === wanted)) select.value = wanted;
  setRepo(select.value || null);
}

function setRepo(repoId) {
  if (repoId === state.repoId) return;
  state.repoId = repoId;
  localSet("repoId", repoId || "");
  newChat();
}

$("repo-select").addEventListener("change", (e) => setRepo(e.target.value));

$("index-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = $("repo-url").value.trim();
  const status = $("index-status");
  const button = e.submitter;
  button.disabled = true;
  status.hidden = false;
  $("index-stage").textContent = "queued";
  try {
    let job = await (await fetch("/repos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    })).json();
    while (job.status === "queued" || job.status === "running") {
      await sleep(1000);
      job = await (await fetch(`/jobs/${job.id}`)).json();
      $("index-progress").value = job.progress;
      $("index-stage").textContent = job.stage;
    }
    if (job.status === "failed") throw new Error(job.error);
    $("index-stage").textContent = "done";
    $("repo-url").value = "";
    await loadRepos(job.repo_id);
    setTimeout(() => { status.hidden = true; }, 1500);
  } catch (err) {
    $("index-stage").textContent = `failed: ${err.message}`;
  } finally {
    button.disabled = false;
  }
});

// ---------- chat ----------

function newChat() {
  state.messages = [];
  $("messages").innerHTML = "";
  $("messages").appendChild(emptyState);
  closeViewer();
}
const emptyState = $("empty");
$("new-chat").addEventListener("click", newChat);

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  if (!question || state.busy || !state.repoId) return;
  $("question").value = "";
  autoGrow();
  ask(question);
});

$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("ask-form").requestSubmit();
  }
});
$("question").addEventListener("input", autoGrow);
function autoGrow() {
  const t = $("question");
  t.style.height = "auto";
  t.style.height = `${t.scrollHeight}px`;
}

function setBusy(busy) {
  state.busy = busy;
  // Switching repo or clearing the chat mid-answer would orphan the streaming message.
  for (const id of ["send", "repo-select", "new-chat"]) $(id).disabled = busy;
}

async function ask(question) {
  setBusy(true);
  emptyState.remove();

  // History = earlier turns only (the current question is sent separately).
  const history = state.messages
    .filter((m) => m.content && !m.error)
    .slice(-HISTORY_MESSAGES)
    .map((m) => ({ role: m.role, content: m.content }));

  addMessage({ role: "user", content: question });
  const msg = addMessage({ role: "assistant", content: "", sources: [] });
  msg.el.querySelector(".bubble").classList.add("typing");

  try {
    const res = await fetch(`/repos/${encodeURIComponent(state.repoId)}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    if (!res.ok) throw new Error(`${res.status} ${(await res.json()).detail || res.statusText}`);
    await readEvents(res, (event) => {
      if (event.query !== undefined) msg.query = event.query;
      if (event.sources) msg.sources = event.sources;
      if (event.token) msg.content += event.token;
      if (event.error) msg.error = event.error;
      scheduleRender(msg, question);
    });
  } catch (err) {
    msg.error = `Request failed: ${err.message}`;
  }
  msg.el.querySelector(".bubble").classList.remove("typing");
  renderAssistant(msg, question, true);
  setBusy(false);
  $("question").focus();
}

/** Parse a Server-Sent Events stream from fetch() and call onEvent with each JSON payload. */
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of block.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") return;
        onEvent(JSON.parse(data));
      }
    }
  }
}

function addMessage(message) {
  const el = document.createElement("div");
  el.className = `msg ${message.role}`;
  el.innerHTML = `<div class="note" hidden></div><div class="bubble"></div><div class="sources" hidden></div>`;
  if (message.role === "user") el.querySelector(".bubble").textContent = message.content;
  $("messages").appendChild(el);
  message.el = el;
  state.messages.push(message);
  scrollToBottom();
  return message;
}

const pendingRenders = new Map();
function scheduleRender(msg, question) {
  if (pendingRenders.has(msg)) return;
  pendingRenders.set(msg, requestAnimationFrame(() => {
    pendingRenders.delete(msg);
    renderAssistant(msg, question, false);
  }));
}

function renderAssistant(msg, question, final) {
  const nearBottom = isNearBottom();
  const note = msg.el.querySelector(".note");
  if (msg.query && msg.query !== question) {
    note.hidden = false;
    note.textContent = `Interpreted as: ${msg.query}`;
  }

  const bubble = msg.el.querySelector(".bubble");
  bubble.innerHTML = DOMPurify.sanitize(marked.parse(msg.content || ""));
  const labels = new Set(msg.sources.map((s) => `${s.path}:${s.start_line}-${s.end_line}`));
  linkCitations(bubble, labels);
  if (msg.error) {
    const err = document.createElement("p");
    err.className = "error";
    err.textContent = msg.error;
    bubble.appendChild(err);
  }
  if (final) bubble.querySelectorAll("pre code").forEach((el) => hljs.highlightElement(el));

  const box = msg.el.querySelector(".sources");
  if (final && msg.sources.length) {
    box.hidden = false;
    box.innerHTML = `<span class="sources-label">Retrieved sources</span>`;
    for (const s of msg.sources) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = `${basename(s.path)}:${s.start_line}-${s.end_line}`;
      chip.title = s.path;
      chip.addEventListener("click", () => openSource(s.path, s.start_line, s.end_line, chip));
      box.appendChild(chip);
    }
  }
  if (nearBottom) scrollToBottom();
}

/** Turn "[path:12-30]" in rendered text (outside code) into links that open the viewer. */
function linkCitations(root, knownLabels) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement.closest("code, pre, a") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);

  for (const node of nodes) {
    const text = node.nodeValue;
    CITATION_RE.lastIndex = 0;
    if (!CITATION_RE.test(text)) continue;
    CITATION_RE.lastIndex = 0;
    const frag = document.createDocumentFragment();
    let last = 0;
    for (const m of text.matchAll(CITATION_RE)) {
      const [whole, path, start, end] = m;
      frag.append(text.slice(last, m.index));
      const a = document.createElement("a");
      a.href = "#";
      a.className = "cite";
      a.textContent = `${basename(path)}:${start}-${end}`;
      const verified = knownLabels.has(`${path}:${start}-${end}`);
      if (!verified) a.classList.add("unverified");
      a.title = verified ? path : `${path} (not one of the retrieved snippets)`;
      a.addEventListener("click", (e) => {
        e.preventDefault();
        openSource(path, Number(start), Number(end), a);
      });
      frag.append(a);
      last = m.index + whole.length;
    }
    frag.append(text.slice(last));
    node.replaceWith(frag);
  }
}

// ---------- source viewer ----------

let activeTrigger = null;

async function openSource(path, start, end, trigger) {
  const viewer = $("viewer");
  viewer.hidden = false;
  activeTrigger?.classList.remove("active");
  activeTrigger = trigger || null;
  activeTrigger?.classList.add("active");

  $("viewer-path").textContent = path;
  $("viewer-path").title = path;
  $("viewer-range").textContent = `lines ${start}–${end}`;
  const body = $("viewer-body");
  body.innerHTML = `<div class="status">Loading…</div>`;

  try {
    const file = await getFile(path);
    $("viewer-github").href = `${file.github_url}#L${start}-L${end}`;
    const lang = HLJS_LANG[file.language];
    const html = lang && hljs.getLanguage(lang)
      ? hljs.highlight(file.content, { language: lang, ignoreIllegals: true }).value
      : escapeHtml(file.content);
    const lines = splitHighlightedLines(html);
    const rows = lines.map((line, i) => {
      const n = i + 1;
      const hl = n >= start && n <= end ? ' class="hl"' : "";
      return `<tr${hl} id="L${n}"><td class="ln">${n}</td><td class="code">${line || " "}</td></tr>`;
    });
    body.innerHTML = `<table class="code-table"><tbody>${rows.join("")}</tbody></table>`;
    const target = body.querySelector(`#L${Math.max(1, start - 3)}`);
    target?.scrollIntoView({ block: "start" });
  } catch (err) {
    body.innerHTML = `<div class="status">Couldn't load ${escapeHtml(path)}: ${escapeHtml(err.message)}</div>`;
    $("viewer-github").removeAttribute("href");
  }
}

function getFile(path) {
  const key = `${state.repoId}:${path}`;
  if (!fileCache.has(key)) {
    const url = `/repos/${encodeURIComponent(state.repoId)}/file?path=${encodeURIComponent(path)}`;
    fileCache.set(key, fetch(url).then(async (res) => {
      if (!res.ok) {
        fileCache.delete(key);
        throw new Error(res.status === 404 ? "not in this repo's index" : `HTTP ${res.status}`);
      }
      return res.json();
    }));
  }
  return fileCache.get(key);
}

/**
 * highlight.js returns one HTML string where a <span> (e.g. a docstring) can cross newlines.
 * Split it into per-line HTML, closing open spans at each line end and reopening them on the
 * next line, so every table row is valid HTML.
 */
function splitHighlightedLines(html) {
  const out = [];
  let open = [];
  for (const line of html.split("\n")) {
    const prefix = open.join("");
    for (const m of line.matchAll(/<span[^>]*>|<\/span>/g)) {
      if (m[0] === "</span>") open.pop();
      else open.push(m[0]);
    }
    out.push(prefix + line + "</span>".repeat(open.length));
  }
  return out;
}

function closeViewer() {
  $("viewer").hidden = true;
  activeTrigger?.classList.remove("active");
  activeTrigger = null;
}
$("viewer-close").addEventListener("click", closeViewer);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("viewer").hidden) closeViewer();
});

// ---------- helpers ----------

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function basename(path) { return path.split("/").pop(); }
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
function isNearBottom() {
  const m = $("messages");
  return m.scrollHeight - m.scrollTop - m.clientHeight < 120;
}
function scrollToBottom() { const m = $("messages"); m.scrollTop = m.scrollHeight; }
function localGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function localSet(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } }

loadRepos();
