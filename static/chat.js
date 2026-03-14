/**
 * Galactic Gadgets RAG Assistant — chat.js
 * @author: Aarti Dashore
 * Seattle University, ARIN 5360
 * @version: 3.0.0+w26
 *
 * Features:
 * - Message bubbles with timestamps
 * - Scrollable chat history with auto-scroll
 * - Loading indicators
 * - Source display per message
 * - Conversation memory (Option A)
 * - Advanced search controls (Option D)
 * - Document upload (Option E)
 * - Theme toggle with persistence
 * - Export conversation
 * - Keyboard shortcuts
 */

"use strict";

// ── State ──────────────────────────────────────────────────
const state = {
  conversation: [],       // { role, content, sources, timestamp }
  isLoading: false,
  settings: {
    temperature: 0.7,
    contextDocs: 3,
    useHybrid: false,
    useReranking: true,
  },
};

// ── DOM refs ───────────────────────────────────────────────
const els = {
  chatHistory:    () => document.getElementById("chatHistory"),
  questionInput:  () => document.getElementById("questionInput"),
  sendBtn:        () => document.getElementById("sendBtn"),
  welcomeScreen:  () => document.getElementById("welcomeScreen"),
  tempSlider:     () => document.getElementById("tempSlider"),
  tempValue:      () => document.getElementById("tempValue"),
  ctxSlider:      () => document.getElementById("ctxSlider"),
  ctxValue:       () => document.getElementById("ctxValue"),
  hybridToggle:   () => document.getElementById("hybridToggle"),
  rerankToggle:   () => document.getElementById("rerankToggle"),
  strategyBadge:  () => document.getElementById("strategyBadge"),
  themeToggle:    () => document.getElementById("themeToggle"),
  themeIcon:      () => document.getElementById("themeIcon"),
  themeLabel:     () => document.getElementById("themeLabel"),
  clearBtn:       () => document.getElementById("clearBtn"),
  exportBtn:      () => document.getElementById("exportBtn"),
  docCount:       () => document.getElementById("docCount"),
  statusIndicator:() => document.getElementById("statusIndicator"),
  headerSub:      () => document.getElementById("headerSub"),
  charCount:      () => document.getElementById("charCount"),
  sidebar:        () => document.getElementById("sidebar"),
  sidebarToggle:  () => document.getElementById("sidebarToggle"),
  menuBtn:        () => document.getElementById("menuBtn"),
  fileInput:      () => document.getElementById("fileInput"),
  uploadArea:     () => document.getElementById("uploadArea"),
  uploadProgress: () => document.getElementById("uploadProgress"),
  progressFill:   () => document.getElementById("progressFill"),
  progressText:   () => document.getElementById("progressText"),
  docList:        () => document.getElementById("docList"),
};

// ── Utilities ──────────────────────────────────────────────
function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function autoResizeTextarea(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 140) + "px";
}

function scrollToBottom(smooth = true) {
  const h = els.chatHistory();
  h.scrollTo({ top: h.scrollHeight, behavior: smooth ? "smooth" : "instant" });
}

// ── Theme ──────────────────────────────────────────────────
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const isDark = theme === "dark";
  els.themeIcon().textContent = isDark ? "☀️" : "🌙";
  els.themeLabel().textContent = isDark ? "Light mode" : "Dark mode";
  localStorage.setItem("gg-theme", theme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  applyTheme(current === "dark" ? "light" : "dark");
}

// ── Settings persistence ───────────────────────────────────
function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("gg-settings") || "{}");
    if (saved.temperature !== undefined) state.settings.temperature = saved.temperature;
    if (saved.contextDocs !== undefined) state.settings.contextDocs = saved.contextDocs;
    if (saved.useHybrid !== undefined) state.settings.useHybrid = saved.useHybrid;
    if (saved.useReranking !== undefined) state.settings.useReranking = saved.useReranking;
  } catch (_) { /* ignore */ }

  els.tempSlider().value = state.settings.temperature;
  els.tempValue().textContent = state.settings.temperature.toFixed(1);
  els.ctxSlider().value = state.settings.contextDocs;
  els.ctxValue().textContent = state.settings.contextDocs;
  els.hybridToggle().checked = state.settings.useHybrid;
  els.rerankToggle().checked = state.settings.useReranking;
  updateStrategyBadge();
}

function saveSettings() {
  localStorage.setItem("gg-settings", JSON.stringify(state.settings));
}

function updateStrategyBadge() {
  const { useHybrid, useReranking } = state.settings;
  let label = "Semantic only";
  if (useHybrid && useReranking) label = "Hybrid + Reranking";
  else if (useHybrid) label = "Hybrid (BM25 + Semantic)";
  else if (useReranking) label = "Semantic + Reranking";
  els.strategyBadge().textContent = label;
}

// ── Health check ───────────────────────────────────────────
async function checkHealth() {
  const si = els.statusIndicator();
  si.textContent = "checking...";
  si.className = "stat-val status-indicator checking";
  try {
    const res = await fetch("/health");
    const data = await res.json();
    const count = data.documents_indexed ?? 0;
    els.docCount().textContent = count;
    si.textContent = data.status === "healthy" ? "online" : data.status;
    si.className = `stat-val status-indicator ${data.status === "healthy" ? "" : "offline"}`;
    els.headerSub().textContent = `${count} documents indexed`;
  } catch (_) {
    si.textContent = "offline";
    si.className = "stat-val status-indicator offline";
    els.docCount().textContent = "—";
  }
}

// ── Welcome screen ─────────────────────────────────────────
function hideWelcome() {
  const w = els.welcomeScreen();
  if (w) w.remove();
}

function fillQuestion(text) {
  const input = els.questionInput();
  input.value = text;
  autoResizeTextarea(input);
  input.focus();
}

// ── Message rendering ──────────────────────────────────────
function renderUserMessage(text, timestamp) {
  const div = document.createElement("div");
  div.className = "message user-message";
  div.innerHTML = `
    <div class="message-bubble">${escHtml(text)}</div>
    <div class="message-time">${formatTime(timestamp)}</div>
  `;
  return div;
}

function renderSourceCards(sources) {
  if (!sources || sources.length === 0) return "";

  const cards = sources.map((src, i) => {
    const filename = src.metadata?.filename || `Document ${i + 1}`;
    const score = src.score != null
      ? `score: ${Number(src.score).toFixed(3)}`
      : src.metadata?.distance != null
        ? `dist: ${Number(src.metadata.distance).toFixed(3)}`
        : "";
    const preview = escHtml((src.text || "").slice(0, 200));
    const full = escHtml(src.text || "");
    const cardId = `src-${Date.now()}-${i}`;

    return `
      <div class="source-card">
        <div class="source-card-header">
          <span class="source-filename">📄 ${escHtml(filename)}</span>
          ${score ? `<span class="source-score">${escHtml(score)}</span>` : ""}
        </div>
        <div class="source-preview" id="preview-${cardId}">${preview}${(src.text || "").length > 200 ? "…" : ""}</div>
        ${(src.text || "").length > 200
          ? `<button class="source-expand" onclick="toggleSourceFull('${cardId}')">Show full context</button>
             <div class="source-full" id="full-${cardId}">${full}</div>`
          : ""}
      </div>
    `;
  }).join("");

  return `
    <div class="message-sources">
      <details>
        <summary>📚 Sources (${sources.length})</summary>
        <div class="source-cards">${cards}</div>
      </details>
    </div>
  `;
}

function toggleSourceFull(cardId) {
  const full = document.getElementById(`full-${cardId}`);
  const btn = full?.previousElementSibling;
  if (!full) return;
  const visible = full.classList.toggle("visible");
  if (btn) btn.textContent = visible ? "Hide full context" : "Show full context";
}

function renderAssistantMessage(text, sources, timestamp) {
  const div = document.createElement("div");
  div.className = "message assistant-message";
  div.innerHTML = `
    <div class="message-bubble">
      <div class="message-content">${escHtml(text)}</div>
    </div>
    ${renderSourceCards(sources)}
    <div class="message-time">${formatTime(timestamp)}</div>
  `;
  return div;
}

function renderErrorMessage(text) {
  const div = document.createElement("div");
  div.className = "message assistant-message";
  div.innerHTML = `
    <div class="error-bubble">⚠️ ${escHtml(text)}</div>
    <div class="message-time">${formatTime(new Date())}</div>
  `;
  return div;
}

function renderTypingIndicator() {
  const div = document.createElement("div");
  div.className = "message assistant-message";
  div.id = "typingIndicator";
  div.innerHTML = `
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <span class="typing-label">Assistant is typing...</span>
    </div>
  `;
  return div;
}

function removeTypingIndicator() {
  document.getElementById("typingIndicator")?.remove();
}

// ── Send message ───────────────────────────────────────────
async function sendMessage() {
  const input = els.questionInput();
  const question = input.value.trim();
  if (!question || state.isLoading) return;

  hideWelcome();
  state.isLoading = true;
  els.sendBtn().disabled = true;
  input.value = "";
  autoResizeTextarea(input);

  const now = new Date();

  // Add user message to DOM and state
  const userMsg = renderUserMessage(question, now);
  els.chatHistory().appendChild(userMsg);
  state.conversation.push({ role: "user", content: question, timestamp: now });

  // Show typing indicator
  const typing = renderTypingIndicator();
  els.chatHistory().appendChild(typing);
  scrollToBottom();

  try {
    // Build conversation history for Option A (memory)
    const history = buildConversationHistory();

    const payload = {
      question,
      n_context_docs: state.settings.contextDocs,
      temperature: state.settings.temperature,
      use_hybrid: state.settings.useHybrid,
      use_reranking: state.settings.useReranking,
      conversation_history: history,
    };

    const res = await fetch("/rag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    removeTypingIndicator();

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      const errMsg = renderErrorMessage(err.detail || `Request failed (${res.status})`);
      els.chatHistory().appendChild(errMsg);
      state.conversation.push({ role: "error", content: err.detail, timestamp: new Date() });
    } else {
      const data = await res.json();
      const answer = data.answer || "No answer returned.";
      const sources = data.sources || data.context || [];
      const ts = new Date();

      const aMsg = renderAssistantMessage(answer, sources, ts);
      els.chatHistory().appendChild(aMsg);
      state.conversation.push({ role: "assistant", content: answer, sources, timestamp: ts });
    }
  } catch (err) {
    removeTypingIndicator();
    const errMsg = renderErrorMessage(
      err.message?.includes("fetch") ? "Could not connect to server." : err.message
    );
    els.chatHistory().appendChild(errMsg);
  } finally {
    state.isLoading = false;
    els.sendBtn().disabled = false;
    input.focus();
    scrollToBottom();
  }
}

// ── Option A: Conversation memory ─────────────────────────
function buildConversationHistory() {
  const MAX_TURNS = 5;
  const exchanges = [];

  for (let i = 0; i < state.conversation.length; i++) {
    const msg = state.conversation[i];
    if (msg.role === "user" && i + 1 < state.conversation.length) {
      const next = state.conversation[i + 1];
      if (next.role === "assistant") {
        exchanges.push({ question: msg.content, answer: next.content });
        i++; // skip assistant
      }
    }
  }

  // Return last MAX_TURNS Q&A pairs only
  return exchanges.slice(-MAX_TURNS);
}

// ── Clear conversation ─────────────────────────────────────
function clearConversation() {
  state.conversation = [];
  const h = els.chatHistory();
  h.innerHTML = `
    <div class="welcome-screen" id="welcomeScreen">
      <div class="welcome-icon">🛸</div>
      <div class="welcome-title">Ask Galactic Gadgets anything</div>
      <div class="welcome-sub">I'll search through the documentation and give you accurate, cited answers.</div>
      <div class="welcome-chips">
        <button class="welcome-chip" onclick="fillQuestion('What products do you offer?')">What products do you offer?</button>
        <button class="welcome-chip" onclick="fillQuestion('How do I troubleshoot connectivity issues?')">Troubleshoot connectivity</button>
        <button class="welcome-chip" onclick="fillQuestion('What are the technical specifications?')">Technical specifications</button>
        <button class="welcome-chip" onclick="fillQuestion('How do I get started?')">How do I get started?</button>
      </div>
    </div>
  `;
}

// ── Export conversation ────────────────────────────────────
function exportConversation() {
  if (state.conversation.length === 0) {
    alert("No conversation to export.");
    return;
  }

  const lines = [`Galactic Gadgets RAG Assistant — Conversation Export`, `Exported: ${new Date().toLocaleString()}`, `${"=".repeat(60)}\n`];

  for (const msg of state.conversation) {
    if (msg.role === "user") {
      lines.push(`[${formatTime(new Date(msg.timestamp))}] YOU:\n${msg.content}\n`);
    } else if (msg.role === "assistant") {
      lines.push(`[${formatTime(new Date(msg.timestamp))}] ASSISTANT:\n${msg.content}\n`);
      if (msg.sources?.length) {
        lines.push(`Sources: ${msg.sources.map(s => s.metadata?.filename || "unknown").join(", ")}\n`);
      }
    }
    lines.push("-".repeat(40));
  }

  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `gg-chat-${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Option E: Document upload ──────────────────────────────
async function uploadFiles(files) {
  if (!files || files.length === 0) return;

  const MAX_SIZE = 10 * 1024 * 1024; // 10MB
  const validFiles = [];

  for (const f of files) {
    if (!f.name.match(/\.(txt|pdf)$/i)) {
      alert(`Skipping "${f.name}" — only .txt and .pdf files are supported.`);
      continue;
    }
    if (f.size > MAX_SIZE) {
      alert(`Skipping "${f.name}" — file exceeds 10MB limit.`);
      continue;
    }
    validFiles.push(f);
  }

  if (validFiles.length === 0) return;

  const progress = els.uploadProgress();
  const fill = els.progressFill();
  const text = els.progressText();
  progress.style.display = "block";

  for (let i = 0; i < validFiles.length; i++) {
    const f = validFiles[i];
    const pct = Math.round(((i) / validFiles.length) * 100);
    fill.style.width = `${pct}%`;
    text.textContent = `Uploading ${f.name}...`;

    try {
      const formData = new FormData();
      formData.append("file", f);

      const res = await fetch("/upload", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(`Failed to upload "${f.name}": ${err.detail || "Unknown error"}`);
      }
    } catch (err) {
      alert(`Upload failed for "${f.name}": ${err.message}`);
    }
  }

  fill.style.width = "100%";
  text.textContent = "Done! Re-indexing documents...";

  await new Promise(r => setTimeout(r, 800));
  progress.style.display = "none";
  fill.style.width = "0%";

  // Refresh health to get updated doc count
  await checkHealth();
  await loadDocumentList();
}

async function loadDocumentList() {
  try {
    const res = await fetch("/documents");
    if (!res.ok) return;
    const data = await res.json();
    const docs = data.documents || [];
    const list = els.docList();
    list.innerHTML = docs.map(d => `
      <div class="doc-item">
        <span class="doc-name" title="${escHtml(d.filename)}">📄 ${escHtml(d.filename)}</span>
        <button class="doc-delete" onclick="deleteDocument('${escHtml(d.filename)}')" title="Delete">✕</button>
      </div>
    `).join("") || '<div style="font-size:0.78rem;color:var(--text-muted);padding:4px 0">No documents listed</div>';
  } catch (_) { /* endpoint may not exist yet */ }
}

async function deleteDocument(filename) {
  if (!confirm(`Delete "${filename}"?`)) return;
  try {
    const res = await fetch(`/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`Failed to delete: ${err.detail || "Unknown error"}`);
      return;
    }
    await checkHealth();
    await loadDocumentList();
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
}

// ── Sidebar ────────────────────────────────────────────────
function toggleSidebar() {
  const sb = els.sidebar();
  sb.classList.toggle("collapsed");
  const collapsed = sb.classList.contains("collapsed");
  els.sidebarToggle().textContent = collapsed ? "›" : "‹";
}

// ── Keyboard shortcuts ─────────────────────────────────────
function setupKeyboard() {
  const input = els.questionInput();
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  input.addEventListener("input", () => {
    autoResizeTextarea(input);
    const len = input.value.length;
    els.charCount().textContent = len > 0 ? `${len} chars` : "";
  });
}

// ── Init ───────────────────────────────────────────────────
function init() {
  // Theme
  const savedTheme = localStorage.getItem("gg-theme") || "dark";
  applyTheme(savedTheme);

  // Settings
  loadSettings();

  // Keyboard
  setupKeyboard();

  // Sliders
  els.tempSlider().addEventListener("input", (e) => {
    state.settings.temperature = Number(e.target.value);
    els.tempValue().textContent = state.settings.temperature.toFixed(1);
    saveSettings();
  });
  els.ctxSlider().addEventListener("input", (e) => {
    state.settings.contextDocs = Number(e.target.value);
    els.ctxValue().textContent = state.settings.contextDocs;
    saveSettings();
  });

  // Option D toggles
  els.hybridToggle().addEventListener("change", (e) => {
    state.settings.useHybrid = e.target.checked;
    updateStrategyBadge();
    saveSettings();
  });
  els.rerankToggle().addEventListener("change", (e) => {
    state.settings.useReranking = e.target.checked;
    updateStrategyBadge();
    saveSettings();
  });

  // Theme toggle
  els.themeToggle().addEventListener("click", toggleTheme);

  // Clear / Export
  els.clearBtn().addEventListener("click", () => {
    if (state.conversation.length === 0 || confirm("Clear the conversation?")) {
      clearConversation();
    }
  });
  els.exportBtn().addEventListener("click", exportConversation);

  // Sidebar
  els.sidebarToggle().addEventListener("click", toggleSidebar);
  els.menuBtn().addEventListener("click", toggleSidebar);

  // File upload
  els.fileInput().addEventListener("change", (e) => uploadFiles(e.target.files));
  const ua = els.uploadArea();
  ua.addEventListener("dragover", (e) => { e.preventDefault(); ua.classList.add("dragover"); });
  ua.addEventListener("dragleave", () => ua.classList.remove("dragover"));
  ua.addEventListener("drop", (e) => {
    e.preventDefault();
    ua.classList.remove("dragover");
    uploadFiles(e.dataTransfer.files);
  });

  // Health
  checkHealth();
  setInterval(checkHealth, 30000);
  loadDocumentList();
}

document.addEventListener("DOMContentLoaded", init);