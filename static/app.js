const $ = (id) => document.getElementById(id);
const messages = $("messages");

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function md(text) {
  return DOMPurify.sanitize(marked.parse(text || ""));
}

function addMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `<div class="bubble">${html}</div>`;
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return wrap.querySelector(".bubble");
}

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

/* ---------- documents ---------- */
async function loadDocs() {
  const docs = await api("/api/documents");
  const list = $("docList"), scope = $("scope");
  const current = scope.value;
  scope.innerHTML = '<option value="">All documents</option>';
  if (!docs.length) {
    list.innerHTML = '<li class="muted">No documents yet</li>';
    return;
  }
  list.innerHTML = "";
  for (const d of docs) {
    const li = document.createElement("li");
    li.innerHTML = `<div class="name">${esc(d.source)}</div>
      <div class="meta">${d.pages} pages · ${d.chunks} chunks</div>
      <div class="actions"><button class="sum">Summarise</button><button class="del">Remove</button></div>`;
    li.querySelector(".sum").onclick = () => summarise(d.source);
    li.querySelector(".del").onclick = () => removeDoc(d.source);
    list.appendChild(li);
    scope.insertAdjacentHTML("beforeend", `<option value="${esc(d.source)}">${esc(d.source)}</option>`);
  }
  scope.value = docs.some((d) => d.source === current) ? current : "";
}

async function upload(files) {
  if (!files.length) return;
  const status = $("uploadStatus");
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  status.textContent = `Processing ${files.length} file(s)… (first upload downloads the embedding model)`;
  try {
    const results = await api("/api/upload", { method: "POST", body: fd });
    status.innerHTML = results.map((r) => r.error
      ? `<div class="err">✗ ${esc(r.source)}: ${esc(r.error)}</div>`
      : `<div>✓ ${esc(r.source)}: ${r.pages} pages, ${r.chunks} chunks</div>`).join("");
  } catch (e) {
    status.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
  $("fileInput").value = "";
  loadDocs();
}

async function removeDoc(source) {
  if (!confirm(`Remove ${source}?`)) return;
  await api(`/api/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
  loadDocs();
}

async function summarise(source) {
  addMessage("user", `Summarise <b>${esc(source)}</b>`);
  const bubble = addMessage("bot", '<span class="typing">Reading the document…</span>');
  try {
    const r = await api("/api/summary", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
    bubble.innerHTML = md(r.summary);
  } catch (e) {
    bubble.innerHTML = `<span class="error">${esc(e.message)}</span>`;
  }
}

/* ---------- chat ---------- */
function renderAnswer(r) {
  let html = md(r.answer);
  if (r.sources.length) {
    html += `<details class="sources"><summary>Sources (${r.sources.length})</summary>` +
      r.sources.map((s) => `<div class="src"><b>[${s.n}]</b> ${esc(s.source)}, page ${s.page}
        <span class="meta">· similarity ${s.score.toFixed(2)}</span>
        <div class="ex">${esc(s.excerpt)}…</div></div>`).join("") + "</details>";
  }
  html += `<div class="timing">retrieval ${r.timings.retrieval}s · generation ${r.timings.generation}s</div>`;
  return html;
}

$("chatForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("question").value.trim();
  if (!q) return;
  $("question").value = "";
  $("sendBtn").disabled = true;
  addMessage("user", esc(q));
  const bubble = addMessage("bot", '<span class="typing">Searching your notes…</span>');
  try {
    const r = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, source: $("scope").value }),
    });
    bubble.innerHTML = renderAnswer(r);
  } catch (err) {
    bubble.innerHTML = `<span class="error">${esc(err.message)}</span>`;
  }
  $("sendBtn").disabled = false;
  messages.scrollTop = messages.scrollHeight;
});

$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chatForm").requestSubmit(); }
});

/* ---------- upload widgets ---------- */
$("fileInput").addEventListener("change", (e) => upload(e.target.files));
const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => upload(e.dataTransfer.files));

loadDocs();
