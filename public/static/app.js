/* StudyMate front end.
 *
 * RAG pipeline split between browser and server:
 *   1. pdf.js extracts text from the PDF here in the browser.
 *   2. The server chunks the text and embeds it with Gemini (/api/index).
 *   3. Chunks + vectors are stored in this browser's IndexedDB (the vector store).
 *   4. A question is embedded by the server (/api/embed-query); the browser ranks
 *      its stored chunks by cosine similarity and sends the best ones to
 *      /api/answer, where Gemini writes a cited answer.
 */
const $ = (id) => document.getElementById(id);
const messages = $("messages");
const PAGES_PER_REQUEST = 10;
const SECTION_CHARS = 10000; // characters of PDF text per summary section

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

let CONFIG = { embedding_model: "", embedding_dim: 0, top_k: 5, min_score: 0.5 };
const docs = new Map(); // source -> {source, pages, builtIn, chunks:[{page, chunk, text, embedding}]}

/* ---------- helpers ---------- */
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
// Gemini writes formulas in LaTeX ($A \neq B$, $$...$$, \(...\), \[...\]). They are taken out
// before Markdown parsing (so "_" and "\" are not mangled), rendered with KaTeX, and put back.
const MATH_RE = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?!\d)/g;
function renderMath(src, display) {
  if (window.katex) {
    try { return katex.renderToString(src, { displayMode: display, throwOnError: false, output: "html" }); }
    catch { /* fall through */ }
  }
  return `<code>${esc(src)}</code>`;
}
function md(text) {
  const math = [];
  const t = (text || "").replace(MATH_RE, (m, a, b, c, d) => {
    math.push(renderMath(a ?? b ?? c ?? d, a != null || b != null));
    return `%%MATH${math.length - 1}%%`;
  });
  return DOMPurify.sanitize(marked.parse(t)).replace(/%%MATH(\d+)%%/g, (_, i) => math[+i]);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const scroller = $("scroll");
const scrollToEnd = () => { scroller.scrollTop = scroller.scrollHeight; };

function addMessage(role, html) {
  showView("home");
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  if (role === "bot") wrap.appendChild($("avatarTpl").content.cloneNode(true));
  wrap.insertAdjacentHTML("beforeend", `<div class="bubble">${html}</div>`);
  messages.appendChild(wrap);
  requestAnimationFrame(() => wrap.scrollIntoView({ block: "start", behavior: "smooth" }));
  return wrap.querySelector(".bubble");
}

/** POST JSON; on 429 (Gemini rate limit) wait and retry a few times.
 *  Requests are abandoned after REQUEST_TIMEOUT_MS so a stuck call can never freeze the page
 *  (the server's own limit is 150 s; callers like apiPatient then retry). */
const REQUEST_TIMEOUT_MS = 160000;
async function api(url, body, onWait) {
  for (let attempt = 0; ; attempt++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS);
    let res;
    try {
      res = await fetch(url, body === undefined ? { signal: ctrl.signal } : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal,
      });
    } catch (e) {
      throw new Error(e.name === "AbortError" ? "Gemini took too long to respond — please try again." : "Network error — check your connection.");
    } finally {
      clearTimeout(timer);
    }
    const data = await res.json().catch(() => ({}));
    if (res.ok) return data;
    if (res.status === 401 && data.login_required) {
      window.location.href = "/login"; // session expired
      throw new Error("Please sign in again.");
    }
    if (res.status === 429 && !data.daily_quota && attempt < 4) {
      const wait = 5 * (attempt + 1);
      if (onWait) onWait(wait);
      await sleep(wait * 1000);
      continue;
    }
    throw new Error(data.error || `Request failed (${res.status})`);
  }
}

/* ---------- vector store: IndexedDB, with an in-memory fallback ---------- */
const store = (() => {
  let dbp = null;
  function open() {
    if (!dbp) {
      dbp = new Promise((resolve) => {
        try {
          const req = indexedDB.open("studymate", 2);
          req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains("docs")) db.createObjectStore("docs", { keyPath: "source" });
            if (!db.objectStoreNames.contains("subjects")) db.createObjectStore("subjects", { keyPath: "id" });
          };
          req.onsuccess = () => resolve(req.result);
          req.onerror = () => resolve(null);
        } catch { resolve(null); }
      });
    }
    return dbp;
  }
  async function tx(name, mode, fn) {
    const db = await open();
    if (!db) return undefined; // private mode etc.: data lasts until page reload
    return new Promise((resolve, reject) => {
      const t = db.transaction(name, mode);
      const r = fn(t.objectStore(name));
      t.oncomplete = () => resolve(r && r.result);
      t.onerror = () => reject(t.error);
    });
  }
  return {
    all: () => tx("docs", "readonly", (s) => s.getAll()).then((r) => r || []),
    put: (doc) => tx("docs", "readwrite", (s) => s.put(doc)),
    remove: (source) => tx("docs", "readwrite", (s) => s.delete(source)),
    subjects: () => tx("subjects", "readonly", (s) => s.getAll()).then((r) => r || []),
    putSubject: (sub) => tx("subjects", "readwrite", (s) => s.put(sub)),
    removeSubject: (id) => tx("subjects", "readwrite", (s) => s.delete(id)),
  };
})();

/** Save a document's metadata (subject, syllabus flag) without the in-memory-only fields. */
function saveDoc(d) {
  const { builtIn, ...rest } = d;
  if (!builtIn) return store.put(rest);
}

function toDoc(raw, builtIn) {
  return {
    source: raw.source, pages: raw.pages, builtIn, subject: raw.subject || "general",
    chunks: raw.chunks.map((c) => ({ ...c, embedding: Float32Array.from(c.embedding) })),
  };
}

/* ---------- subjects: separate spaces for each course ---------- */
const subjects = new Map(); // id -> {id, name, created}
let currentSubject = "general";
const subjectOf = (d) => d.subject || "general";
const subjectDocs = (id = currentSubject) => [...docs.values()].filter((d) => subjectOf(d) === id);
const syllabusOf = (id = currentSubject) => subjectDocs(id).find((d) => d.role === "syllabus");
const subjectName = (id = currentSubject) => (subjects.get(id) || { name: "General" }).name;

function rememberSubject() { try { localStorage.setItem("studymate-subject", currentSubject); } catch {} }

async function loadSubjects() {
  for (const s of await store.subjects()) subjects.set(s.id, s);
  if (!subjects.has("general")) {
    const g = { id: "general", name: "General", created: Date.now() };
    subjects.set("general", g);
    await store.putSubject(g);
  }
  let saved = null;
  try { saved = localStorage.getItem("studymate-subject"); } catch {}
  currentSubject = saved && subjects.has(saved) ? saved : "general";
}

function switchSubject(id) {
  if (!subjects.has(id)) return;
  currentSubject = id;
  rememberSubject();
  $("scope").value = "";
  renderDocs();
}

async function newSubject() {
  const name = (prompt("Name of the new subject (e.g. DBMS, Signals & Systems):") || "").trim();
  if (!name) return;
  const id = "s" + Date.now().toString(36);
  const s = { id, name: name.slice(0, 60), created: Date.now() };
  subjects.set(id, s);
  await store.putSubject(s);
  switchSubject(id);
}

async function renameSubject() {
  const s = subjects.get(currentSubject);
  const name = (prompt("New name for this subject:", s.name) || "").trim();
  if (!name) return;
  s.name = name.slice(0, 60);
  await store.putSubject(s);
  renderDocs();
}

async function deleteSubject() {
  if (currentSubject === "general") { alert("The General subject can't be deleted."); return; }
  const inside = subjectDocs();
  const msg = inside.length
    ? `Delete the subject “${subjectName()}”? Its ${inside.length} document(s) will be moved to General.`
    : `Delete the empty subject “${subjectName()}”?`;
  if (!confirm(msg)) return;
  for (const d of inside) { d.subject = "general"; if (d.role === "syllabus") delete d.role; await saveDoc(d); }
  await store.removeSubject(currentSubject);
  subjects.delete(currentSubject);
  switchSubject("general");
}

async function moveDoc(source, to) {
  const d = docs.get(source);
  if (!d || !subjects.has(to)) return;
  d.subject = to;
  if (d.role === "syllabus") delete d.role;
  await saveDoc(d);
  renderDocs();
}

async function toggleSyllabus(source) {
  const d = docs.get(source);
  if (!d) return;
  const makeIt = d.role !== "syllabus";
  for (const other of subjectDocs(subjectOf(d))) {
    if (other.role === "syllabus") { delete other.role; await saveDoc(other); }
  }
  if (makeIt) { d.role = "syllabus"; await saveDoc(d); }
  renderDocs();
}

async function loadDocs() {
  CONFIG = await api("/api/config");
  await loadSubjects();
  // Built-in course documents indexed ahead of time (scripts/build_sample_index.py).
  try {
    const res = await fetch("/static/sample_index.json");
    if (res.ok) {
      const idx = await res.json();
      if (idx.model === CONFIG.embedding_model && idx.dim === CONFIG.embedding_dim) {
        idx.docs.forEach((d) => docs.set(d.source, toDoc(d, true)));
      }
    }
  } catch { /* no built-in docs */ }
  let stale = 0;
  for (const d of await store.all()) {
    if (d.model !== CONFIG.embedding_model || d.dim !== CONFIG.embedding_dim) {
      stale++;
      await store.remove(d.source);
      continue;
    }
    if (!subjects.has(subjectOf(d))) d.subject = "general";
    docs.set(d.source, { ...d, builtIn: false });
  }
  // videos indexed before the timestamp fix have (almost) all notes at 00:00
  const badVideos = [...docs.values()].filter((d) => d.kind === "video" && d.chunks.length > 4 &&
    new Set(d.chunks.map((c) => c.page)).size <= 2);
  if (badVideos.length) {
    $("uploadStatus").innerHTML = badVideos.map((d) => `<div class="err">${ico("alert")} “${esc(d.source)}” was added before a timestamp fix, so its times show 00:00. Remove it (bin icon) and add the link again.</div>`).join("");
  }
  if (stale) {
    $("uploadStatus").innerHTML =
      `<div class="err">${stale} document(s) were indexed with an older model and were removed. Please upload them again.</div>`;
  }
  renderDocs();
}

const ico = (name) => `<svg class="ic"><use href="#i-${name}"/></svg>`;
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function docCard(d, { move = false } = {}) {
  const li = document.createElement("li");
  li.className = "doc" + (d.role === "syllabus" ? " syllabus" : "");
  const video = d.kind === "video";
  const meta = video ? `YouTube${d.author ? " · " + esc(d.author) : ""} · ${fmtTs(d.duration || 0)} · ${plural(d.chunks.length, "chunk")}`
                     : `${plural(d.pages, "page")}${d.imagePages ? ` · ${d.imagePages} read from images` : ""} · ${plural(d.chunks.length, "chunk")}${d.builtIn ? " · built-in" : ""}`;
  const badge = d.role === "syllabus" ? `<span class="badge">Syllabus</span>`
    : d.role === "paper" ? `<span class="badge paper">Past paper${d.paper && d.paper.year ? " " + d.paper.year : ""}</span>` : "";
  li.innerHTML = `<div class="doc-icon${video ? " video" : ""}">${ico(video ? "play" : "file")}</div>
    <div><div class="name">${video ? `<button class="linklike">${esc(d.source)}</button>` : esc(d.source)} ${badge}</div>
      <div class="meta">${meta}</div></div>`;
  if (video) li.querySelector(".linklike").onclick = () => openPlayer(d.videoId, 0, d.source);
  const syl = !video && !d.builtIn && d.role !== "paper"
    ? `<button class="btn syl" title="Mark as this subject's syllabus">${ico("star")}${d.role === "syllabus" ? "Syllabus ✓" : "Set as syllabus"}</button>` : "";
  const mover = move && !d.builtIn
    ? `<select class="select mover" aria-label="Move to subject">${[...subjects.values()].map((s) =>
        `<option value="${esc(s.id)}"${s.id === subjectOf(d) ? " selected" : ""}>${esc(s.name)}</option>`).join("")}</select>` : "";
  li.insertAdjacentHTML("beforeend", `<div class="actions">${mover}${syl}<button class="btn primary sum">${ico("spark")}Summarise</button>
      ${d.builtIn ? "" : `<button class="btn danger del" aria-label="Remove ${esc(d.source)}">${ico("trash")}</button>`}</div>`);
  li.querySelector(".sum").onclick = () => summarise(d.source);
  if (!d.builtIn) li.querySelector(".del").onclick = () => removeDoc(d.source);
  const sb = li.querySelector(".syl"); if (sb) sb.onclick = () => toggleSyllabus(d.source);
  const mv = li.querySelector(".mover"); if (mv) mv.onchange = () => moveDoc(d.source, mv.value);
  return li;
}

const bySyllabusThenName = (a, b) => (b.role === "syllabus") - (a.role === "syllabus") || a.source.localeCompare(b.source);

function renderDocs() {
  // subject switcher
  const sel = $("subjectSel");
  sel.innerHTML = [...subjects.values()].sort((a, b) => (a.id === "general") - (b.id === "general") || a.name.localeCompare(b.name))
    .map((s) => `<option value="${esc(s.id)}">${esc(s.name)} (${subjectDocs(s.id).length})</option>`).join("");
  sel.value = currentSubject;
  $("subjectTitle").textContent = subjectName();
  const syl = syllabusOf();
  $("subjectSyllabus").innerHTML = syl ? `${ico("star")} Syllabus: <b>${esc(syl.source)}</b>` : "No syllabus set — mark a PDF as the syllabus to keep viva questions on-topic.";

  const inSubject = subjectDocs().sort(bySyllabusThenName);
  const scope = $("scope");
  const current = scope.value;
  scope.innerHTML = `<option value="">Whole subject</option>` +
    inSubject.map((d) => `<option value="${esc(d.source)}">${esc(d.source)}</option>`).join("");
  scope.value = inSubject.some((d) => d.source === current) ? current : "";

  const vs = $("vivaScope");
  if (vs) {
    const keep = vs.value;
    vs.innerHTML = `<option value="">Whole subject: ${esc(subjectName())}</option>` +
      inSubject.filter((d) => d.role !== "syllabus").map((d) => `<option value="${esc(d.source)}">${esc(d.source)}</option>`).join("");
    vs.value = inSubject.some((d) => d.source === keep) ? keep : "";
    $("vivaSubject").textContent = subjectName();
    $("vivaStart").disabled = !inSubject.length;
    $("vivaHint").textContent = inSubject.length
      ? (syl ? `Questions will stay within the syllabus (${syl.source}). ` : "") + "Tip: use Chrome or Edge to answer by voice. You can always type instead."
      : `Add a PDF or a YouTube lecture to “${subjectName()}” on the Home page first.`;
  }

  // home: this subject only
  const list = $("docList");
  list.innerHTML = "";
  if (!inSubject.length) {
    list.innerHTML = `<li class="empty">${ico("file")}<div><strong>No documents in “${esc(subjectName())}” yet</strong>
      <span>Upload its syllabus, notes and lecture videos above.</span></div></li>`;
  } else inSubject.forEach((d) => list.appendChild(docCard(d)));

  // My Documents: every subject, grouped, with a "move to" control
  const groups = $("docGroups");
  groups.innerHTML = "";
  for (const s of [...subjects.values()].sort((a, b) => (a.id === "general") - (b.id === "general") || a.name.localeCompare(b.name))) {
    const ds = subjectDocs(s.id).sort(bySyllabusThenName);
    const box = document.createElement("div");
    box.className = "subject-group";
    box.innerHTML = `<div class="group-head"><h3>${ico("folder")}${esc(s.name)}</h3><span>${plural(ds.length, "item")}</span></div>`;
    const ul = document.createElement("ul");
    ul.className = "doc-list";
    if (!ds.length) ul.innerHTML = `<li class="muted-row">Empty</li>`;
    ds.forEach((d) => ul.appendChild(docCard(d, { move: true })));
    box.appendChild(ul);
    groups.appendChild(box);
  }

  renderPapers();
  const user = [...docs.values()].filter((d) => !d.builtIn);
  $("storageInfo").innerHTML = `<dt>Subjects</dt><dd>${subjects.size}</dd><dt>Documents</dt><dd>${user.length}</dd>
    <dt>Pages</dt><dd>${user.reduce((n, d) => n + (d.kind === "video" ? 0 : d.pages), 0)}</dd>
    <dt>Chunks</dt><dd>${user.reduce((n, d) => n + d.chunks.length, 0)}</dd>`;
}

/* ---------- YouTube lectures ---------- */
function fmtTs(sec) {
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** Gemini watches the video in 10-minute windows and writes timestamped notes;
 *  the notes are indexed like PDF pages whose "page" is the second in the video. */
/** Live progress panel for a video: overall bar, counts, time left, and one chip per part. */
function videoProgress(line, title, total) {
  const parts = [];            // {start, end, state: waiting | watching | retrying | done | empty}
  const t0 = Date.now();
  let notes = 0, stage = "watch", indexDone = 0, indexTotal = 0;
  const label = (p) => `${fmtTs(p.start)}–${fmtTs(p.end)}`;
  const icon = { waiting: "", watching: '<span class="spin"></span>', retrying: "↻", done: "✓", empty: "–" };
  function render() {
    const known = total || parts.length;
    const done = parts.filter((p) => p.state === "done" || p.state === "empty").length;
    let pct, head, sub;
    if (stage === "watch") {
      pct = known ? Math.round((100 * done) / known) : 0;
      const left = Math.max(0, known - done);
      const secs = (Date.now() - t0) / 1000;
      const eta = done ? Math.round((secs / done) * left) : null;
      head = `Watching the lecture — ${done} of ${known}${total ? "" : "+"} part${known === 1 ? "" : "s"} done · ${left} left`;
      sub = `${notes} notes so far · ${eta !== null && left ? `about ${eta < 60 ? eta + " s" : Math.ceil(eta / 60) + " min"} remaining` : "starting…"}` +
            (parts.some((p) => p.state === "retrying") ? " · Gemini is busy, retrying a part…" : "");
    } else {
      pct = indexTotal ? Math.round((100 * indexDone) / indexTotal) : 100;
      head = `Indexing notes for search — ${indexDone} of ${indexTotal}`;
      sub = `All ${known} parts watched · ${notes} notes`;
    }
    line.innerHTML = `<div class="vprog">
        <div class="vprog-title">${ico("play")}<b>${esc(title)}</b></div>
        <div class="vprog-head"><span>${esc(head)}</span><span>${pct}%</span></div>
        <div class="bar"><i style="width:${pct}%"></i></div>
        <div class="vprog-sub">${esc(sub)}</div>
        ${stage === "watch" ? `<div class="chips-row">${parts.filter((p) => p.state !== "empty").map((p) =>
          `<span class="chip ${p.state}" title="${p.state}">${icon[p.state]} ${label(p)}</span>`).join("")}</div>` : ""}
      </div>`;
  }
  return {
    parts, render,
    add(start, end) { const p = { start, end, state: "waiting" }; parts.push(p); return p; },
    set(p, state) { p.state = state; render(); },
    addNotes(n) { notes += n; render(); },
    indexing(doneN, totalN) { stage = "index"; indexDone = doneN; indexTotal = totalN; render(); },
  };
}

/** Gemini watches the video in 5-minute windows (three at a time) and writes timestamped
 *  notes; the notes are indexed like PDF pages whose "page" is the second in the video. */
async function addVideo(url, line) {
  line.textContent = "Checking the link…";
  const info = await api("/api/youtube/info", { url });
  const source = `${info.title} (YouTube)`;
  // If YouTube told us the length we know every window up front; otherwise keep
  // starting windows until one reports the end of the video.
  const WINDOW = 300, PARALLEL = 3;
  const limit = info.duration ? info.duration : 3 * 3600;
  const total = info.duration ? Math.ceil(info.duration / WINDOW) : null;
  const prog = videoProgress(line, info.title, total);
  if (total) for (let s = 0; s < info.duration; s += WINDOW) prog.add(s, Math.min(s + WINDOW, info.duration));
  prog.render();
  const segments = [];
  let next = 0, endAt = Infinity;
  async function worker() {
    while (true) {
      const start = next;
      if (start >= endAt || start >= limit) return;
      next += WINDOW;
      const part = prog.parts.find((p) => p.start === start) || prog.add(start, start + WINDOW);
      prog.set(part, "watching");
      const r = await apiPatient("/api/youtube/window", { video_id: info.video_id, start },
        () => prog.set(part, "retrying"));
      segments.push(...r.segments);
      if (r.end) endAt = Math.min(endAt, start);
      prog.set(part, r.segments.length ? "done" : "empty");
      prog.addNotes(r.segments.length);
    }
  }
  await Promise.all(Array.from({ length: PARALLEL }, worker));
  segments.sort((a, b) => a.t - b.t);
  if (!segments.length) throw new Error("Gemini couldn't read any content from this video.");
  const pages = segments.map((g) => ({ page: g.t, text: `[${fmtTs(g.t)}] ${g.text}` }));
  const chunks = [];
  prog.indexing(0, pages.length);
  for (let i = 0; i < pages.length; i += 20) {
    const r = await api("/api/index", { source, pages: pages.slice(i, i + 20) });
    chunks.push(...r.chunks);
    prog.indexing(Math.min(i + 20, pages.length), pages.length);
  }
  const last = segments[segments.length - 1].t;
  const doc = {
    source, subject: currentSubject, kind: "video", videoId: info.video_id, author: info.author, url: info.url,
    duration: info.duration || last + 30, pages: segments.length, model: CONFIG.embedding_model, dim: CONFIG.embedding_dim,
    chunks: chunks.map((c) => ({ page: c.page, chunk: c.chunk, text: c.text, embedding: Float32Array.from(c.embedding) })),
  };
  await store.put(doc);
  docs.set(source, { ...doc, builtIn: false });
  return doc;
}

$("ytForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("ytUrl"), url = input.value.trim();
  if (!url) return;
  const btn = $("ytBtn"); btn.disabled = true;
  const status = $("uploadStatus"); status.innerHTML = "";
  const line = document.createElement("div"); status.appendChild(line);
  try {
    const d = await addVideo(url, line);
    line.innerHTML = `<span class="ok">${ico("check")}</span> ${esc(d.source)}: ${fmtTs(d.duration)} watched, ${d.chunks.length} chunks — ready for questions`;
    input.value = "";
  } catch (err) {
    line.innerHTML = `<span class="err">${ico("alert")} ${esc(err.message)}</span>`;
  }
  btn.disabled = false;
  renderDocs();
});

/* in-app player that starts at the cited second */
function openPlayer(videoId, t, title) {
  if (!/^[A-Za-z0-9_-]{11}$/.test(videoId || "")) return;
  $("playerTitle").textContent = `${title || "Lecture"} — from ${fmtTs(t)}`;
  $("playerFrame").src = `https://www.youtube-nocookie.com/embed/${videoId}?start=${Math.floor(t)}&autoplay=1&rel=0`;
  $("playerOpen").href = `https://www.youtube.com/watch?v=${videoId}&t=${Math.floor(t)}s`;
  $("player").hidden = false;
}
function closePlayer() { $("player").hidden = true; $("playerFrame").src = "about:blank"; }
$("playerClose").addEventListener("click", closePlayer);
$("player").addEventListener("click", (e) => { if (e.target === $("player")) closePlayer(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("player").hidden) closePlayer(); });

document.addEventListener("click", (e) => {
  const ts = e.target.closest(".ts");
  if (ts) { openPlayer(ts.dataset.vid, +ts.dataset.t, ts.dataset.title); return; }
  const cite = e.target.closest(".cite");
  if (!cite) return;
  const bubble = cite.closest(".bubble");
  const src = (bubble._sources || []).find((s) => s.n === +cite.dataset.n);
  if (src && src.kind === "video") { openPlayer(src.video_id, src.page, src.source); return; }
  const det = bubble.querySelector("details.sources");
  if (det) {
    det.open = true;
    const row = det.querySelector(`.src[data-n="${cite.dataset.n}"]`);
    if (row) { row.scrollIntoView({ block: "center", behavior: "smooth" }); row.classList.add("flash"); setTimeout(() => row.classList.remove("flash"), 1200); }
  }
});

/* ---------- PDF ingestion ---------- */
const SCANNED_CHARS = 100;      // a page with less text than this is treated as a scan
const VISION_BATCH = 3, VISION_PARALLEL = 2;

async function extractPages(pdf) {
  const O = pdfjsLib.OPS;
  const imageOps = new Set([O.paintImageXObject, O.paintInlineImageXObject, O.paintImageXObjectRepeat,
                            O.paintJpegXObject, O.paintImageMaskXObject].filter((x) => x !== undefined));
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    let text = content.items.map((it) => it.str + (it.hasEOL ? "\n" : "")).join("");
    text = text.replace(/­/g, "").replace(/-\n(\w)/g, "$1")
               .replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
    const ops = await page.getOperatorList();
    pages.push({ page: i, text, hasImages: ops.fnArray.some((f) => imageOps.has(f)) });
  }
  return pages;
}

/** Draw a canvas source (PDF page or photo) to a white-backed JPEG data URL under ~1 MB. */
function toJpeg(draw, w, h) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.floor(w); canvas.height = Math.floor(h);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  return Promise.resolve(draw(ctx)).then(() => {
    let q = 0.75, url = canvas.toDataURL("image/jpeg", q);
    while (url.length > 1_400_000 && q > 0.35) { q -= 0.1; url = canvas.toDataURL("image/jpeg", q); }
    canvas.width = canvas.height = 0;
    return url;
  });
}

/** Render one PDF page to a JPEG data URL (default about 1100 px wide; exam papers use more for small print). */
async function renderPage(pdf, n, width = 1100) {
  const page = await pdf.getPage(n);
  const base = page.getViewport({ scale: 1 });
  const viewport = page.getViewport({ scale: Math.min(2.5, width / base.width) });
  return toJpeg((ctx) => page.render({ canvasContext: ctx, viewport }).promise, viewport.width, viewport.height);
}

/** Read a photo (JPG/PNG/HEIC that the browser can open) as a JPEG data URL, at most `width` px wide. */
async function renderImage(file, width = 1600) {
  const bmp = await createImageBitmap(file);
  const k = Math.min(1, width / bmp.width, 2200 / bmp.height);
  const url = await toJpeg((ctx) => ctx.drawImage(bmp, 0, 0, bmp.width * k, bmp.height * k), bmp.width * k, bmp.height * k);
  bmp.close && bmp.close();
  return url;
}

/** Let Gemini read page images. jobs: [{p: {page, text}, mode: "transcribe"|"visuals", image: () => Promise<dataURL>}].
 *  A batch that fails is retried one page at a time, so one bad page never loses its neighbours. */
async function visionRead(source, jobs, onProgress) {
  let seen = 0, transcribed = 0, described = 0;
  const failed = [];
  const apply = (batch, r) => {
    for (const j of batch) {
      const got = (r.pages.find((x) => x.page === j.p.page) || {}).text || "";
      if (!got) { if (j.mode === "transcribe") failed.push(j); continue; }
      if (j.mode === "transcribe") { j.p.text = got; transcribed++; }
      else { j.p.text += "\n\n" + got; described++; }
    }
  };
  const send = async (batch) => {
    const payload = [];
    for (const j of batch) payload.push({ page: j.p.page, mode: j.mode, text: j.mode === "visuals" ? j.p.text : "", image: await j.image() });
    return apiPatient("/api/vision/pages", { source, pages: payload });
  };
  const batches = [];
  for (let i = 0; i < jobs.length; i += VISION_BATCH) batches.push(jobs.slice(i, i + VISION_BATCH));
  onProgress(0, jobs.length);
  await runLimited(batches.map((batch) => async () => {
    try { apply(batch, await send(batch)); }
    catch (e) {
      if (/quota for today/.test(e.message)) throw e;
      for (const j of batch) {                       // retry page by page
        try { apply([j], await send([j])); } catch (e2) { console.warn("vision failed for page", j.p.page, e2); if (j.mode === "transcribe") failed.push(j); }
      }
    }
    seen += batch.length; onProgress(seen, jobs.length);
  }), VISION_PARALLEL);
  return { transcribed, described, failed: failed.length };
}

/** Read a PDF into pages of text, sending scanned pages (and, optionally, pictures) to Gemini vision.
 *  opts.paper: exam papers — pages that are mostly image with little text are treated as scans too,
 *  rendered sharper, and every page must end up readable. */
async function readPdf(file, onStatus, opts = {}) {
  const source = file.name;
  onStatus(`${source}: reading PDF…`);
  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  const all = await extractPages(pdf);
  const readImages = opts.paper || ($("readImages") ? $("readImages").checked : true);
  const scanLimit = opts.paper ? 400 : SCANNED_CHARS;   // scanned papers often carry a stray line of text ("Scanned with …")
  const width = opts.paper ? 1600 : 1100;
  const jobs = [];
  for (const p of all) {
    const image = () => renderPage(pdf, p.page, width);
    if (p.text.length < SCANNED_CHARS || (opts.paper && p.hasImages && p.text.length < scanLimit)) jobs.push({ p, mode: "transcribe", image });
    else if (readImages && p.hasImages && !opts.paper) jobs.push({ p, mode: "visuals", image });
  }
  let v = { transcribed: 0, described: 0, failed: 0 };
  if (jobs.length) {
    v = await visionRead(source, jobs, (done, n) => onStatus(`${source}: Gemini is reading ${n} scanned/picture page${n === 1 ? "" : "s"} — ${done} of ${n} done…`));
  }
  const pages = all.filter((p) => p.text.trim()).map(({ page, text }) => ({ page, text }));
  if (!pages.length) throw new Error("no readable text, even from the page images");
  return { source, pages, numPages: pdf.numPages, ...v };
}

/** Photos of one paper → pages of text (every photo is transcribed). */
async function readPhotos(files, name, onStatus) {
  const all = files.map((f, i) => ({ page: i + 1, text: "" }));
  const jobs = all.map((p, i) => ({ p, mode: "transcribe", image: () => renderImage(files[i]) }));
  const v = await visionRead(name, jobs, (done, n) => onStatus(`${name}: Gemini is reading ${n} photo${n === 1 ? "" : "s"} — ${done} of ${n} done…`));
  const pages = all.filter((p) => p.text.trim());
  if (!pages.length) throw new Error("Gemini couldn't read any text from these photos — try clearer, well-lit pictures");
  return { source: name, pages, numPages: files.length, ...v };
}

/** Chunk + embed pages on the server and save the document in this browser. */
async function indexDoc(read, onStatus, extra = {}) {
  const { source, pages, numPages } = read;
  const chunks = [];
  for (let i = 0; i < pages.length; i += PAGES_PER_REQUEST) {
    onStatus(`${source}: indexing page ${pages[Math.min(i + PAGES_PER_REQUEST, pages.length) - 1].page} of ${numPages}…`);
    const r = await api("/api/index", { source, pages: pages.slice(i, i + PAGES_PER_REQUEST) },
      (s) => onStatus(`${source}: Gemini rate limit, retrying in ${s}s…`));
    chunks.push(...r.chunks);
  }
  const doc = {
    source, subject: currentSubject, pages: pages.length, imagePages: read.transcribed + read.described, scannedPages: read.transcribed,
    model: CONFIG.embedding_model, dim: CONFIG.embedding_dim, ...extra,
    chunks: chunks.map((c) => ({ page: c.page, chunk: c.chunk, text: c.text,
                                 embedding: Float32Array.from(c.embedding) })),
  };
  await store.put(doc);
  docs.set(source, { ...doc, builtIn: false });
  return doc;
}

async function ingest(file, line) {
  const status = (t) => { line.textContent = t; };
  return indexDoc(await readPdf(file, status), status);
}

async function upload(files) {
  if (!files.length) return;
  const status = $("uploadStatus");
  status.innerHTML = "";
  showView("home");
  for (const file of files) {
    const line = document.createElement("div");
    status.appendChild(line);
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      line.innerHTML = `<span class="err">${ico("alert")} ${esc(file.name)}: only PDF files are supported</span>`;
      continue;
    }
    try {
      const d = await ingest(file, line);
      const img = d.imagePages ? `, ${d.imagePages} read from images` : "";
      line.innerHTML = `<span class="ok">${ico("check")}</span> ${esc(d.source)}: ${d.pages} pages${img}, ${d.chunks.length} chunks — ready for questions`;
    } catch (e) {
      line.innerHTML = `<span class="err">${ico("alert")} ${esc(file.name)}: ${esc(e.message)}</span>`;
    }
    renderDocs();
  }
  $("fileInput").value = "";
}

async function removeDoc(source) {
  if (!confirm(`Remove ${source}?`)) return;
  await store.remove(source);
  docs.delete(source);
  renderDocs();
}

/* ---------- retrieval (cosine similarity over stored vectors) ---------- */
function search(queryVec, scope) {
  const q = Float32Array.from(queryVec);
  const hits = [];
  for (const d of docs.values()) {
    if (subjectOf(d) !== currentSubject) continue;
    if (scope && d.source !== scope) continue;
    for (const c of d.chunks) {
      let s = 0;
      for (let i = 0; i < q.length; i++) s += q[i] * c.embedding[i]; // unit vectors: dot = cosine
      if (s >= CONFIG.min_score) {
        hits.push({ source: d.source, page: c.page, text: c.text, score: s,
                    kind: d.kind === "video" ? "video" : "pdf", video_id: d.videoId || "" });
      }
    }
  }
  hits.sort((a, b) => b.score - a.score);
  return hits.slice(0, CONFIG.top_k);
}

/* ---------- summaries (map-reduce: detailed notes per section, then an overview) ---------- */

/** Rebuild each page's text from its chunks, dropping the overlap between neighbours. */
function pagesFromChunks(chunks) {
  const byPage = new Map();
  for (const c of [...chunks].sort((a, b) => a.page - b.page || a.chunk - b.chunk)) {
    const prev = byPage.get(c.page);
    if (prev === undefined) { byPage.set(c.page, c.text); continue; }
    let k = Math.min(prev.length, c.text.length, 400);
    while (k > 20 && !prev.endsWith(c.text.slice(0, k))) k--;
    byPage.set(c.page, prev + " " + (k > 20 ? c.text.slice(k) : c.text));
  }
  return [...byPage].map(([page, text]) => ({ page, text: text.trim() }));
}

/** Group consecutive pages into sections of about SECTION_CHARS characters. */
function makeSections(pages) {
  const sections = [];
  let cur = [], size = 0;
  for (const p of pages) {
    if (cur.length && size + p.text.length > SECTION_CHARS) { sections.push(cur); cur = []; size = 0; }
    cur.push(p); size += p.text.length;
  }
  if (cur.length) sections.push(cur);
  return sections;
}

/** Run async tasks with at most `limit` in flight, keeping results in order. */
async function runLimited(tasks, limit) {
  const results = new Array(tasks.length);
  let next = 0;
  async function worker() {
    while (next < tasks.length) { const i = next++; results[i] = await tasks[i](); }
  }
  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

/** Call the API, retrying once more if Gemini was busy (502/503). */
async function apiPatient(url, body, onWait) {
  try { return await api(url, body, onWait); }
  catch (e) {
    if (/quota for today/.test(e.message)) throw e; // retrying won't help until tomorrow
    if (onWait) onWait(10);
    await sleep(10000);
    return api(url, body, onWait);
  }
}

const pageRange = (a, b, video) => video ? `${fmtTs(a)}–${fmtTs(b)}`
  : (a === b ? `Page ${a}` : `Pages ${a}–${b}`);

async function summarise(source) {
  const doc = docs.get(source);
  const video = doc.kind === "video", kind = video ? "video" : "pdf";
  addMessage("user", `Summarise <b>${esc(source)}</b>`);
  const bubble = addMessage("bot", '<span class="typing">Reading the document…</span>');
  const sections = makeSections(pagesFromChunks(doc.chunks));
  const notes = new Array(sections.length).fill(null);
  let status = "";
  const draw = () => {
    const done = notes.filter((n) => n !== null).length;
    let html = `<div class="progress">${esc(status || `Writing detailed notes: ${done} of ${sections.length} sections done…`)}</div>`;
    html += md(notes.filter((n) => n !== null && !n.error).map((n) => n.notes).join("\n\n"));
    bubble.innerHTML = html;
  };
  draw();

  await runLimited(sections.map((pages, i) => async () => {
    const range = pageRange(pages[0].page, pages[pages.length - 1].page, video);
    try {
      notes[i] = await apiPatient("/api/summary/section", { source, pages, kind },
        () => { status = `Gemini is busy — retrying ${range}…`; draw(); });
    } catch (e) {
      notes[i] = { error: e.message, start_page: pages[0].page, end_page: pages[pages.length - 1].page,
                   notes: `> ⚠️ Could not summarise ${range}: ${e.message}` };
    }
    status = "";
    draw();
  }), 2); // 2 at a time keeps within Gemini's per-minute limits

  const ok = notes.filter((n) => !n.error);
  let overview = "";
  if (ok.length) {
    status = "Writing the overview and key takeaways…";
    draw();
    try {
      overview = (await apiPatient("/api/summary/overview",
        { source, total_pages: video ? doc.duration || 0 : doc.pages, sections: ok, kind })).overview;
    } catch (e) {
      overview = `> ⚠️ Could not write the overview: ${e.message}`;
    }
  }

  const full = `# Summary: ${source}\n\n${overview}\n\n## Detailed notes\n\n` +
               notes.map((n) => n.notes).join("\n\n");
  bubble.innerHTML = md(full);
  const failed = notes.length - ok.length;
  if (failed) {
    bubble.insertAdjacentHTML("afterbegin",
      `<div class="error">${failed} section(s) could not be summarised because Gemini was busy. Click Summarise again to retry.</div>`);
  }
  const a = document.createElement("a");
  a.className = "download";
  a.textContent = "Download summary (.md)";
  a.download = source.replace(/\.pdf$/i, "").replace(/[\\/:*?"<>|]/g, "") + " - summary.md";
  a.href = URL.createObjectURL(new Blob([full], { type: "text/markdown" }));
  bubble.appendChild(a);
}

/* ---------- chat ---------- */
function where(s) {
  return s.kind === "video"
    ? `<button class="ts" data-vid="${esc(s.video_id || "")}" data-t="${s.page}" data-title="${esc(s.source)}">${ico("play")}${fmtTs(s.page)}</button>`
    : `page ${s.page}`;
}

function renderAnswer(r, retrieval) {
  // make inline citations like [3] clickable: videos jump to the moment, PDFs open the sources list
  let html = md(r.answer).replace(/\[(\d{1,2})\](?![^<]*>)/g, (m, n) =>
    r.sources.some((s) => s.n === +n) ? `<button class="cite" data-n="${n}">[${n}]</button>` : m);
  if (r.sources.length) {
    html += `<details class="sources"><summary>Sources (${r.sources.length})</summary>` +
      r.sources.map((s) => `<div class="src" data-n="${s.n}"><b>[${s.n}]</b> ${esc(s.source)}, ${where(s)}
        <span class="meta">· similarity ${s.score.toFixed(2)}</span>
        <div class="ex">${esc(s.excerpt)}…</div></div>`).join("") + "</details>";
  }
  html += `<div class="timing">retrieval ${retrieval.toFixed(2)}s · generation ${r.timings.generation}s</div>`;
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
    if (!subjectDocs().length) throw new Error(`“${subjectName()}” has no documents yet — upload a PDF or add a video first.`);
    const t0 = performance.now();
    const { embedding } = await api("/api/embed-query", { question: q });
    const passages = search(embedding, $("scope").value);
    const retrieval = (performance.now() - t0) / 1000;
    bubble.innerHTML = '<span class="typing">Writing an answer…</span>';
    const r = await api("/api/answer", { question: q, passages });
    bubble.innerHTML = renderAnswer(r, retrieval);
    bubble._sources = r.sources;
    bubble.closest(".msg").scrollIntoView({ block: "start", behavior: "smooth" });
  } catch (err) {
    bubble.innerHTML = `<span class="error">${esc(err.message)}</span>`;
  }
  $("sendBtn").disabled = false;
  $("question").focus();
});

const question = $("question");
function autoGrow() { question.style.height = "auto"; question.style.height = Math.min(question.scrollHeight, 140) + "px"; }
question.addEventListener("input", autoGrow);
question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chatForm").requestSubmit(); autoGrow(); }
});

/* ---------- upload widgets ---------- */
$("fileInput").addEventListener("change", (e) => upload([...e.target.files]));
const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => upload([...e.dataTransfer.files]));

/* ---------- viva practice ---------- */
const viva = { results: [], used: new Set(), current: null, followUp: null, n: 0, total: 5, level: "medium", scope: "" };

/** Pick the chunks of one random, not-yet-used page (or video moment) as the question's material. */
function vivaPassages() {
  const groups = new Map();
  for (const d of docs.values()) {
    if (subjectOf(d) !== viva.subject) continue;
    if (viva.scope && d.source !== viva.scope) continue;
    if (!viva.scope && d.role === "syllabus" && subjectDocs(viva.subject).length > 1) continue; // ask from notes/lectures
    if (!viva.scope && d.role === "paper") continue;                                         // past papers are questions, not material
    for (const c of d.chunks) {
      // videos: group notes into ~2-minute stretches; PDFs: one group per page
      const key = d.kind === "video" ? `${d.source}|${Math.floor(c.page / 120)}` : `${d.source}|${c.page}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({ source: d.source, page: c.page, text: c.text, chunk: c.chunk,
                             kind: d.kind === "video" ? "video" : "pdf", video_id: d.videoId || "" });
    }
  }
  let keys = [...groups.keys()].filter((k) => !viva.used.has(k));
  if (!keys.length) { viva.used.clear(); keys = [...groups.keys()]; }
  const key = keys[Math.floor(Math.random() * keys.length)];
  viva.used.add(key);
  return groups.get(key).sort((a, b) => a.page - b.page || a.chunk - b.chunk).slice(0, 4);
}

function hush() {
  if ("speechSynthesis" in window && (speechSynthesis.speaking || speechSynthesis.pending)) speechSynthesis.cancel();
}

function speak(text) {
  if (!$("vivaVoice").checked || !("speechSynthesis" in window)) return;
  hush();
  const u = new SpeechSynthesisUtterance(text);
  const voices = speechSynthesis.getVoices();
  u.voice = voices.find((v) => /en-IN/i.test(v.lang)) || voices.find((v) => /^en/i.test(v.lang)) || null;
  u.rate = 0.98;
  speechSynthesis.speak(u);
}

/* voice answers: Web Speech API (Chrome, Edge, Safari) */
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, recBase = "";
function stopMic() {
  if (rec) { rec.onend = null; rec.stop(); rec = null; }
  $("vivaMic").classList.remove("on");
  $("vivaMic").querySelector("span").textContent = "Speak answer";
  $("vivaMicStatus").textContent = "";
}
$("vivaMic").addEventListener("click", () => {
  if (rec) { stopMic(); return; }
  if (!Recognition) { $("vivaMicStatus").textContent = "Voice input isn't supported in this browser — please type your answer (Chrome or Edge support it)."; return; }
  hush();
  rec = new Recognition();
  rec.lang = "en-IN"; rec.continuous = true; rec.interimResults = true;
  recBase = $("vivaAnswer").value ? $("vivaAnswer").value.trim() + " " : "";
  rec.onresult = (e) => {
    let finalText = "", interim = "";
    for (let i = 0; i < e.results.length; i++) {
      (e.results[i].isFinal ? (finalText += e.results[i][0].transcript + " ") : (interim += e.results[i][0].transcript));
    }
    $("vivaAnswer").value = recBase + finalText + interim;
  };
  rec.onerror = (e) => { $("vivaMicStatus").textContent = e.error === "not-allowed" ? "Microphone access was blocked — allow it in the browser, or type instead." : `Voice input stopped (${e.error}).`; };
  rec.onend = () => { if (rec) rec.start(); };   // keep listening through pauses until the student stops
  rec.start();
  $("vivaMic").classList.add("on");
  $("vivaMic").querySelector("span").textContent = "Stop";
  $("vivaMicStatus").textContent = "Listening… speak your answer, then press Stop.";
});

function vivaBusy(on, text) {
  ["vivaSubmit", "vivaSkip", "vivaMic"].forEach((id) => { $(id).disabled = on; });
  if (text !== undefined) $("vivaMicStatus").innerHTML = text;
}

function syllabusText() {
  const s = syllabusOf(viva.subject);
  return s ? pagesFromChunks(s.chunks).map((p) => p.text).join("\n").slice(0, 3000) : "";
}

function fetchQuestion(passages, follow) {
  return apiPatient("/api/viva/question", {
    passages, difficulty: viva.level, syllabus: viva.syllabus,
    previous: [...viva.results.map((x) => x.question), ...(viva.current ? [viva.current.question] : [])],
    follow_up: follow ? { question: viva.current.question, answer: follow.answer, missing: follow.missing } : null,
  }).then((r) => ({ ...r, passages }));
}

/** Prepare the next (non-follow-up) question while the student is answering this one. */
function prefetchNext() {
  if (viva.n >= viva.total) { viva.prefetch = null; return; }
  const p = fetchQuestion(vivaPassages(), null);
  p.catch(() => {});                      // a failed prefetch is simply fetched again later
  viva.prefetch = p;
}

async function vivaAsk() {
  stopMic();
  $("vivaNext").hidden = true;
  $("vivaFeedback").innerHTML = "";
  $("vivaAnswer").value = "";
  viva.n++;
  const follow = viva.followUp;
  $("vivaStep").textContent = `Question ${viva.n} of ${viva.total}${follow ? " · follow-up" : ""}`;
  $("vivaQuestion").innerHTML = '<span class="typing">The examiner is preparing a question…</span>';
  vivaBusy(true, "");
  try {
    let r;
    if (follow) {
      r = await fetchQuestion(viva.current.passages, follow);    // keep the prepared one for later
    } else {
      const ready = viva.prefetch;
      viva.prefetch = null;
      r = ready ? await ready.catch(() => fetchQuestion(vivaPassages(), null)) : await fetchQuestion(vivaPassages(), null);
    }
    viva.current = { question: r.question, topic: r.topic, passages: r.passages, isFollowUp: !!follow, at: Date.now() };
    viva.followUp = null;
    $("vivaQuestion").textContent = r.question;
    vivaBusy(false, "");
    speak(r.question);
    $("vivaAnswer").focus();
    if (!viva.prefetch) prefetchNext();
  } catch (e) {
    $("vivaQuestion").innerHTML = `<span class="error">${esc(e.message)}</span>`;
    viva.n--;
    $("vivaNext").hidden = false; $("vivaNext").textContent = "Try again";
    vivaBusy(false, "");
  }
}

function fbList(items) {
  return items.length ? "<ul>" + items.map((x) => `<li>${md(x).replace(/^<p>|<\/p>\s*$/g, "")}</li>`).join("") + "</ul>" : "<p>—</p>";
}

async function vivaSubmit(skipped) {
  stopMic();
  const answer = skipped ? "" : $("vivaAnswer").value.trim();
  if (!skipped && !answer) { $("vivaMicStatus").textContent = "Speak or type an answer first (or press Skip)."; return; }
  hush();
  vivaBusy(true, '<span class="typing">The examiner is marking your answer…</span>');
  const cur = viva.current;
  let g;
  try {
    g = await apiPatient("/api/viva/grade", { question: cur.question, answer, passages: cur.passages, difficulty: viva.level });
  } catch (e) {
    vivaBusy(false, `<span class="error">${esc(e.message)}</span>`);
    return;
  }
  viva.results.push({ question: cur.question, topic: cur.topic, answer, ...g, followUp: cur.isFollowUp });
  const sources = cur.passages.map((p, i) => ({ n: i + 1, source: p.source, page: p.page, kind: p.kind, video_id: p.video_id, score: 1, excerpt: p.text.slice(0, 200) }));
  const cls = g.score >= 7 ? "good" : g.score >= 4 ? "mid" : "low";
  const fb = document.createElement("div");
  fb.className = "bubble viva-fb";
  fb._sources = sources;
  const withCites = (html) => html.replace(/\[(\d{1,2})\](?![^<]*>)/g, (m, n) => sources.some((s) => s.n === +n) ? `<button class="cite" data-n="${n}">[${n}]</button>` : m);
  fb.innerHTML = withCites(`<span class="score ${cls}">${g.score}/10</span>
    <p><b>${esc(g.verdict || "")}</b></p>
    <div class="fb-h ok">What you got right</div>${fbList(g.strengths)}
    <div class="fb-h miss">What you missed</div>${fbList(g.missing)}
    ${g.model_answer ? `<div class="fb-h">Model answer</div>${md(g.model_answer)}` : ""}`) +
    `<details class="sources"><summary>Material used (${sources.length})</summary>` +
    sources.map((s) => `<div class="src" data-n="${s.n}"><b>[${s.n}]</b> ${esc(s.source)}, ${where(s)}<div class="ex">${esc(s.excerpt)}…</div></div>`).join("") + "</details>";
  $("vivaFeedback").innerHTML = "";
  $("vivaFeedback").appendChild(fb);
  vivaBusy(false, "");
  $("vivaRunning").textContent = `Running score: ${viva.results.reduce((a, r) => a + r.score, 0)} / ${viva.results.length * 10}`;
  if (g.follow_up && !cur.isFollowUp && viva.n < viva.total) viva.followUp = { answer, missing: g.missing };
  const more = viva.n < viva.total;
  $("vivaNext").hidden = false;
  $("vivaNext").textContent = more ? (viva.followUp ? "Follow-up question" : "Next question") : "See my report";
  speak(`${g.score} out of 10. ${g.verdict || ""}`);
}

function vivaReport() {
  stopMic();
  hush();
  viva.prefetch = null;
  $("vivaSession").hidden = true;
  const rs = viva.results;
  const answered = rs.filter((r) => r.answer).length;
  const total = rs.reduce((a, r) => a + r.score, 0), max = rs.length * 10;
  const pct = max ? Math.round((100 * total) / max) : 0;
  const letter = pct >= 90 ? "A+" : pct >= 80 ? "A" : pct >= 70 ? "B+" : pct >= 60 ? "B" : pct >= 50 ? "C" : pct >= 40 ? "D" : "F";
  const verdict = !rs.length ? "No questions answered yet"
    : pct >= 85 ? "Excellent — ready for the viva" : pct >= 65 ? "Good — revise the weak topics"
    : pct >= 40 ? "Needs work — revise and try again" : "Not ready yet — study the material first";
  const mins = Math.max(1, Math.round((Date.now() - viva.startedAt) / 60000));
  const weak = [...new Set(rs.filter((r) => r.score <= 5).map((r) => r.topic || r.question.slice(0, 40)))];
  const strong = [...new Set(rs.filter((r) => r.score >= 8).map((r) => r.topic || r.question.slice(0, 40)))];
  const cls = (s) => (s >= 7 ? "good" : s >= 4 ? "mid" : "low");

  const mdReport = `# Viva practice report — ${subjectName(viva.subject)}\n\n` +
    `**Score:** ${total}/${max} (${pct}%) · **Grade:** ${letter} · ${verdict}\n\n` +
    `Questions: ${rs.length} · answered: ${answered} · skipped: ${rs.length - answered} · time: ${mins} min · difficulty: ${viva.level}\n\n` +
    rs.map((r, i) => `## Q${i + 1}${r.followUp ? " (follow-up)" : ""}: ${r.question}\n\n**Your answer:** ${r.answer || "(skipped)"}\n\n**Score:** ${r.score}/10 — ${r.verdict}\n\n**What you got right:**\n${r.strengths.map((m) => "- " + m).join("\n") || "- —"}\n\n**What you missed:**\n${r.missing.map((m) => "- " + m).join("\n") || "- nothing"}\n\n**Model answer:** ${r.model_answer}\n`).join("\n") +
    (weak.length ? `\n## Topics to revise\n${weak.map((w) => "- " + w).join("\n")}\n` : "") +
    (strong.length ? `\n## Strong topics\n${strong.map((w) => "- " + w).join("\n")}\n` : "");

  const box = $("vivaReport");
  box.hidden = false;
  box.innerHTML = `<div class="card report-card">
      <div class="report-top">
        <div class="ring ${rs.length ? cls(total / Math.max(1, rs.length)) : "low"}" style="--p:${pct}"><div><b>${pct}%</b><span>${letter}</span></div></div>
        <div><div class="examiner-label">Viva result · ${esc(subjectName(viva.subject))}</div>
          <h2 style="margin:4px 0">${total} / ${max} marks</h2>
          <p style="margin:0">${esc(verdict)}</p>
          <p class="note-small" style="margin-top:6px">${rs.length} question(s) · ${answered} answered · ${rs.length - answered} skipped · ${mins} min · ${esc(viva.level)}</p></div>
      </div>
      ${rs.length ? `<table class="report-table"><tr><th>#</th><th>Question</th><th>Score</th></tr>
        ${rs.map((r, i) => `<tr><td>${i + 1}</td><td>${esc(r.question)}${r.followUp ? " <i>(follow-up)</i>" : ""}
          <details><summary>Feedback</summary><p><b>${esc(r.verdict)}</b></p>
          <div class="fb-h miss">Missed</div>${fbList(r.missing)}${r.model_answer ? `<div class="fb-h">Model answer</div>${md(r.model_answer)}` : ""}</details></td>
          <td><span class="pill-score ${cls(r.score)}">${r.score}/10</span></td></tr>`).join("")}</table>` : ""}
      <div class="two-col">
        <div><h4>Topics to revise</h4>${weak.length ? `<ul>${weak.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : "<p>None — well done!</p>"}</div>
        <div><h4>Strong topics</h4>${strong.length ? `<ul>${strong.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : "<p>—</p>"}</div>
      </div>
      <div class="row" style="margin-top:14px"><a class="btn" id="vivaDownload">${ico("file")}Download report</a><button class="btn primary" id="vivaAgain" type="button">Practise again</button></div></div>`;
  const a = $("vivaDownload");
  a.download = `StudyMate viva report - ${subjectName(viva.subject).replace(/[\\/:*?"<>|]/g, "")}.md`;
  a.href = URL.createObjectURL(new Blob([mdReport], { type: "text/markdown" }));
  $("vivaAgain").onclick = () => { box.hidden = true; $("vivaSetup").hidden = false; };
  box.scrollIntoView({ block: "start" });
  if (rs.length) speak(`Viva complete. You scored ${total} out of ${max}. ${verdict}.`);
}

$("vivaStart").addEventListener("click", () => {
  if (!subjectDocs().length) return;
  Object.assign(viva, { results: [], used: new Set(), current: null, followUp: null, n: 0, prefetch: null,
    total: +$("vivaCount").value, level: $("vivaLevel").value, scope: $("vivaScope").value,
    subject: currentSubject, startedAt: Date.now() });
  viva.syllabus = syllabusText();
  $("vivaSetup").hidden = true; $("vivaReport").hidden = true; $("vivaSession").hidden = false;
  $("vivaRunning").textContent = "";
  vivaAsk();
});
$("vivaSubmit").addEventListener("click", () => vivaSubmit(false));
$("vivaSkip").addEventListener("click", () => vivaSubmit(true));
$("vivaNext").addEventListener("click", () => (viva.n < viva.total || !viva.current ? vivaAsk() : vivaReport()));
$("vivaEnd").addEventListener("click", () => vivaReport());   // always show the score so far
$("vivaSpeak").addEventListener("click", () => { const v = $("vivaVoice").checked; $("vivaVoice").checked = true; speak($("vivaQuestion").textContent); $("vivaVoice").checked = v; });
if ("speechSynthesis" in window) speechSynthesis.getVoices();

/* ---------- Exam Predictor: past papers -> topics -> likely questions ---------- */
const papersOf = (id = currentSubject) => subjectDocs(id).filter((d) => d.role === "paper")
  .sort((a, b) => ((a.paper && a.paper.year) || 9999) - ((b.paper && b.paper.year) || 9999) || a.source.localeCompare(b.source));
let predicting = false;

/** Read the questions out of a paper (text from its chunks) and remember them on the document. */
async function extractPaper(d, onStatus) {
  onStatus(`${d.source}: finding the questions…`);
  const pages = pagesFromChunks(d.chunks);
  const r = await apiPatient("/api/papers/extract", { source: d.source, pages },
    () => onStatus(`${d.source}: Gemini is busy, retrying…`));
  d.paper = { exam: r.exam, year: (d.paper && d.paper.yearSetByUser) ? d.paper.year : r.year, questions: r.questions,
              yearSetByUser: !!(d.paper && d.paper.yearSetByUser) };
  await saveDoc(d);
  return d;
}

async function addPapers(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const status = $("paperStatus");
  status.innerHTML = "";
  const pdfs = files.filter((f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name));
  const photos = files.filter((f) => f.type.startsWith("image/"));
  const other = files.filter((f) => !pdfs.includes(f) && !photos.includes(f));
  const groups = pdfs.map((f) => ({ name: f.name, run: (st) => readPdf(f, st, { paper: true }) }));
  if (photos.length) {
    const base = photos[0].name.replace(/\.[^.]+$/, "");
    const name = `${base}${photos.length > 1 ? ` (+${photos.length - 1} photos)` : ""}.jpg`;
    groups.push({ name, run: (st) => readPhotos(photos, name, st) });
  }
  for (const f of other) status.insertAdjacentHTML("beforeend", `<div><span class="err">${ico("alert")} ${esc(f.name)}: add a PDF or photos (JPG/PNG)</span></div>`);
  for (const g of groups) {
    const line = document.createElement("div");
    status.appendChild(line);
    const st = (t) => { line.textContent = t; };
    try {
      if (docs.has(g.name) && !confirm(`${g.name} is already added. Replace it?`)) { line.remove(); continue; }
      const read = await g.run(st);
      const doc = await indexDoc(read, st, { role: "paper" });
      await extractPaper(docs.get(doc.source), st);
      const p = docs.get(doc.source).paper;
      const scanned = read.transcribed ? ` · ${read.transcribed} scanned page${read.transcribed === 1 ? "" : "s"} read` : "";
      const lost = read.failed ? ` · ${read.failed} page${read.failed === 1 ? "" : "s"} could not be read` : "";
      line.innerHTML = `<span class="ok">${ico("check")}</span> ${esc(doc.source)}: ${plural(p.questions.length, "question")} found${p.year ? ` · ${p.year}` : ""}${scanned}${lost}`;
    } catch (e) {
      line.innerHTML = `<span class="err">${ico("alert")} ${esc(g.name)}: ${esc(e.message)}</span>`;
    }
    renderDocs();
  }
  $("paperInput").value = "";
}

function renderPapers() {
  if (!$("paperList")) return;
  $("predSubject").textContent = subjectName();
  const list = $("paperList");
  const papers = papersOf();
  list.innerHTML = papers.length ? "" : `<li class="muted-row">No past papers in “${esc(subjectName())}” yet. Add two or more years for the best prediction.</li>`;
  for (const d of papers) {
    const li = document.createElement("li");
    li.className = "doc paper-row";
    const p = d.paper;
    const meta = p ? `${plural(p.questions.length, "question")}${p.exam ? " · " + esc(p.exam) : ""}${d.scannedPages ? ` · ${d.scannedPages} scanned page${d.scannedPages === 1 ? "" : "s"}` : ""}`
                   : `<span class="err">questions not read yet</span>`;
    li.innerHTML = `<div class="doc-icon">${ico("file")}</div>
      <div><div class="name">${esc(d.source)}</div><div class="meta">${meta}</div></div>
      <div class="actions">
        <label class="year-in">Year <input type="number" min="1990" max="2100" placeholder="?" value="${p && p.year ? p.year : ""}"></label>
        ${p ? "" : `<button class="btn reread">Read questions</button>`}
        <button class="btn danger del" aria-label="Remove ${esc(d.source)}">${ico("trash")}</button>
      </div>`;
    li.querySelector(".del").onclick = () => removeDoc(d.source);
    li.querySelector(".year-in input").onchange = async (e) => {
      const y = parseInt(e.target.value, 10);
      d.paper = { ...(d.paper || { questions: [] }), year: y >= 1990 && y <= 2100 ? y : null, yearSetByUser: true };
      await saveDoc(d); renderDocs();
    };
    const rr = li.querySelector(".reread");
    if (rr) rr.onclick = async () => {
      rr.disabled = true;
      const line = document.createElement("div"); $("paperStatus").appendChild(line);
      try { await extractPaper(d, (t) => { line.textContent = t; }); line.innerHTML = `<span class="ok">${ico("check")}</span> ${esc(d.source)}: ${plural(d.paper.questions.length, "question")} found`; }
      catch (e) { line.innerHTML = `<span class="err">${ico("alert")} ${esc(e.message)}</span>`; }
      renderDocs();
    };
    list.appendChild(li);
  }
  const ready = papers.filter((d) => d.paper && d.paper.questions.length);
  $("predictBtn").disabled = !ready.length || predicting;
  $("predictHint").textContent = !ready.length ? "Add at least one past paper above."
    : ready.length === 1 ? "One paper gives a rough guess (no trend yet). Add more years for a real prediction."
    : `${ready.length} papers, ${ready.reduce((n, d) => n + d.paper.questions.length, 0)} questions ready${syllabusOf() ? " · topics will be mapped to your syllabus" : " · tip: set a syllabus PDF on Home to map topics to units"}.`;
  const sub = subjects.get(currentSubject);
  if (!predicting) {
    if (sub && sub.prediction) renderPrediction(sub.prediction, papersKey(ready) !== sub.prediction.key);
    else $("predictOut").innerHTML = "";
  }
}

const papersKey = (papers) => papers.map((d) => `${d.source}:${d.paper.year || ""}:${d.paper.questions.length}`).join("|");

/** Best passages in this subject's notes and lectures (not papers or the syllabus) for a query vector. */
function studyLinks(vec, k = 2) {
  const q = Float32Array.from(vec), best = [];
  for (const d of subjectDocs()) {
    if (d.role === "paper" || d.role === "syllabus") continue;
    let top = null;
    for (const c of d.chunks) {
      let s = 0;
      for (let i = 0; i < q.length; i++) s += q[i] * c.embedding[i];
      if (!top || s > top.score) top = { source: d.source, page: c.page, score: s, kind: d.kind === "video" ? "video" : "pdf", video_id: d.videoId || "" };
    }
    if (top && top.score >= CONFIG.min_score) best.push(top);
  }
  return best.sort((a, b) => b.score - a.score).slice(0, k);
}

async function runPrediction() {
  const papers = papersOf().filter((d) => d.paper && d.paper.questions.length);
  if (!papers.length || predicting) return;
  predicting = true; $("predictBtn").disabled = true;
  const out = $("predictOut");
  const step = (t) => { out.innerHTML = `<div class="card pred-wait"><span class="spin"></span> ${esc(t)}</div>`; };
  try {
    const nq = papers.reduce((n, d) => n + d.paper.questions.length, 0);
    step(`Grouping ${nq} questions from ${plural(papers.length, "paper")} into topics…`);
    const syl = syllabusOf();
    const result = await apiPatient("/api/papers/predict", {
      subject: subjectName(),
      syllabus: syl ? pagesFromChunks(syl.chunks).map((p) => p.text).join("\n").slice(0, 6000) : "",
      papers: papers.map((d) => ({ source: d.source, year: d.paper.year, questions: d.paper.questions })),
    }, () => step("Gemini is busy — retrying…"));
    step("Finding where your notes and lectures cover each topic…");
    const top = result.topics.slice(0, 40);
    try {
      const queries = top.map((t) => `${t.topic}. ${(t.predicted[0] || t.asked[0].text).slice(0, 300)}`);
      const { embeddings } = await api("/api/embed-query", { queries });
      top.forEach((t, i) => { t.links = studyLinks(embeddings[i]); });
    } catch (e) { console.warn("linking topics to notes failed", e); }
    const pred = { key: papersKey(papers), at: Date.now(), ...result };
    const sub = subjects.get(currentSubject);
    sub.prediction = pred;
    await store.putSubject(sub);
    renderPrediction(pred, false);
  } catch (e) {
    out.innerHTML = `<div class="card"><span class="error">${ico("alert")} ${esc(e.message)}</span></div>`;
  }
  predicting = false;
  renderPapers();
}

function renderPrediction(p, stale) {
  const out = $("predictOut");
  const years = p.papers.map((x) => x.year).filter(Boolean);
  const span = years.length ? (Math.min(...years) === Math.max(...years) ? `${years[0]}` : `${Math.min(...years)}–${Math.max(...years)}`) : "";
  const nq = p.papers.reduce((n, x) => n + x.questions, 0);
  const hot = p.topics.filter((t) => t.probability >= 70).length;
  const cls = (t) => t.probability >= 70 ? "hi" : t.probability >= 45 ? "mid" : "lo";
  const linkHtml = (l) => l.kind === "video"
    ? `<button class="study-link" data-video="${esc(l.video_id)}" data-t="${l.page}" data-title="${esc(l.source)}">${ico("play")}${esc(l.source.replace(/ \(YouTube\)$/, ""))} · ${fmtTs(l.page)}</button>`
    : `<span class="study-link">${ico("file")}${esc(l.source)} · p. ${l.page}</span>`;
  const topicCard = (t, i) => `<li class="pred-topic ${cls(t)}">
      <div class="pt-head">
        <span class="pt-rank">#${i + 1}</span>
        <div class="pt-name"><b>${esc(t.topic)}</b>${t.unit ? `<span class="pt-unit">${esc(t.unit)}</span>` : ""}</div>
        <div class="pt-prob"><span>${t.probability}%</span><small>${esc(t.label)}</small></div>
      </div>
      <div class="bar"><i style="width:${t.probability}%"></i></div>
      <div class="pt-stats">Asked in ${t.papers_asked} of ${p.papers.length} paper${p.papers.length === 1 ? "" : "s"}${t.years.length ? ` (${t.years.join(", ")})` : ""} · ${plural(t.times_asked, "question")}${t.marks ? ` · ${t.marks} marks in total` : ""}</div>
      ${t.predicted.length ? `<div class="pt-sub">Likely questions</div><ol class="pt-pred">${t.predicted.map((q) => `<li>${md(q).replace(/^<p>|<\/p>\s*$/g, "")}</li>`).join("")}</ol>` : ""}
      ${t.tip ? `<p class="pt-tip">${ico("spark")}${esc(t.tip)}</p>` : ""}
      ${t.links && t.links.length ? `<div class="pt-sub">Study it from</div><div class="pt-links">${t.links.map(linkHtml).join("")}</div>` : ""}
      <details class="pt-past"><summary>Past questions (${t.asked.length})</summary><ul>${t.asked.map((q) =>
        `<li><span class="pt-year">${q.year || "?"}${q.no ? " · Q" + esc(q.no.replace(/^Q/i, "")) : ""}${q.marks ? ` · ${q.marks}m` : ""}</span>${md(q.text).replace(/^<p>|<\/p>\s*$/g, "")}</li>`).join("")}</ul></details>
      <div class="row pt-actions"><button class="btn explain" data-topic="${esc(t.topic)}">${ico("book")}Explain this topic</button></div>
    </li>`;
  out.innerHTML = `
    ${stale ? `<div class="card stale">${ico("alert")} Your past papers changed since this prediction. <button class="btn primary" id="rerun">Predict again</button></div>` : ""}
    <div class="pred-stats">
      <div><b>${p.papers.length}</b><span>paper${p.papers.length === 1 ? "" : "s"}${span ? " · " + span : ""}</span></div>
      <div><b>${nq}</b><span>questions read</span></div>
      <div><b>${p.topics.length}</b><span>topics found</span></div>
      <div><b>${hot}</b><span>very likely topics</span></div>
    </div>
    <div class="section-head"><h3>${ico("target")}Most likely in your next exam</h3>
      <button class="btn" id="predDownload">Download report</button></div>
    <p class="note-small">Probability = how many papers asked the topic, with recent years counting more. It's a study guide, not a guarantee.</p>
    <ol class="pred-list">${p.topics.map(topicCard).join("")}</ol>
    ${p.unasked && p.unasked.length ? `<div class="card"><h3 class="unasked-h">${ico("star")}Syllabus topics never asked yet</h3>
      <p class="note-small">Not in any paper so far — examiners sometimes add these, so don't skip them.</p>
      <div class="pt-links">${p.unasked.map((u) => `<span class="study-link">${esc(u.topic)}${u.unit ? ` · ${esc(u.unit)}` : ""}</span>`).join("")}</div></div>` : ""}
    <p class="note-small">Predicted ${new Date(p.at).toLocaleString()}.</p>`;
  const rr = $("rerun"); if (rr) rr.onclick = runPrediction;
  $("predDownload").onclick = () => downloadPrediction(p);
}

function downloadPrediction(p) {
  const lines = [`# Exam prediction — ${subjectName()}`, "",
    `Based on ${p.papers.length} past paper(s): ${p.papers.map((x) => `${x.source}${x.year ? ` (${x.year})` : ""}`).join(", ")}.`,
    "Probability = share of papers that asked the topic, with recent years weighted more.", ""];
  p.topics.forEach((t, i) => {
    lines.push(`## ${i + 1}. ${t.topic} — ${t.probability}% (${t.label})`);
    if (t.unit) lines.push(`Unit: ${t.unit}`);
    lines.push(`Asked in ${t.papers_asked} of ${p.papers.length} papers${t.years.length ? ` (${t.years.join(", ")})` : ""}, ${t.times_asked} question(s).`, "");
    if (t.predicted.length) { lines.push("**Likely questions**"); t.predicted.forEach((q, k) => lines.push(`${k + 1}. ${q}`)); lines.push(""); }
    lines.push("**Past questions**"); t.asked.forEach((q) => lines.push(`- (${q.year || "?"}${q.marks ? `, ${q.marks} marks` : ""}) ${q.text}`));
    if (t.links && t.links.length) lines.push("", "**Study from:** " + t.links.map((l) => l.kind === "video" ? `${l.source} at ${fmtTs(l.page)}` : `${l.source}, page ${l.page}`).join("; "));
    lines.push("");
  });
  if (p.unasked && p.unasked.length) { lines.push("## Syllabus topics never asked yet"); p.unasked.forEach((u) => lines.push(`- ${u.topic}${u.unit ? ` (${u.unit})` : ""}`)); }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/markdown" }));
  a.download = `exam-prediction-${subjectName().replace(/[^\w-]+/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

$("predictOut").addEventListener("click", (e) => {
  const v = e.target.closest(".study-link[data-video]");
  if (v) { openPlayer(v.dataset.video, +v.dataset.t, v.dataset.title); return; }
  const ex = e.target.closest(".explain");
  if (ex) {
    showView("home");
    $("question").value = `Explain "${ex.dataset.topic}" for my exam, with the key points and an example.`;
    $("chatForm").requestSubmit();
  }
});
$("predictBtn").addEventListener("click", runPrediction);
$("paperInput").addEventListener("change", (e) => addPapers(e.target.files));
{
  const pd = $("paperDrop");
  ["dragenter", "dragover"].forEach((ev) => pd.addEventListener(ev, (e) => { e.preventDefault(); pd.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => pd.addEventListener(ev, (e) => { e.preventDefault(); pd.classList.remove("drag"); }));
  pd.addEventListener("drop", (e) => addPapers(e.dataTransfer.files));
}

/* ---------- views (Home / My Documents / Settings / About) ---------- */
const VIEWS = ["home", "documents", "viva", "predict", "settings", "about"];
function showView(name) {
  if (!VIEWS.includes(name)) name = "home";
  document.querySelectorAll("[data-view]").forEach((v) => { v.hidden = v.dataset.view !== name; });
  document.querySelectorAll(".nav-btn").forEach((b) => {
    if (b.dataset.viewLink === name) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
  $("composerWrap").hidden = name !== "home";
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  closeMenu();
}
document.addEventListener("click", (e) => {
  const link = e.target.closest("[data-view-link]");
  if (!link) return;
  e.preventDefault();
  showView(link.dataset.viewLink);
  scroller.scrollTop = 0;
});
window.addEventListener("hashchange", () => showView(location.hash.slice(1)));

/* mobile menu */
const sidebar = $("sidebar"), scrim = $("scrim");
function closeMenu() { sidebar.classList.remove("open"); scrim.hidden = true; }
$("menuBtn").addEventListener("click", () => { sidebar.classList.add("open"); scrim.hidden = false; });
scrim.addEventListener("click", closeMenu);

/* settings */
$("clearAll").addEventListener("click", async () => {
  const mine = [...docs.values()].filter((d) => !d.builtIn);
  if (!mine.length || !confirm(`Remove all ${mine.length} document(s) from this browser?`)) return;
  for (const d of mine) { await store.remove(d.source); docs.delete(d.source); }
  renderDocs();
});

$("subjectSel").addEventListener("change", (e) => switchSubject(e.target.value));
$("newSubject").addEventListener("click", newSubject);
$("renameSubject").addEventListener("click", renameSubject);
$("deleteSubject").addEventListener("click", deleteSubject);

showView(location.hash.slice(1) || "home");
loadDocs().then(() => {
  $("modelInfo").innerHTML = `<dt>Answers</dt><dd>${esc(CONFIG.chat_model || "Gemini")} (with automatic fallbacks)</dd>
    <dt>Embeddings</dt><dd>${esc(CONFIG.embedding_model)} · ${CONFIG.embedding_dim} dimensions</dd>
    <dt>Retrieval</dt><dd>Top ${CONFIG.top_k} passages, cosine similarity ≥ ${CONFIG.min_score}</dd>`;
}).catch((e) => {
  $("docList").innerHTML = `<li class="empty error">${ico("alert")}<div><strong>Could not start</strong><span>${esc(e.message)}</span></div></li>`;
});
