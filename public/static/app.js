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
const md = (text) => DOMPurify.sanitize(marked.parse(text || ""));
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
  const badge = d.role === "syllabus" ? `<span class="badge">Syllabus</span>` : "";
  li.innerHTML = `<div class="doc-icon${video ? " video" : ""}">${ico(video ? "play" : "file")}</div>
    <div><div class="name">${video ? `<button class="linklike">${esc(d.source)}</button>` : esc(d.source)} ${badge}</div>
      <div class="meta">${meta}</div></div>`;
  if (video) li.querySelector(".linklike").onclick = () => openPlayer(d.videoId, 0, d.source);
  const syl = !video && !d.builtIn
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
async function addVideo(url, line) {
  line.textContent = "Checking the link…";
  const info = await api("/api/youtube/info", { url });
  const source = `${info.title} (YouTube)`;
  // Watch several 5-minute windows in parallel. If YouTube told us the length we know
  // every window up front; otherwise keep starting windows until one reports the end.
  const WINDOW = 300, PARALLEL = 3;
  const limit = info.duration ? info.duration : 3 * 3600;
  const total = info.duration ? Math.ceil(info.duration / WINDOW) : null;
  const segments = [];
  let next = 0, endAt = Infinity, done = 0, busy = false;
  const show = () => {
    line.textContent = `${info.title}: Gemini is watching ${PARALLEL} parts at a time — ` +
      (total ? `${done} of ${total} parts done` : `${done} parts done`) +
      ` (${segments.length} notes)${busy ? " · Gemini is busy, retrying…" : ""}`;
  };
  show();
  async function worker() {
    while (true) {
      const start = next;
      if (start >= endAt || start >= limit) return;
      next += WINDOW;
      const r = await apiPatient("/api/youtube/window", { video_id: info.video_id, start },
        () => { busy = true; show(); });
      busy = false;
      segments.push(...r.segments);
      if (r.end) endAt = Math.min(endAt, start);
      done++; show();
    }
  }
  await Promise.all(Array.from({ length: PARALLEL }, worker));
  segments.sort((a, b) => a.t - b.t);
  if (!segments.length) throw new Error("Gemini couldn't read any content from this video.");
  const pages = segments.map((g) => ({ page: g.t, text: `[${fmtTs(g.t)}] ${g.text}` }));
  const chunks = [];
  for (let i = 0; i < pages.length; i += 20) {
    line.textContent = `${info.title}: indexing notes ${Math.min(i + 20, pages.length)} of ${pages.length}…`;
    const r = await api("/api/index", { source, pages: pages.slice(i, i + 20) },
      (s) => { line.textContent = `${info.title}: Gemini rate limit, retrying in ${s}s…`; });
    chunks.push(...r.chunks);
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

/** Render one PDF page to a JPEG data URL about 1100 px wide. */
async function renderPage(pdf, n) {
  const page = await pdf.getPage(n);
  const base = page.getViewport({ scale: 1 });
  const viewport = page.getViewport({ scale: Math.min(2, 1100 / base.width) });
  const canvas = document.createElement("canvas");
  canvas.width = Math.floor(viewport.width); canvas.height = Math.floor(viewport.height);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  await page.render({ canvasContext: ctx, viewport }).promise;
  let q = 0.72, url = canvas.toDataURL("image/jpeg", q);
  while (url.length > 1_400_000 && q > 0.35) { q -= 0.12; url = canvas.toDataURL("image/jpeg", q); }
  canvas.width = canvas.height = 0;
  return url;
}

async function ingest(file, line) {
  const source = file.name;
  line.textContent = `${source}: reading PDF…`;
  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  const all = await extractPages(pdf);
  const readImages = $("readImages") ? $("readImages").checked : true;

  // Pages Gemini should look at: scans are transcribed; pages with pictures get figure descriptions.
  const jobs = [];
  for (const p of all) {
    if (p.text.length < SCANNED_CHARS) jobs.push({ p, mode: "transcribe" });
    else if (readImages && p.hasImages) jobs.push({ p, mode: "visuals" });
  }
  let seen = 0, transcribed = 0, described = 0;
  if (jobs.length) {
    const batches = [];
    for (let i = 0; i < jobs.length; i += VISION_BATCH) batches.push(jobs.slice(i, i + VISION_BATCH));
    const status = () => { line.textContent = `${source}: Gemini is reading ${jobs.length} page image(s) — ${seen} done…`; };
    status();
    await runLimited(batches.map((batch) => async () => {
      const payload = [];
      for (const j of batch) {
        payload.push({ page: j.p.page, mode: j.mode, text: j.mode === "visuals" ? j.p.text : "",
                       image: await renderPage(pdf, j.p.page) });
      }
      try {
        const r = await apiPatient("/api/vision/pages", { source, pages: payload });
        for (const j of batch) {
          const got = (r.pages.find((x) => x.page === j.p.page) || {}).text || "";
          if (!got) continue;
          if (j.mode === "transcribe") { j.p.text = got; transcribed++; }
          else { j.p.text += "\n\n" + got; described++; }
        }
      } catch (e) {
        console.warn("vision failed for pages", batch.map((j) => j.p.page), e);
      }
      seen += batch.length; status();
    }), VISION_PARALLEL);
  }
  const pages = all.filter((p) => p.text.trim()).map(({ page, text }) => ({ page, text }));
  if (!pages.length) throw new Error("no readable text, even from the page images");

  const chunks = [];
  for (let i = 0; i < pages.length; i += PAGES_PER_REQUEST) {
    line.textContent = `${source}: indexing page ${pages[Math.min(i + PAGES_PER_REQUEST, pages.length) - 1].page} of ${pdf.numPages}…`;
    const r = await api("/api/index", { source, pages: pages.slice(i, i + PAGES_PER_REQUEST) },
      (s) => { line.textContent = `${source}: Gemini rate limit, retrying in ${s}s…`; });
    chunks.push(...r.chunks);
  }
  const doc = {
    source, subject: currentSubject, pages: pages.length, imagePages: transcribed + described, scannedPages: transcribed,
    model: CONFIG.embedding_model, dim: CONFIG.embedding_dim,
    chunks: chunks.map((c) => ({ page: c.page, chunk: c.chunk, text: c.text,
                                 embedding: Float32Array.from(c.embedding) })),
  };
  await store.put(doc);
  docs.set(source, { ...doc, builtIn: false });
  return doc;
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

/* ---------- views (Home / My Documents / Settings / About) ---------- */
const VIEWS = ["home", "documents", "viva", "settings", "about"];
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
