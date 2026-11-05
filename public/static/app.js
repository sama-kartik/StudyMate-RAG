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

function addMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `<div class="bubble">${html}</div>`;
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return wrap.querySelector(".bubble");
}

/** POST JSON; on 429 (Gemini rate limit) wait and retry a few times. */
async function api(url, body, onWait) {
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(url, body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok) return data;
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
          const req = indexedDB.open("studymate", 1);
          req.onupgradeneeded = () => req.result.createObjectStore("docs", { keyPath: "source" });
          req.onsuccess = () => resolve(req.result);
          req.onerror = () => resolve(null);
        } catch { resolve(null); }
      });
    }
    return dbp;
  }
  async function tx(mode, fn) {
    const db = await open();
    if (!db) return undefined; // private mode etc.: documents last until page reload
    return new Promise((resolve, reject) => {
      const t = db.transaction("docs", mode);
      const r = fn(t.objectStore("docs"));
      t.oncomplete = () => resolve(r && r.result);
      t.onerror = () => reject(t.error);
    });
  }
  return {
    all: () => tx("readonly", (s) => s.getAll()).then((r) => r || []),
    put: (doc) => tx("readwrite", (s) => s.put(doc)),
    remove: (source) => tx("readwrite", (s) => s.delete(source)),
  };
})();

function toDoc(raw, builtIn) {
  return {
    source: raw.source, pages: raw.pages, builtIn,
    chunks: raw.chunks.map((c) => ({ ...c, embedding: Float32Array.from(c.embedding) })),
  };
}

async function loadDocs() {
  CONFIG = await api("/api/config");
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
    docs.set(d.source, { ...d, builtIn: false });
  }
  if (stale) {
    $("uploadStatus").innerHTML =
      `<div class="err">${stale} document(s) were indexed with an older model and were removed. Please upload them again.</div>`;
  }
  renderDocs();
}

function renderDocs() {
  const list = $("docList"), scope = $("scope");
  const current = scope.value;
  scope.innerHTML = '<option value="">All documents</option>';
  if (!docs.size) {
    list.innerHTML = '<li class="muted">No documents yet</li>';
    return;
  }
  list.innerHTML = "";
  for (const d of [...docs.values()].sort((a, b) => a.source.localeCompare(b.source))) {
    const li = document.createElement("li");
    li.innerHTML = `<div class="name">${esc(d.source)}</div>
      <div class="meta">${d.pages} pages · ${d.chunks.length} chunks${d.builtIn ? " · built-in" : ""}</div>
      <div class="actions"><button class="sum">Summarise</button>${d.builtIn ? "" : '<button class="del">Remove</button>'}</div>`;
    li.querySelector(".sum").onclick = () => summarise(d.source);
    if (!d.builtIn) li.querySelector(".del").onclick = () => removeDoc(d.source);
    list.appendChild(li);
    scope.insertAdjacentHTML("beforeend", `<option value="${esc(d.source)}">${esc(d.source)}</option>`);
  }
  scope.value = docs.has(current) ? current : "";
}

/* ---------- PDF ingestion ---------- */
async function extractPages(file) {
  const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const content = await (await pdf.getPage(i)).getTextContent();
    let text = content.items.map((it) => it.str + (it.hasEOL ? "\n" : "")).join("");
    text = text.replace(/­/g, "").replace(/-\n(\w)/g, "$1")
               .replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
    if (text) pages.push({ page: i, text });
  }
  return { pages, numPages: pdf.numPages };
}

async function ingest(file, line) {
  const source = file.name;
  line.textContent = `${source}: reading PDF…`;
  const { pages, numPages } = await extractPages(file);
  if (!pages.length) throw new Error("no extractable text (the PDF may be scanned images)");

  const chunks = [];
  for (let i = 0; i < pages.length; i += PAGES_PER_REQUEST) {
    line.textContent = `${source}: indexing page ${pages[Math.min(i + PAGES_PER_REQUEST, pages.length) - 1].page} of ${numPages}…`;
    const r = await api("/api/index", { source, pages: pages.slice(i, i + PAGES_PER_REQUEST) },
      (s) => { line.textContent = `${source}: Gemini rate limit, retrying in ${s}s…`; });
    chunks.push(...r.chunks);
  }
  const doc = {
    source, pages: pages.length, model: CONFIG.embedding_model, dim: CONFIG.embedding_dim,
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
  for (const file of files) {
    const line = document.createElement("div");
    status.appendChild(line);
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      line.innerHTML = `<span class="err">✗ ${esc(file.name)}: only PDF files are supported</span>`;
      continue;
    }
    try {
      const d = await ingest(file, line);
      line.textContent = `✓ ${d.source}: ${d.pages} pages, ${d.chunks.length} chunks`;
    } catch (e) {
      line.innerHTML = `<span class="err">✗ ${esc(file.name)}: ${esc(e.message)}</span>`;
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
    if (scope && d.source !== scope) continue;
    for (const c of d.chunks) {
      let s = 0;
      for (let i = 0; i < q.length; i++) s += q[i] * c.embedding[i]; // unit vectors: dot = cosine
      if (s >= CONFIG.min_score) hits.push({ source: d.source, page: c.page, text: c.text, score: s });
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

const pageRange = (a, b) => (a === b ? `Page ${a}` : `Pages ${a}–${b}`);

async function summarise(source) {
  const doc = docs.get(source);
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
    const range = pageRange(pages[0].page, pages[pages.length - 1].page);
    try {
      notes[i] = await apiPatient("/api/summary/section", { source, pages },
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
        { source, total_pages: doc.pages, sections: ok })).overview;
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
  a.download = source.replace(/\.pdf$/i, "") + " - summary.md";
  a.href = URL.createObjectURL(new Blob([full], { type: "text/markdown" }));
  bubble.appendChild(a);
}

/* ---------- chat ---------- */
function renderAnswer(r, retrieval) {
  let html = md(r.answer);
  if (r.sources.length) {
    html += `<details class="sources"><summary>Sources (${r.sources.length})</summary>` +
      r.sources.map((s) => `<div class="src"><b>[${s.n}]</b> ${esc(s.source)}, page ${s.page}
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
    if (!docs.size) throw new Error("Upload a PDF first, then ask about it.");
    const t0 = performance.now();
    const { embedding } = await api("/api/embed-query", { question: q });
    const passages = search(embedding, $("scope").value);
    const retrieval = (performance.now() - t0) / 1000;
    bubble.innerHTML = '<span class="typing">Writing an answer…</span>';
    const r = await api("/api/answer", { question: q, passages });
    bubble.innerHTML = renderAnswer(r, retrieval);
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
$("fileInput").addEventListener("change", (e) => upload([...e.target.files]));
const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => upload([...e.dataTransfer.files]));

loadDocs().catch((e) => {
  $("docList").innerHTML = `<li class="muted error">Could not start: ${esc(e.message)}</li>`;
});
