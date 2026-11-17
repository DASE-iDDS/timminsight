/* TiMMInsight demo client: a live workspace over the real system. d3 v7 is vendored under /static/vendor (no CDN). Chart colours
   come from the validated palette: series 1-3 (blue, orange, aqua). */
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = {s1: css("--series-1"), s2: css("--series-2"), s3: css("--series-3"), text: css("--text"), text2: css("--text-2"),
  text3: css("--text-3"), line: css("--line"), line2: css("--line-2"), surface: css("--chart-surface"), mention: css("--mention"), accent: css("--accent")};
const MODALITY = {table: C.s1, text: C.s2, image: C.s3};
const fmt = (v, d = 3) => (typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toLocaleString("en-US", {maximumFractionDigits: 1}) : +v.toFixed(d)) : v);
const human = (s) => String(s).replaceAll("_", " ");

async function api(path, opts = {}) {
  const r = await fetch(path, {headers: {"Content-Type": "application/json"}, ...opts});
  if (!r.ok) {
    let m = r.statusText;
    try { const d = (await r.json()).detail; m = Array.isArray(d) ? d.map((e) => `${(e.loc || []).join(".")}: ${e.msg}`).join("; ") : d || m; } catch (_) {}
    throw new Error(m);
  }
  return r.json();
}
const S = {config: null, dataset: null, session: null, graph: null, graphDrawn: false};

/* ---------------- tooltip (labels are untrusted: textContent only) ---------------- */
const tip = $("#tooltip");
function showTip(rows, ev) {
  tip.replaceChildren(...rows.map(([v, l]) => el("div", {}, v !== undefined && v !== null ? el("span", {class: "v"}, String(v)) : null, l ? el("span", {class: "l"}, (v !== undefined && v !== null ? "  " : "") + l) : null)));
  tip.hidden = false;
  const x = Math.min(ev.clientX + 16, innerWidth - tip.offsetWidth - 8), y = Math.min(ev.clientY + 16, innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };

/* ---------------- tabs, mode, key ---------------- */
function setTab(name) {
  for (const b of document.querySelectorAll(".tabs button")) { const on = b.dataset.tab === name; b.classList.toggle("on", on); b.setAttribute("aria-selected", on); }
  for (const t of ["chat", "graph", "data"]) $("#tab-" + t).hidden = t !== name;
  if (name === "graph" && S.graph && !S.graphDrawn) drawGraph();
}
for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => setTab(b.dataset.tab);

function clearViews() {
  $("#chat").replaceChildren(el("div", {class: "empty"}, el("b", {}, "Ask a multimodal question."),
    el("span", {}, "Choose a dataset or import your own, build its metadata graph, then ask. TiQC clarifies the question against the graph; TiTSP plans and executes it.")));
  $("#evidence").replaceChildren(el("p", {class: "sub pad"}, "The query context Φ, the logical plan with propagated confidences, the Pareto front of physical plans, and the execution trace appear here once a question is planned."));
  $("#graph").replaceChildren(); $("#graph-stats").replaceChildren(); $("#graph-legend").replaceChildren(); $("#graph-table").replaceChildren();
  $("#data-view").replaceChildren(el("p", {class: "sub"}, "No dataset open.")); $("#table-list").replaceChildren(el("p", {class: "sub"}, "No dataset open."));
  S.graph = null; S.graphDrawn = false;
}
function chatStart() { const e = $("#chat .empty"); if (e) e.remove(); }

$("#key-btn").onclick = $("#key-banner-btn").onclick = () => { $("#key-input").value = ""; $("#key-dialog").showModal(); };
$("#key-dialog").addEventListener("close", async () => {
  if ($("#key-dialog").returnValue !== "ok") return;
  try { await api("/api/key", {method: "POST", body: JSON.stringify({key: $("#key-input").value})}); await loadConfig(); } catch (e) { alert(e.message); }
});
async function loadConfig() {
  S.config = await api("/api/config");
  $("#model-chip").textContent = `${S.config.backbone} · ${S.config.vision}`;
  $("#key-btn").textContent = S.config.has_key ? "Model key set" : "Set model key";
  $("#key-banner").hidden = S.config.has_key;
}

/* ---------------- data browser ---------------- */
function cellNode(v) {
  if (v && typeof v === "object" && "image" in v) return v.image ? el("img", {src: v.image, alt: ""}) : "(image)";
  return v === null || v === undefined ? "" : String(v);
}
function table(cols, rows, render = (r, c) => cellNode(r[c]), rowClass = () => "") {
  return el("div", {class: "tblwrap"}, el("table", {class: "t"},
    el("thead", {}, el("tr", {}, cols.map((c) => el("th", {}, c)))),
    el("tbody", {}, rows.map((r) => el("tr", {class: rowClass(r)}, cols.map((c) => el("td", {title: typeof r[c] === "string" ? r[c] : null}, render(r, c))))))));
}
function renderTables(preview) {
  const list = $("#table-list"), view = $("#data-view"); list.replaceChildren(); view.replaceChildren();
  for (const t of preview.tables) {
    list.append(el("button", {class: "item", onclick: () => { setTab("data"); document.getElementById(`tbl-${t.name}`)?.scrollIntoView({behavior: "smooth"}); }},
      el("span", {class: "t"}, t.name), el("span", {class: "m"}, `${fmt(t.rows)} rows · ${t.columns.length} columns`)));
    const roles = [...t.text_columns.map((c) => el("span", {class: "chip"}, `text · ${c}`)), ...t.image_columns.map((c) => el("span", {class: "chip"}, `image · ${c}`))];
    view.append(el("div", {class: "card", id: `tbl-${t.name}`},
      el("div", {class: "card-h"}, t.name, el("span", {class: "chip out"}, `${fmt(t.rows)} rows`), ...roles), table(t.columns, t.sample)));
  }
}

/* ---------------- metadata graph: d3-force, drag/pin, zoom, highlight, search ---------------- */
function renderGraph(g) { S.graph = g; S.graphDrawn = false; if (!$("#tab-graph").hidden) drawGraph(); renderGraphTable(g); renderGraphMeta(g); }

function legendSwatch(kind, color) {
  const s = d3.create("svg").attr("width", 18).attr("height", 14);
  if (kind === "source") s.append("rect").attr("x", 2).attr("y", 1).attr("width", 12).attr("height", 12).attr("rx", 3).attr("fill", color);
  else if (kind === "cross") s.append("circle").attr("cx", 8).attr("cy", 7).attr("r", 5).attr("fill", C.surface).attr("stroke", C.text).attr("stroke-width", 2);
  else if (kind === "entity") s.append("circle").attr("cx", 8).attr("cy", 7).attr("r", 5).attr("fill", color);
  else s.append("line").attr("x1", 0).attr("x2", 18).attr("y1", 7).attr("y2", 7).attr("stroke", color).attr("stroke-width", kind === "rel" ? 3 : 1.5).attr("stroke-dasharray", kind === "fk" ? "4 3" : null);
  return s.node();
}
function renderGraphMeta(g) {
  $("#graph-stats").replaceChildren(
    el("span", {class: "chip"}, `${g.stats.sources} sources`), el("span", {class: "chip"}, `${fmt(g.stats.entities)} entities`),
    el("span", {class: "chip accent"}, `${fmt(g.stats.cross_modal_entities)} cross-modal`), el("span", {class: "chip"}, `${fmt(g.stats.relations)} relations`),
    el("span", {class: "sub"}, `top ${g.nodes.filter((n) => n.kind === "entity").length} entities by mentions`));
  $("#graph-legend").replaceChildren(
    ...["table", "text", "image"].map((m) => el("span", {class: "k"}, legendSwatch("source", MODALITY[m]), `${m} source`)),
    ...["table", "text", "image"].map((m) => el("span", {class: "k"}, legendSwatch("entity", MODALITY[m]), `entity in ${m}`)),
    el("span", {class: "k"}, legendSwatch("cross"), "cross-modal entity (ring)"),
    el("span", {class: "k"}, legendSwatch("mention", C.mention), "mention"),
    el("span", {class: "k"}, legendSwatch("rel", C.accent), "relationship (width ∝ J_w·conf)"),
    el("span", {class: "k"}, legendSwatch("fk", C.text2), "foreign key"));
  $("#graph-names").replaceChildren(...g.nodes.map((n) => el("option", {value: n.label})));
}
function renderGraphTable(g) {
  const rows = g.nodes.filter((n) => n.kind === "entity").sort((a, b) => b.mentions - a.mentions)
    .map((n) => ({entity: n.label, type: human(n.type), modalities: n.modalities.join(", "), mentions: n.mentions, "cross-modal": n.cross_modal ? "yes" : ""}));
  const srcs = g.nodes.filter((n) => n.kind === "source").map((n) => ({source: n.label, modality: n.modality, records: n.records ?? ""}));
  $("#graph-table").replaceChildren(el("div", {class: "data-view"},
    el("div", {class: "card"}, el("div", {class: "card-h"}, "Sources"), table(["source", "modality", "records"], srcs)),
    el("div", {class: "card"}, el("div", {class: "card-h"}, "Entities"), table(["entity", "type", "modalities", "mentions", "cross-modal"], rows))));
}
$("#gview-graph").onclick = () => { $("#gview-graph").classList.add("on"); $("#gview-table").classList.remove("on"); $("#graph").hidden = false; $("#graph-table").hidden = true; if (S.graph && !S.graphDrawn) drawGraph(); };
$("#gview-table").onclick = () => { $("#gview-table").classList.add("on"); $("#gview-graph").classList.remove("on"); $("#graph").hidden = true; $("#graph-table").hidden = false; };

function nodeColor(n) { return n.kind === "source" ? MODALITY[n.modality] || C.text3 : MODALITY[n.modalities[0]] || C.text3; }
function nodeRadius(n) { return n.kind === "source" ? 11 : 4 + Math.min(7, Math.sqrt(n.mentions || 1) * 1.2); }

function drawGraph() {
  const g = S.graph; S.graphDrawn = true;
  const box = $("#graph"); box.replaceChildren();
  const W = box.clientWidth || 900, H = box.clientHeight || 560;
  const nodes = g.nodes.map((n) => ({...n}));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const links = g.links.filter((l) => byId.has(l.source) && byId.has(l.target)).map((l) => ({...l}));
  const degree = new Map(nodes.map((n) => [n.id, 0]));
  const nbrs = new Map(nodes.map((n) => [n.id, new Set()]));
  for (const l of links) { degree.set(l.source, degree.get(l.source) + 1); degree.set(l.target, degree.get(l.target) + 1); nbrs.get(l.source).add(l.target); nbrs.get(l.target).add(l.source); }
  const hubs = new Set(nodes.filter((n) => n.kind === "source").map((n) => n.id).concat(
    nodes.filter((n) => n.kind === "entity").sort((a, b) => degree.get(b.id) - degree.get(a.id)).slice(0, 10).map((n) => n.id)));

  const svg = d3.select(box).append("svg").attr("viewBox", [0, 0, W, H]).attr("role", "img").attr("aria-label", "Metadata graph");
  const layer = svg.append("g");
  const width = (l) => (l.kind === "relation" ? 0.8 + 4 * (l.weight || 0) : l.kind === "foreign_key" ? 1.5 : 0.8);
  const link = layer.append("g").selectAll("line").data(links).join("line")
    .attr("stroke", (l) => (l.kind === "relation" ? C.accent : l.kind === "foreign_key" ? C.text2 : C.mention))
    .attr("stroke-opacity", (l) => (l.kind === "relation" ? 0.5 : 0.9)).attr("stroke-width", width)
    .attr("stroke-dasharray", (l) => (l.kind === "foreign_key" ? "4 3" : null));
  link.filter((l) => l.kind !== "mention").on("mousemove", (ev, l) => showTip([[l.kind === "relation" ? human(l.type) : `FK ${l.type}`, l.kind === "relation" ? "relationship" : ""],
    [byId.get(l.source.id || l.source).label + " → " + byId.get(l.target.id || l.target).label], l.weight !== undefined ? [fmt(l.weight, 3), "weight J_w·conf"] : [null]], ev)).on("mouseleave", hideTip);

  const node = layer.append("g").selectAll("g").data(nodes).join("g").attr("class", "node").attr("tabindex", 0);
  node.append("circle").attr("r", (n) => nodeRadius(n) + 8).attr("fill", "transparent");               // bigger hit target
  node.filter((n) => n.kind === "source").append("rect").attr("x", -11).attr("y", -11).attr("width", 22).attr("height", 22).attr("rx", 5)
    .attr("fill", nodeColor).attr("stroke", C.surface).attr("stroke-width", 2);
  node.filter((n) => n.kind === "entity").append("circle").attr("r", nodeRadius)
    .attr("fill", (n) => (n.cross_modal ? C.surface : nodeColor(n))).attr("stroke", (n) => (n.cross_modal ? C.text : C.surface)).attr("stroke-width", 2);
  node.filter((n) => n.kind === "entity" && n.cross_modal).append("circle").attr("r", (n) => nodeRadius(n) - 3).attr("fill", nodeColor);
  const label = layer.append("g").selectAll("text").data(nodes).join("text")
    .attr("class", (n) => "label" + (n.kind === "source" ? " src" : "")).attr("dy", 4)
    .text((n) => (n.label.length > 30 ? n.label.slice(0, 29) + "…" : n.label));

  const tipRows = (n) => n.kind === "source"
    ? [[n.label, ""], [n.modality, "source"], n.records != null ? [fmt(n.records), "records"] : [null], [degree.get(n.id), "linked entities shown"]]
    : [[n.label, ""], [human(n.type), "entity type"], [n.mentions, "mentions"], [n.modalities.join(", "), "modalities"], n.cross_modal ? ["cross-modal", ""] : [null], [degree.get(n.id), "links shown"]];
  node.on("mousemove", (ev, n) => { showTip(tipRows(n), ev); hover = n.id; placeLabels(); }).on("mouseleave", () => { hideTip(); hover = null; placeLabels(); });

  let focus = null, hover = null, zoomK = 1;
  function highlight(id) {
    focus = id;
    const keep = id ? new Set([id, ...nbrs.get(id)]) : null;
    node.attr("opacity", (n) => (!keep || keep.has(n.id) ? 1 : 0.12));
    link.attr("opacity", (l) => (!keep || ((l.source.id || l.source) === id || (l.target.id || l.target) === id) ? 1 : 0.06));
    placeLabels();
  }
  node.on("click", (ev, n) => { ev.stopPropagation(); highlight(focus === n.id ? null : n.id); });
  svg.on("click", () => highlight(null));

  // labels: hubs, the hovered node, the focus neighbourhood, and on zoom everything that fits without collision
  function placeLabels() {
    const t = d3.zoomTransform(svg.node());
    const want = nodes.filter((n) => hubs.has(n.id) || n.id === hover || (focus && (n.id === focus || nbrs.get(focus).has(n.id))) || zoomK >= 1.8)
      .sort((a, b) => (b.id === hover) - (a.id === hover) || (b.kind === "source") - (a.kind === "source") || degree.get(b.id) - degree.get(a.id));
    const boxes = [], shown = new Set(), side = new Map();
    const hit = (x, y, w, h) => boxes.some((b) => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y);
    for (const n of want) {
      const r = nodeRadius(n), w = Math.min(30, n.label.length) * (n.kind === "source" ? 7.2 : 6.4) / zoomK, h = 14 / zoomK;
      let x = n.x + r + 4 / zoomK; const y = n.y - h / 2;
      if (n.kind === "source" && hit(x, y, w, h)) { x = n.x - r - 4 / zoomK - w; side.set(n.id, "left"); }   // sources: try the left side
      if (focus && !(n.id === focus || nbrs.get(focus).has(n.id)) && n.id !== hover) continue;
      // sources are always labelled; others only where they do not collide
      if (n.kind !== "source" && n.id !== hover && boxes.some((b) => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y)) continue;
      boxes.push({x, y, w, h}); shown.add(n.id);
    }
    S.labelSide = side;
    label.attr("display", (n) => (shown.has(n.id) ? null : "none"))
      .attr("text-anchor", (n) => (side.get(n.id) === "left" ? "end" : "start"))
      .attr("x", (n) => (side.get(n.id) === "left" ? n.x - nodeRadius(n) - 4 / zoomK : n.x + nodeRadius(n) + 4 / zoomK)).attr("y", (n) => n.y)
      .attr("font-size", (n) => (n.kind === "source" ? 12 : 11) / Math.max(1, zoomK) * Math.min(zoomK, 1.6));
    void t;
  }

  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((n) => n.id).distance((l) => (l.kind === "mention" ? 70 : l.kind === "foreign_key" ? 180 : 45))
      .strength((l) => (l.kind === "mention" ? 0.25 : l.kind === "foreign_key" ? 0.15 : 0.5)))
    .force("charge", d3.forceManyBody().strength((n) => (n.kind === "source" ? -420 : -60)))
    .force("collide", d3.forceCollide().radius((n) => nodeRadius(n) + 3))
    .force("x", d3.forceX(W / 2).strength(0.04)).force("y", d3.forceY(H / 2).strength(0.06));
  function ticked() {
    link.attr("x1", (l) => l.source.x).attr("y1", (l) => l.source.y).attr("x2", (l) => l.target.x).attr("y2", (l) => l.target.y);
    node.attr("transform", (n) => `translate(${n.x},${n.y})`);
    label.attr("x", (n) => ((S.labelSide || new Map()).get(n.id) === "left" ? n.x - nodeRadius(n) - 4 / zoomK : n.x + nodeRadius(n) + 4 / zoomK)).attr("y", (n) => n.y);
  }
  sim.on("tick", ticked).on("end", () => { placeLabels(); if (!S.fitted) { fit(); S.fitted = true; } });
  sim.tick(120); ticked(); placeLabels();

  node.call(d3.drag()
    .on("start", (ev, n) => { if (!ev.active) sim.alphaTarget(0.25).restart(); n.fx = n.x; n.fy = n.y; })
    .on("drag", (ev, n) => { n.fx = ev.x; n.fy = ev.y; })
    .on("end", (ev) => { if (!ev.active) sim.alphaTarget(0); }));                   // stays pinned (fx/fy kept)
  node.on("dblclick", (ev, n) => { ev.stopPropagation(); n.fx = n.fy = null; sim.alpha(0.3).restart(); });

  const zoom = d3.zoom().scaleExtent([0.25, 6]).on("zoom", (ev) => { layer.attr("transform", ev.transform); zoomK = ev.transform.k; placeLabels(); });
  svg.call(zoom).on("dblclick.zoom", null);
  function fit() {
    const xs = nodes.map((n) => n.x), ys = nodes.map((n) => n.y);
    const x0 = Math.min(...xs) - 40, x1 = Math.max(...xs) + 120, y0 = Math.min(...ys) - 30, y1 = Math.max(...ys) + 30;
    const k = Math.min(4, 0.95 * Math.min(W / (x1 - x0), H / (y1 - y0)));
    svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2));
  }
  S.fitted = false; fit();
  $("#graph-fit").onclick = fit;
  $("#graph-search").onchange = () => {
    const q = $("#graph-search").value.trim().toLowerCase(); if (!q) return highlight(null);
    const n = nodes.find((x) => x.label.toLowerCase() === q) || nodes.find((x) => x.label.toLowerCase().includes(q));
    if (!n) return;
    highlight(n.id);
    svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity.translate(W / 2, H / 2).scale(Math.max(zoomK, 2)).translate(-n.x, -n.y));
  };
}

/* ---------------- evidence ---------------- */
function sec(title, note, ...body) { return el("details", {class: "sec", open: true}, el("summary", {}, title, note ? el("span", {class: "sub"}, note) : null), el("div", {class: "sec-body"}, ...body)); }
function bindingRow(op, b) {
  const parts = Object.entries(b).map(([k, v]) => {
    let val;
    if (k === "conditions" && Array.isArray(v)) val = v.map((c) => `${c.column} ${c.op} ${JSON.stringify(c.value)}`).join(" ∧ ");
    else if (k === "condition" && v && typeof v === "object") val = `${v.op} ${JSON.stringify(v.value)}`;
    else val = Array.isArray(v) ? v.join(", ") : typeof v === "object" ? JSON.stringify(v) : String(v);
    return el("div", {}, el("span", {class: "kk"}, `${k}: `), el("code", {}, val));
  });
  return el("tr", {}, el("td", {class: "op"}, op), el("td", {class: "kvs"}, parts));
}
function renderPlanTree(rootData) {
  const root = d3.hierarchy(rootData, (d) => d.children);
  const NW = 168, NH = 64, GX = 16, GY = 32;
  d3.tree().nodeSize([NW + GX, NH + GY])(root);
  let x0 = Infinity, x1 = -Infinity; root.each((d) => { x0 = Math.min(x0, d.x); x1 = Math.max(x1, d.x); });
  const W = x1 - x0 + NW + 8, H = (root.height + 1) * (NH + GY) - GY + 8;
  const svg = d3.create("svg").attr("viewBox", [x0 - NW / 2 - 4, -4, W, H]).attr("width", W).attr("height", H).attr("role", "img").attr("aria-label", "Logical plan, parent above child");
  svg.append("g").selectAll("path").data(root.links()).join("path").attr("fill", "none").attr("stroke", "#B6BECC").attr("stroke-width", 1.5)
    .attr("d", d3.linkVertical().source((d) => [d.source.x, d.source.y + NH]).target((d) => [d.target.x, d.target.y]));
  const g = svg.append("g").selectAll("g").data(root.descendants()).join("g").attr("transform", (d) => `translate(${d.x - NW / 2},${d.y})`).attr("tabindex", 0);
  g.append("rect").attr("width", NW).attr("height", NH).attr("rx", 8)
    .attr("fill", (d) => (d.data.cross_modal ? css("--accent-tint") : "#FFFFFF")).attr("stroke", (d) => (d.data.cross_modal ? C.accent : C.line)).attr("stroke-width", 1.2);
  g.append("text").attr("x", 10).attr("y", 18).attr("font-size", 12).attr("font-weight", 600).attr("fill", C.text)
    .text((d) => {   // the title shares the top line with the row count: cap it by the space left
      const s = d.data.op === "SCAN" && d.data.source ? `SCAN ${d.data.source}` : d.data.op;
      const rowsPx = d.data.rows != null ? (String(fmt(d.data.rows)).length + 5) * 6.2 + 12 : 0;
      const room = Math.floor((NW - 20 - rowsPx) / 7.4);   // ~7.4 px per 12-px bold character
      return s.length > room ? s.slice(0, room - 1) + "…" : s; });
  g.append("text").attr("x", 10).attr("y", 34).attr("font-size", 11).attr("fill", C.text2).attr("font-family", "IBM Plex Mono, Menlo, monospace")
    .text((d) => { const s = d.data.impl || "–"; return s.length > 24 ? s.slice(0, 23) + "…" : s; });
  const BAR = 72;
  g.append("rect").attr("x", 10).attr("y", 44).attr("width", BAR).attr("height", 6).attr("rx", 3).attr("fill", C.line2);
  g.append("rect").attr("x", 10).attr("y", 44).attr("width", (d) => BAR * (d.data.confidence || 0)).attr("height", 6).attr("rx", 3).attr("fill", C.s1);
  g.append("text").attr("x", 10 + BAR + 6).attr("y", 50).attr("font-size", 11).attr("fill", C.text).text((d) => (d.data.confidence != null ? `q ${d.data.confidence.toFixed(3)}` : "q –"));
  g.append("text").attr("x", NW - 10).attr("y", 18).attr("text-anchor", "end").attr("font-size", 11).attr("fill", C.text3).text((d) => (d.data.rows != null ? `${fmt(d.data.rows)} rows` : ""));
  g.on("mousemove", (ev, d) => showTip([[d.data.op, d.data.source ? `on ${d.data.source}` : ""], [d.data.impl || "–", "implementation"],
    [d.data.confidence, "propagated confidence q"], d.data.n ? [fmt(d.data.n), "source size n"] : [null], d.data.rows != null ? [fmt(d.data.rows), "output rows"] : [null],
    d.data.cross_modal ? ["cross-modal operator", ""] : [null]], ev)).on("mouseleave", hideTip);
  return el("div", {class: "plan"}, svg.node());
}
function renderPareto(points) {
  const W = 408, H = 240, M = {l: 68, r: 16, t: 16, b: 44};
  const x = d3.scaleLinear().domain(d3.extent(points, (p) => p.latency)).nice().range([M.l, W - M.r]);
  const y = d3.scaleLinear().domain(d3.extent(points, (p) => p.confidence)).nice().range([H - M.b, M.t]);
  if (x.domain()[0] === x.domain()[1]) x.domain([x.domain()[0] - 1, x.domain()[1] + 1]);
  if (y.domain()[0] === y.domain()[1]) y.domain([y.domain()[0] - 0.01, y.domain()[1] + 0.01]);
  const svg = d3.create("svg").attr("viewBox", [0, 0, W, H]).attr("width", W).attr("height", H).attr("role", "img").attr("aria-label", "Pareto front, estimated latency against confidence");
  svg.append("g").selectAll("line").data(y.ticks(4)).join("line").attr("x1", M.l).attr("x2", W - M.r).attr("y1", y).attr("y2", y).attr("stroke", C.line2);
  svg.append("g").selectAll("text").data(y.ticks(4)).join("text").attr("x", M.l - 8).attr("y", (d) => y(d) + 4).attr("text-anchor", "end").attr("font-size", 11).attr("fill", C.text3).text((d) => d.toFixed(3));
  svg.append("line").attr("x1", M.l).attr("x2", W - M.r).attr("y1", H - M.b).attr("y2", H - M.b).attr("stroke", C.line);
  svg.append("g").selectAll("text").data(x.ticks(5)).join("text").attr("x", x).attr("y", H - M.b + 16).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", C.text3).text((d) => fmt(d, 1));
  svg.append("text").attr("x", (M.l + W - M.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", C.text2).text("estimated latency (lower is better) →");
  svg.append("text").attr("transform", `translate(12,${(M.t + H - M.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", C.text2).text("confidence (higher is better) →");
  const sorted = [...points].sort((a, b) => a.selected - b.selected);
  const g = svg.append("g").selectAll("g").data(sorted).join("g").attr("transform", (p) => `translate(${x(p.latency)},${y(p.confidence)})`).attr("tabindex", 0);
  g.append("circle").attr("r", 12).attr("fill", "transparent");
  g.append("circle").attr("r", (p) => (p.selected ? 6 : 4.5)).attr("fill", (p) => (p.selected ? C.s2 : C.s1)).attr("stroke", C.surface).attr("stroke-width", 2);
  g.on("mousemove", (ev, p) => showTip([[p.selected ? "selected by the weighted Tchebycheff objective" : "Pareto-optimal assignment", ""],
    [fmt(p.latency), "latency"], [fmt(p.resource), "resource"], [p.confidence, "confidence"], [Object.values(p.assignment).join(", "), ""]], ev)).on("mouseleave", hideTip);
  const s = points.find((p) => p.selected);
  if (s) svg.append("text").attr("x", x(s.latency) + (x(s.latency) > W - 110 ? -10 : 10)).attr("text-anchor", x(s.latency) > W - 110 ? "end" : "start").attr("y", y(s.confidence) - 10).attr("font-size", 11).attr("font-weight", 600).attr("fill", C.text).text("selected");
  return el("div", {class: "chart"}, svg.node(), el("div", {class: "legend", style: "border:0;padding:4px 0;background:none"},
    el("span", {class: "k"}, legendDot(C.s1), "Pareto-optimal assignment"), el("span", {class: "k"}, legendDot(C.s2), "selected")));
}
function legendDot(c) { const s = d3.create("svg").attr("width", 12).attr("height", 12); s.append("circle").attr("cx", 6).attr("cy", 6).attr("r", 4.5).attr("fill", c); return s.node(); }
function renderAnswerChart(run) {
  const rows = run.result.sample, spec = run.chart || {};
  if (!rows.length) return el("p", {class: "sub"}, "The query returned no rows.");
  const yk = spec.y && spec.y in rows[0] ? spec.y : Object.keys(rows[0]).find((k) => typeof rows[0][k] === "number");
  const xk = spec.x && spec.x in rows[0] ? spec.x : Object.keys(rows[0]).find((k) => k !== yk);
  const tbl = table(Object.keys(rows[0]), rows.slice(0, 20));
  if (!yk || !xk || spec.chart === "table" || rows.length < 2) return tbl;          // a one-bar chart is an anti-pattern
  const data = rows.slice(0, 12).map((r) => ({k: String(r[xk] ?? ""), v: +r[yk] || 0}));
  const W = 408, L = 136, rowH = 24, H = data.length * rowH + 8, xs = d3.scaleLinear().domain([0, d3.max(data, (d) => d.v) || 1]).range([0, W - L - 56]);
  const svg = d3.create("svg").attr("viewBox", [0, 0, W, H]).attr("width", W).attr("height", H).attr("role", "img").attr("aria-label", `${yk} by ${xk}`);
  const g = svg.append("g").selectAll("g").data(data).join("g").attr("transform", (d, i) => `translate(0,${i * rowH + 4})`);
  g.append("text").attr("x", L - 8).attr("y", 13).attr("text-anchor", "end").attr("font-size", 11).attr("fill", C.text2).text((d) => (d.k.length > 20 ? d.k.slice(0, 19) + "…" : d.k));
  g.append("rect").attr("x", L).attr("y", 4).attr("height", rowH - 10).attr("width", (d) => Math.max(1, xs(d.v))).attr("rx", 3).attr("fill", C.s1)
    .on("mousemove", (ev, d) => showTip([[fmt(d.v), yk], [d.k, xk]], ev)).on("mouseleave", hideTip);
  return el("div", {class: "chart"}, el("div", {class: "cap"}, `${human(yk)} by ${human(xk)}`), svg.node(), tbl);
}
function renderEvidence(run) {
  const ev = $("#evidence"); ev.replaceChildren();
  const phi = run.phi;
  ev.append(sec("Query context Φ", "compiled by the analyzer",
    el("dl", {class: "kv"},
      el("dt", {}, "Required Σreq"), el("dd", {}, el("div", {class: "ops"}, phi.required.map((o) => el("span", {class: "chip accent"}, o)))),
      el("dt", {}, "Optional Σopt"), el("dd", {}, phi.optional.length ? el("div", {class: "ops"}, phi.optional.map((o) => el("span", {class: "chip"}, o))) : "–"),
      el("dt", {}, "Root ρ"), el("dd", {}, phi.root),
      el("dt", {}, "Modalities M"), el("dd", {}, phi.modalities.join(", ")),
      el("dt", {}, "Sources"), el("dd", {}, phi.source_sizes.map(([s, n]) => `${s} (${fmt(n)})`).join(", "))),
    el("div", {}, el("div", {class: "cap", style: "margin-bottom:4px"}, "Bindings 𝓑"),
      el("table", {class: "bind"}, el("tbody", {}, Object.entries(phi.bindings).map(([op, b]) => bindingRow(op, b)))),
      el("details", {class: "raw", style: "margin-top:8px"}, el("summary", {}, "Raw JSON"), el("pre", {class: "code"}, JSON.stringify(phi.bindings, null, 2))))));
  ev.append(sec("Logical plan", "Stage 1 · MCTS over the OCG", renderPlanTree(run.plan),
    el("div", {class: "cap"}, "Parent above child. Shaded: cross-modal operators. Bar: propagated confidence q."),
    el("div", {}, el("div", {class: "cap", style: "margin-bottom:4px"}, "Cached candidates ranked by Q"),
      table(["plan", "Q", "cost", "conf"], run.candidates.map((c) => ({plan: c.plan, Q: c.q, cost: c.cost, conf: c.confidence, sel: c.selected})),
        (r, c) => (c === "plan" ? el("span", {class: "mono wrapcell"}, r.plan) : String(r[c])), (r) => (r.sel ? "sel" : "")))));
  const assign = Object.entries(run.selected.assignment).map(([nid, impl]) => ({operator: (run.trace.find((t) => t.id === nid) || {}).op || nid, implementation: impl}));
  ev.append(sec("Physical plan", "Stage 2 · Pareto front + Tchebycheff", renderPareto(run.pareto),
    el("dl", {class: "kv"}, el("dt", {}, "Selected"), el("dd", {}, `latency ${fmt(run.selected.latency)} · resource ${fmt(run.selected.resource)} · confidence ${run.selected.confidence}`)),
    table(["operator", "implementation"], assign, (r, c) => (c === "implementation" ? el("span", {class: "mono"}, r[c]) : r[c]))));
  ev.append(sec("Execution trace", `${run.trace.length} operators`, table(["operator", "implementation", "rows"],
    run.trace.map((t) => ({operator: t.op, implementation: t.impl, rows: fmt(t.rows)})), (r, c) => (c === "implementation" ? el("span", {class: "mono"}, r[c]) : r[c]))));
  ev.append(sec("Answer", `${fmt(run.result.rows)} result rows`, el("p", {class: "answer", style: "margin:0"}, run.answer), renderAnswerChart(run),
    el("div", {class: "cap"}, Object.entries(run.timing || {}).map(([k, v]) => `${k.replace("_ms", "")} ${v} ms`).join(" · "))));
}

/* ---------------- conversation ---------------- */
function msg(who, ...body) {
  chatStart();
  const m = el("div", {class: "msg" + (who === "You" ? " user" : who === "Error" ? " error" : "")}, el("div", {class: "who"}, who), ...body);
  $("#chat").append(m); m.scrollIntoView({behavior: "smooth", block: "end"}); return m;
}
function roundCard(r, onChoose) {
  const detected = (r.detected || []).map((a) => `${human(a.ambiguity_type)} ${a.severity}`).join(" · ");
  const card = msg("TiQC",
    el("div", {class: "row wrap"}, el("span", {class: "chip warn"}, `Round ${r.round}`), el("span", {class: "chip"}, human(r.ambiguity_type)), el("span", {class: "chip out"}, `severity ${r.severity}`)),
    el("div", {class: "q", style: "margin-top:8px"}, r.question),
    r.description ? el("div", {class: "desc"}, r.description) : null,
    detected ? el("div", {class: "meta"}, `Detected (≥ θamb): ${detected}`) : null);
  card.append(el("div", {class: "opts"}, r.options.map((o) => el("button", {class: "opt" + (r.answer === o ? " chosen" : ""), disabled: !onChoose, onclick: () => onChoose && onChoose(o)}, o))));
  if (onChoose) {
    const free = el("input", {class: "inp grow", placeholder: "Or describe what you mean", "aria-label": "Free-form answer"});
    card.append(el("form", {class: "row", style: "margin-top:8px", onsubmit: (e) => { e.preventDefault(); if (free.value.trim()) onChoose(free.value.trim()); }}, free, el("button", {class: "btn ghost"}, "Answer")));
  }
  return card;
}

/* ---------------- live: datasets, import, roles, graph ---------------- */
async function renderDatasets() {
  await loadConfig();
  const list = $("#dataset-list"); list.replaceChildren();
  for (const d of S.config.datasets) list.append(el("button", {class: "item" + (S.dataset === d ? " on" : ""), onclick: () => selectDataset(d)},
    el("span", {class: "t"}, d), el("span", {class: "m"}, d.startsWith("upload/") ? "your dataset" : "benchmark")));
}
async function selectDataset(d) {
  S.dataset = d; S.session = null; clearViews();
  $("#crumb-dataset").textContent = d; renderDatasets();
  try {
    if (d.startsWith("upload/")) renderRoleEditor(await api(`/api/uploads/${d.split("/")[1]}`));
    renderTables(await api(`/api/datasets/${d}/preview`));
    if (!d.startsWith("upload/")) setTab("data");
  } catch (e) { msg("Error", e.message); }
}
function renderRoleEditor(info) {
  const name = info.dataset.split("/")[1], boxes = [];
  const card = msg("Import", el("div", {class: "q"}, `Column roles of ${info.dataset}`),
    el("div", {class: "desc"}, "Detected with the same rules as the benchmark datasets. Correct them, save, then build the metadata graph."));
  for (const t of info.tables) {
    const rows = t.all_columns.map((c) => {
      const tx = el("input", {type: "checkbox", checked: t.text_columns.includes(c), "aria-label": `${t.name}.${c} is text`});
      const im = el("input", {type: "checkbox", checked: t.image_columns.includes(c), disabled: !t.image_columns.includes(c), "aria-label": `${t.name}.${c} holds images`});
      boxes.push({t: t.name, c, tx, im});
      return el("tr", {}, el("td", {class: "k"}, c), el("td", {}, tx), el("td", {}, im));
    });
    card.append(el("div", {class: "card", style: "margin-top:8px"}, el("div", {class: "card-h"}, t.name, el("span", {class: "chip out"}, `${fmt(t.rows)} rows`)),
      el("table", {class: "t"}, el("thead", {}, el("tr", {}, el("th", {}, "column"), el("th", {}, "text"), el("th", {}, "image"))), el("tbody", {}, rows))));
  }
  card.append(el("div", {class: "row", style: "margin-top:8px"}, el("button", {class: "btn ghost", onclick: async () => {
    const roles = {};
    for (const b of boxes) { roles[b.t] ??= {text_columns: [], image_columns: []}; if (b.tx.checked) roles[b.t].text_columns.push(b.c); if (b.im.checked) roles[b.t].image_columns.push(b.c); }
    try { await api(`/api/uploads/${name}/roles`, {method: "PUT", body: JSON.stringify({roles})}); renderTables(await api(`/api/datasets/${info.dataset}/preview`)); msg("Import", "Column roles saved."); }
    catch (e) { msg("Error", e.message); }
  }}, "Save roles"), el("button", {class: "btn", onclick: () => $("#build-graph").click()}, "Build metadata graph"),
    el("span", {class: "sub"}, "Image columns hold image files, bytes, or URLs.")));
  setTab("chat");
}
const drop = $("#drop");
for (const e of ["dragenter", "dragover"]) drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); });
for (const e of ["dragleave", "drop"]) drop.addEventListener(e, () => drop.classList.remove("over"));
drop.addEventListener("drop", (ev) => { ev.preventDefault(); $("#upload-files").files = ev.dataTransfer.files; showFiles(); });
$("#upload-files").onchange = showFiles;
function showFiles() { const f = [...$("#upload-files").files]; $("#drop-files").textContent = f.length ? `${f.length} file${f.length > 1 ? "s" : ""}: ${f.map((x) => x.name).slice(0, 4).join(", ")}${f.length > 4 ? " …" : ""}` : ""; }
$("#upload-form").onsubmit = async (e) => {
  e.preventDefault();
  const name = $("#upload-name").value.trim(), files = $("#upload-files").files;
  if (!name || !files.length) return alert("Give the dataset a name and choose files.");
  const fd = new FormData(); fd.append("name", name); for (const f of files) fd.append("files", f);
  try {
    const r = await fetch("/api/uploads", {method: "POST", body: fd}); const body = await r.json(); if (!r.ok) throw new Error(body.detail);
    await renderDatasets(); await selectDataset(body.dataset);
    msg("Import", `Imported ${body.manifest.tables.length} table(s), ${body.manifest.images.length} image(s), ${body.manifest.texts.length} text file(s)` + (body.manifest.skipped.length ? `; skipped ${body.manifest.skipped.join(", ")}` : "") + ".");
  } catch (err) { msg("Error", err.message); }
};
$("#build-graph").onclick = async () => {
  if (!S.dataset) return alert("Choose a dataset first.");
  try {
    const r = await api("/api/graphs", {method: "POST", body: JSON.stringify({dataset: S.dataset})});
    if (r.status === "ready") { renderGraph(r.graph); setTab("graph"); msg("TiMetaGraph", "Loaded the cached metadata graph."); return; }
    const log = el("div", {class: "progress"}), cancel = el("button", {class: "btn ghost", onclick: () => api(`/api/graphs/${r.build}/cancel`, {method: "POST"})}, "Cancel");
    msg("TiMetaGraph", el("div", {class: "q"}, "Building the metadata graph offline"),
      el("div", {class: "desc"}, "TiProcessor per modality, cross-modal entity resolution, relationship discovery."), log, el("div", {style: "margin-top:8px"}, cancel));
    const es = new EventSource(`/api/graphs/${r.build}/events`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.event === "progress") log.append(el("div", {}, ev.message));
      else { es.close(); cancel.disabled = true;
        if (ev.event === "graph") { log.append(el("div", {class: "done"}, "Done.")); renderGraph(ev.graph); setTab("graph"); }
        else log.append(el("div", {}, `${ev.event}: ${ev.message}`)); }
    };
  } catch (e) { msg("Error", e.message); }
};

/* ---------------- live: ask, clarify, run ---------------- */
$("#ask-form").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#question").value.trim(); if (!q || !S.dataset) return;
  $("#question").value = ""; setTab("chat"); msg("You", q);
  try { const r = await api("/api/sessions", {method: "POST", body: JSON.stringify({dataset: S.dataset, question: q})}); S.session = r.session; await handleRound(r); }
  catch (err) { msg("Error", err.message); }
};
async function handleRound(r) {
  if (r.round) {
    roundCard(r.round, async (answer) => {
      msg("You", answer);
      try { await handleRound(await api(`/api/sessions/${S.session}/answer`, {method: "POST", body: JSON.stringify({answer})})); } catch (err) { msg("Error", err.message); }
    });
    return;
  }
  msg("TiQC", el("span", {}, "No ambiguity reaches θamb. Planning: ", el("b", {}, r.query)));
  const wait = msg("TiTSP", "Planning (MCTS over the operator compatibility graph, uncertainty propagation, Pareto selection) and executing …");
  try {
    const out = await api(`/api/sessions/${S.session}/run`, {method: "POST"});
    wait.remove(); renderEvidence(out.run);
    msg("TiMMInsight", el("div", {class: "answer"}, out.run.answer), el("div", {class: "meta"}, `${fmt(out.run.result.rows)} result rows · ${out.run.seconds} s`));
  } catch (err) { wait.remove(); msg("Error", err.message); }
}

async function restoreSession(sid) {
  const t = await api(`/api/sessions/${sid}`);
  S.dataset = t.dataset; S.session = sid; $("#crumb-dataset").textContent = t.dataset; renderDatasets();
  renderTables(await api(`/api/datasets/${t.dataset}/preview`));
  try { renderGraph(await api(`/api/graphs/view?dataset_name=${encodeURIComponent(t.dataset)}`)); } catch (_) {}
  msg("You", t.question);
  for (const rd of t.rounds) { roundCard(rd, null); msg("You", rd.answer, el("div", {class: "rewrite"}, "Rewritten: ", el("b", {}, rd.query_after))); }
  if (t.round) return handleRound({round: t.round});
  if (t.run) {
    msg("TiQC", el("span", {}, t.rounds.length ? "No remaining ambiguity reaches θamb. Clarified query: " : "No ambiguity reaches θamb. Planned: ", el("b", {}, t.query)));
    renderEvidence(t.run);
    msg("TiMMInsight", el("div", {class: "answer"}, t.run.answer), el("div", {class: "meta"}, `${fmt(t.run.result.rows)} result rows · ${t.run.seconds} s`));
  }
}

/* ---------------- start; deep links: /?session=<id> or /?dataset=<name>[&graph=1], [&tab=chat|graph|data] ---------------- */
(async () => {
  await renderDatasets();
  const q = new URLSearchParams(location.search);
  if (q.get("session")) await restoreSession(q.get("session"));
  else if (q.get("dataset")) { await selectDataset(q.get("dataset")); if (q.get("graph")) await $("#build-graph").onclick(); }
  if (q.get("tab")) setTab(q.get("tab"));
})();
