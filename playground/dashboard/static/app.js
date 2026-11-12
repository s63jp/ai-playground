// AI Experimental Playground dashboard. Read-only: it never sends anything to a running agent.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const $ = (s) => document.querySelector(s);
// XSS policy: agents (language models) author free text that this page displays. Every value that can
// carry agent- or API-supplied text goes through esc() before being placed in HTML. Values interpolated
// without esc() are server-side enums, numbers, or ids validated server-side (agent ids: [a-z0-9_-]).
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const api = async (p, opt) => { const r = await fetch(p, opt); if (!r.ok) throw new Error(`${p}: ${r.status} ${await r.text()}`); return r.json(); };
const fmt = (v, d = 2) => (typeof v === 'number' ? v.toFixed(d) : v ?? '–');

// ------------------------------------------------------------------ tabs
function showTab(name) {
  const b = document.querySelector(`#tabs button[data-tab="${name}"]`);
  if (!b) return;
  document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === `tab-${name}`));
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);   // deep link to a view
  ({ memory: loadMemory, analytics: loadAnalytics, compare: loadCompare, bench: loadBench }[name] || (() => {}))();
  if (name === 'watch') resize();
}
document.querySelectorAll('#tabs button').forEach((b) => b.onclick = () => showTab(b.dataset.tab));

// ------------------------------------------------------------------ 3D viewer
const spec = await api('/api/world/spec');
const viewer = $('#viewer');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewer.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1014);
// World coordinates are z-up; three.js is y-up. One group rotates the whole world.
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 400);
camera.position.set(-14, 18, 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(1, 1, 0);
scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-10, 25, 12);
scene.add(sun);

const H = spec.half_extent;
const sea = new THREE.Mesh(new THREE.PlaneGeometry(H * 6, H * 6), new THREE.MeshStandardMaterial({ color: 0x1b3b57 }));
sea.position.z = spec.sea_level + 1.9;
world.add(sea);
const grid = new THREE.GridHelper(2 * H, 2 * H, 0x555555, 0x3a3a3a);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.005;
world.add(grid);
const axes = new THREE.AxesHelper(3);
axes.position.set(0, 0, 0.01);
world.add(axes);
for (let k = -H; k <= H; k += 4) {          // coordinate labels every 4 m
  for (const [x, y] of [[k, -H - 0.8], [-H - 0.8, k]]) {
    const c = document.createElement('canvas'); c.width = 64; c.height = 32;
    const g = c.getContext('2d'); g.fillStyle = '#8b93a1'; g.font = '22px sans-serif'; g.fillText(String(k), 4, 24);
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c) }));
    s.position.set(x, y, 0.2); s.scale.set(1.2, 0.6, 1); world.add(s);
  }
}

const meshes = {};
function makeMesh(o) {
  const col = new THREE.Color(o.colour);
  const mat = new THREE.MeshStandardMaterial({ color: col, roughness: 0.8 });
  let geo;
  if (o.shape === 'sphere') geo = new THREE.SphereGeometry(o.dimensions.radius, 24, 16);
  else if (o.shape === 'torus') { geo = new THREE.TorusGeometry(o.dimensions.major_radius, o.dimensions.tube_radius, 12, 40); mat.metalness = 0.6; }
  else geo = new THREE.BoxGeometry(o.dimensions.x, o.dimensions.y, o.dimensions.z);
  const m = new THREE.Mesh(geo, mat);
  m.position.set(...o.position);
  if (o.kind === 'block') { const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x111111 })); m.add(e); }
  world.add(m);
  return m;
}
for (const o of spec.objects) meshes[o.id] = makeMesh(o);
const blockTemplate = spec.objects.find((o) => o.kind === 'block');
function ensureMesh(id) {
  if (meshes[id]) return meshes[id];
  const colour = spec.colours[id] || '#ffffff';
  meshes[id] = makeMesh({ ...blockTemplate, id, colour, position: [0, 0, -50] });   // generated in-world
  return meshes[id];
}
const agentMesh = new THREE.Mesh(new THREE.BoxGeometry(spec.agent_body.width, spec.agent_body.width, spec.agent_body.height),
  new THREE.MeshStandardMaterial({ color: 0x5aa9ff, emissive: 0x0b2a4a }));
const nose = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.3, 12), new THREE.MeshStandardMaterial({ color: 0xffffff }));
nose.position.set(0, 0, spec.agent_body.height / 2 + 0.25); agentMesh.add(nose);
world.add(agentMesh);
const trails = new THREE.Group(); world.add(trails);

function resize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();
(function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

// ------------------------------------------------------------------ replay state
// Frames are stored as deltas with periodic keyframes; we reconstruct full poses on load.
let frames = [], poses = [], cur = 0, playing = false, sessionId = null, lastSeen = { a: 0, e: 0, m: 0 };
let flights = [];

function rebuild(from = 0) {
  let state = from > 0 ? structuredClone(poses[from - 1]) : { objects: {}, present: null };
  for (let i = from; i < frames.length; i++) {
    const f = frames[i];
    if (f.key) { state = { objects: {}, present: f.present }; }
    for (const [id, p] of Object.entries(f.objects)) state.objects[id] = p;
    if (f.present) for (const id of Object.keys(state.objects)) if (!f.present.includes(id)) delete state.objects[id];
    poses[i] = { t: f.t, agent: f.agent, held: f.held, objects: { ...state.objects }, present: state.present };
  }
}
function show(i) {
  if (!poses.length) return;
  cur = Math.max(0, Math.min(i, poses.length - 1));
  const p = poses[cur];
  for (const [id, m] of Object.entries(meshes)) {
    const s = spec.objects.find((o) => o.id === id);
    if (s && !['block', 'rock', 'ring'].includes(s.kind)) continue;
    m.visible = !!p.objects[id];
  }
  for (const [id, q] of Object.entries(p.objects)) {
    const m = ensureMesh(id);
    m.visible = true; m.position.set(q[0], q[1], q[2]);
    m.quaternion.set(q[4], q[5], q[6], q[3]);   // MuJoCo w,x,y,z -> three x,y,z,w
  }
  agentMesh.position.set(...p.agent);
  $('#r-scrub').value = cur; $('#r-time').textContent = `t=${fmt(p.t, 1)}s  frame ${cur + 1}/${poses.length}`;
  trails.children.forEach((l) => { l.visible = l.userData.t0 <= p.t; });
}
$('#r-scrub').oninput = (e) => { playing = false; $('#r-play').textContent = '▶'; show(+e.target.value); };
$('#r-play').onclick = () => { playing = !playing; $('#r-play').textContent = playing ? '❚❚' : '▶'; if (playing && cur >= poses.length - 1) show(0); };
setInterval(() => {
  if (!playing || !poses.length) return;
  const speed = +$('#r-speed').value, tNext = poses[cur].t + 0.05 * speed;
  let i = cur; while (i < poses.length - 1 && poses[i].t < tNext) i++;
  show(i); if (i >= poses.length - 1) { playing = false; $('#r-play').textContent = '▶'; }
}, 50);

function drawFlights(evs) {
  // Trajectory reconstructed only from recorded launch state + gravity; the endpoint is the recorded impact.
  for (const e of evs) {
    if (e.type !== 'flight_end' || e.data.kind !== 'throw' || !e.data.flight_time) continue;
    const d = e.data, p0 = d.launch_position, v = d.launch_velocity, pts = [];
    for (let k = 0; k <= 30; k++) {
      const t = d.flight_time * k / 30;
      pts.push(new THREE.Vector3(p0[0] + v[0] * t, p0[1] + v[1] * t, p0[2] + v[2] * t - 4.905 * t * t));
    }
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineDashedMaterial({ color: 0xe8b04b, dashSize: 0.2, gapSize: 0.12 }));
    l.computeLineDistances(); l.userData.t0 = d.t_start; trails.add(l);
  }
}

// ------------------------------------------------------------------ watch tab
const agents = (await api('/api/agents')).agents.map((a) => a.agent_id);
for (const sel of ['#w-agent', '#m-agent', '#a-agent']) for (const a of agents) $(sel).insertAdjacentHTML('beforeend', `<option>${esc(a)}</option>`);

async function loadSessions() {
  const a = $('#w-agent').value;
  const s = (await api(`/api/sessions?agent_id=${a}`)).sessions;
  $('#w-session').innerHTML = s.map((x) => `<option value="${+x.id}">#${+x.id} ${esc(x.mode)}${x.task_id ? ':' + esc(x.task_id) : ''} s${+x.session_number} (${esc(x.status)})</option>`).join('');
  if (s.length) await openSession(s[0].id);
}
async function openSession(id) {
  sessionId = +id; frames = []; poses = []; flights = []; lastSeen = { a: 0, e: 0, m: 0 };
  trails.clear();
  ['#w-actions', '#w-msgs', '#w-exps'].forEach((s) => $(s).innerHTML = '');
  await refresh(true);
  show(poses.length - 1);
}
$('#w-agent').onchange = loadSessions;
$('#w-session').onchange = (e) => openSession(e.target.value);

async function refresh(first = false) {
  if (sessionId == null) return;
  const sid = sessionId;
  const [fr, acts, evs, msgs, detail] = await Promise.all([
    api(`/api/sessions/${sid}/frames?start=${frames.length}&limit=5000`),
    api(`/api/sessions/${sid}/actions?after=${lastSeen.a}`),
    api(`/api/sessions/${sid}/events?after=${lastSeen.e}&types=flight_end,object_appeared,object_lost,collision`),
    api(`/api/sessions/${sid}/messages?after=${lastSeen.m}`),
    api(`/api/sessions/${sid}`),
  ]);
  if (sid !== sessionId) return;
  const atEnd = cur >= poses.length - 1;
  if (fr.frames.length) { const from = frames.length; frames.push(...fr.frames); rebuild(from); $('#r-scrub').max = poses.length - 1; }
  for (const a of acts.actions) {
    lastSeen.a = a.idx;
    const r = a.result, err = r.success ? '' : ` <span class="bad">${esc(r.error.code)}</span>`;
    let extra = '';
    if (a.tool === 'throw' && r.success && r.max_height != null) extra = ` peak ${fmt(r.max_height)} m → ${esc((r.resting_on || []).join(','))}`;
    if (a.tool === 'record_discovery' && r.success) extra = ` asked ${r.requested_status}, granted <span class="pill ${r.granted_status}">${r.granted_status}</span>`;
    $('#w-actions').insertAdjacentHTML('afterbegin', `<div class="row mono small">#${a.idx} ${a.source !== 'agent' ? '<span class="muted">[' + a.source + ']</span> ' : ''}${esc(a.tool)}${err}${extra}</div>`);
    if (a.tool === 'record_experiment' && r.success) {
      const v = r.evidence_check;
      $('#w-exps').insertAdjacentHTML('afterbegin', `<div class="row"><b>${r.experiment_id}</b> ${esc(a.args.hypothesis)}<br>
        <span class="small muted">result:</span> ${esc(a.args.actual_result)} · <span class="small">${a.args.supported ? 'supported' : 'not supported'}</span>
        · evidence <span class="${v.counts_as_evidence ? 'ok' : 'bad'}">${v.verdict}</span></div>`);
    }
  }
  if (evs.events.length) { lastSeen.e = evs.events[evs.events.length - 1].n; drawFlights(evs.events); }
  for (const m of msgs.messages) {
    lastSeen.m = m.id;
    if (m.role === 'assistant') $('#w-msgs').insertAdjacentHTML('afterbegin', `<div class="row say">“${esc(m.content)}”</div>`);
  }
  const st = detail.final_state;
  const s = detail.summary;
  $('#w-status').textContent = `${detail.status}${detail.end_reason ? ' · ' + detail.end_reason : ''} · ${detail.action_count} actions · sim ${fmt(detail.sim_time, 1)} s · ${detail.adapter}/${detail.model}`;
  const last = poses[poses.length - 1];
  $('#w-state').innerHTML = last ? `<div class="mono small">body at [${last.agent.map((v) => fmt(v)).join(', ')}] · holding ${esc(last.held || 'nothing')}</div>
    ${s ? `<div class="small">behaviour: ${Object.entries(s.behaviour).map(([k, v]) => `${k.replace(/_/g, ' ')} <b>${v}</b>`).join(' · ')}</div>` : ''}` : '';
  const recs = await api(`/api/sessions/${sid}/records?kind=goal`);
  const goals = {};
  for (const g of recs.records) goals[g.key] = { ...(goals[g.key] || {}), ...g.data };
  $('#w-goals').innerHTML = Object.entries(goals).map(([k, g]) => `<div class="row"><b>${k}</b> ${esc(g.title)} <span class="pill">${g.status}</span>
    ${g.independently_verified === true ? '<span class="ok small">verified by physics</span>' : g.independently_verified === false ? '<span class="bad small">claim NOT verified</span>' : ''}</div>`).join('') || '<span class="muted">none set</span>';
  const integ = await api(`/api/sessions/${sid}/integrity`);
  const iv = integ.what_was_independently_verified, ic = integ.what_the_agent_claims;
  $('#w-integ').innerHTML = `<table>
    <tr><th></th><th>claimed by agent</th><th>independently verified</th></tr>
    <tr><td>achievements</td><td>${ic.achievement_claims}</td><td><span class="ok">${iv.claims_verified}</span> / <span class="bad">${iv.claims_rejected} rejected</span></td></tr>
    <tr><td>experiments</td><td>${ic.experiments_recorded}</td><td><span class="ok">${iv.experiments_consistent_with_engine} consistent</span> · <span class="bad">${iv.experiments_contradicted_by_engine} contradicted</span> · ${iv.experiments_without_valid_evidence} no evidence</td></tr>
    <tr><td>"confirmed" discoveries</td><td>${ic.discoveries_requested_confirmed} requested</td><td>${iv.discoveries_granted_confirmed} granted</td></tr>
    </table><div class="small ${integ.event_chain.ok ? 'ok' : 'bad'}">event log hash chain: ${integ.event_chain.ok ? 'intact' : 'BROKEN'} (${integ.event_chain.events} events)</div>`;
  if ($('#w-follow').checked && (first || atEnd) && !playing) show(poses.length - 1);
}
setInterval(async () => {
  if (!$('#tab-watch').classList.contains('active')) return;
  try {
    if ($('#w-follow').checked) {
      const live = (await api('/api/live')).running.filter((r) => !r.stale);
      if (live.length && live[0].id !== sessionId && live[0].agent_id) {
        $('#w-agent').value = live[0].agent_id; await loadSessions(); return;
      }
    }
    await refresh();
  } catch (e) { $('#w-status').textContent = String(e); }
}, 1500);

// ------------------------------------------------------------------ memory tab
let memFile = 'handbook', memData = null;
document.querySelectorAll('#m-filetabs button').forEach((b) => b.onclick = () => {
  document.querySelectorAll('#m-filetabs button').forEach((x) => x.classList.toggle('active', x === b));
  memFile = b.dataset.f; renderFile();
});
function renderFile() {
  if (!memData) return;
  let txt = memData.files[memFile];
  if (memFile === 'experiments') txt = JSON.stringify(memData.experiments, null, 2);
  if (memFile === 'programs') txt = JSON.stringify(memData.programs, null, 2);
  $('#m-file').textContent = txt;
}
async function loadMemory() {
  const a = $('#m-agent').value;
  memData = await api(`/api/agents/${a}/memory`);
  const s = memData.stats;
  $('#m-stats').textContent = `${s.sessions_completed} sessions · ${s.discoveries_total} handbook entries (${s.discoveries.confirmed} confirmed, ${s.discoveries.probable} probable, ${s.discoveries.hypothesis} hypotheses, ${s.discoveries.disproven} disproven) · ${s.experiments_verified}/${s.experiments} experiments pass the evidence check`;
  $('#m-items').innerHTML = memData.items.map((i) => `<div class="row"><span class="mono">${i.id}</span>
    <span class="pill ${i.status || ''}">${i.kind}${i.status ? ': ' + i.status : ''}</span> ${esc((i.text || '').slice(0, 180))}
    ${i.flags && i.flags.challenged ? '<span class="warn small">challenged</span>' : ''}
    <div>${i.kind === 'discovery' ? `<button class="btn" data-op="challenge" data-id="${i.id}">challenge</button>
      <button class="btn" data-op="mark_unverified" data-id="${i.id}">mark unverified</button>` : ''}
      <button class="btn" data-op="remove" data-id="${i.id}">remove</button></div></div>`).join('');
  $('#m-items').querySelectorAll('button').forEach((b) => b.onclick = async () => {
    const reason = prompt(`Reason for ${b.dataset.op} ${b.dataset.id}?`); if (reason === null) return;
    await api(`/api/agents/${a}/memory/${b.dataset.id}/${b.dataset.op}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reason }) });
    loadMemory();
  });
  $('#m-audit').innerHTML = memData.audit.map((x) => `<div class="row small">${new Date(x.ts * 1000).toLocaleString()} · ${esc(x.actor)} · <b>${esc(x.op)}</b> ${esc(x.item_id)} ${esc(x.reason || '')}</div>`).join('') || '<span class="muted">no researcher interventions</span>';
  renderFile();
}
$('#m-agent').onchange = loadMemory;

// ------------------------------------------------------------------ analytics tab
async function loadAnalytics() {
  const a = $('#a-agent').value;
  const tl = (await api(`/api/analytics${a ? '?agent_id=' + a : ''}`)).timeline.filter((t) => t.mode === 'exploration');
  const max = Math.max(1, ...tl.map((t) => t.action_count || 0));
  $('#a-chart').innerHTML = `<div class="bars">${tl.map((t) => `<div title="#${t.id} ${esc(t.agent_id)} s${t.session_number}: ${t.action_count} actions" style="height:${100 * (t.action_count || 0) / max}%"></div>`).join('')}</div>
    <div class="small muted">exploration sessions in order (bar = actions)</div>`;
  $('#a-table').innerHTML = `<table><tr><th>session</th><th>agent</th><th>#</th><th>actions</th><th>failed</th><th>sim s</th>
    <th>experiments ok/bad</th><th>claims ok/rejected</th><th>handbook</th><th>goals</th><th>pretend</th><th>repeat+vary</th><th>own sake</th><th>end</th></tr>
    ${tl.map((t) => `<tr><td>${t.id}</td><td>${esc(t.agent_id)}</td><td>${t.session_number}</td><td>${t.action_count}</td><td>${t.failed_actions ?? '–'}</td><td>${fmt(t.sim_time, 0)}</td>
    <td>${t.experiments_ok ?? '–'}/${t.experiments_bad ?? '–'}</td><td>${t.claims_verified ?? '–'}/${t.claims_rejected ?? '–'}</td>
    <td>${t.discoveries ? Object.entries(t.discoveries).map(([k, v]) => v ? `${v} ${k}` : '').filter(Boolean).join(', ') : '–'}</td>
    <td>${t.behaviour?.self_directed_goals ?? '–'}</td><td>${t.behaviour?.pretend_play ?? '–'}</td><td>${t.behaviour?.repetition_with_variation ?? '–'}</td><td>${t.behaviour?.activity_for_its_own_sake ?? '–'}</td><td>${t.end_reason ?? ''}</td></tr>`).join('')}</table>`;
}
$('#a-agent').onchange = loadAnalytics;

// ------------------------------------------------------------------ compare tab
async function loadCompare() {
  const rows = (await api('/api/compare')).agents;
  $('#c-table').innerHTML = `<table><tr><th>agent</th><th>sessions</th><th>confirmed</th><th>probable</th><th>hypotheses</th><th>disproven</th>
    <th>experiments (pass evidence check)</th><th>memory quality</th><th>mistakes</th><th>programs</th><th>claims verified</th>
    <th>goals</th><th>pretend</th><th>repeat+vary</th><th>own sake</th><th>benchmark completion</th></tr>
    ${rows.map((r) => `<tr><td>${esc(r.agent_id)}</td><td>${r.sessions_completed}</td><td>${r.discoveries.confirmed}</td><td>${r.discoveries.probable}</td>
    <td>${r.discoveries.hypothesis}</td><td>${r.discoveries.disproven}</td><td>${r.experiments_verified}/${r.experiments}</td><td>${r.memory_quality ?? '–'}</td>
    <td>${r.mistakes}</td><td>${r.programs}</td><td>${r.claims_verified}/${r.claims}</td><td>${r.behaviour.self_directed_goals}</td><td>${r.behaviour.pretend_play}</td>
    <td>${r.behaviour.repetition_with_variation}</td><td>${r.behaviour.activity_for_its_own_sake}</td><td>${r.benchmark_completion_rate ?? '–'}</td></tr>`).join('')}</table>
    <div class="small muted">memory quality = share of recorded experiments whose claims agree with the engine's measurements.</div>`;
  $('#c-disc').innerHTML = rows.map((r) => `<div class="row"><b>${esc(r.agent_id)}</b>${r.discovery_statements.map((d) => `<div class="small">
    <span class="pill ${d.status}">${d.status}</span> ${esc(d.statement)}</div>`).join('') || ' <span class="muted">nothing yet</span>'}</div>`).join('');
}

// ------------------------------------------------------------------ benchmarks tab
async function loadBench() {
  const runs = (await api('/api/benchmarks')).runs;
  $('#b-runs').innerHTML = runs.map((r) => {
    const key = r.kind === 'ablation' ? 'condition' : 'grp';
    const groups = [...new Set(r.results.map((x) => x[key]))];
    const tasks = [...new Set(r.results.map((x) => x.task_id))];
    return `<div class="card"><b>Run ${r.id}</b> · ${r.kind} · ${esc(r.label || '')} <span class="muted small">${new Date(r.created_at * 1000).toLocaleString()}</span>
      ${r.kind === 'ablation' ? `<div class="small">removed from ${esc(r.params.agent_id)}: ${r.params.removed_items.map((i) => `<b>${i.id}</b> ${esc(i.text)}`).join('; ')}</div>` : `<div class="small">Group A (experienced): ${esc(r.params.group_a.join(', '))} · Group B (fresh): ${esc(r.params.group_b.join(', '))}</div>`}
      <table><tr><th>task</th>${groups.map((g) => `<th>${g}: done</th><th>score</th><th>actions</th><th>failed</th><th>sim s</th><th>memory used</th>`).join('')}</tr>
      ${tasks.map((t) => `<tr><td>${esc(t)}</td>${groups.map((g) => { const x = r.results.filter((y) => y.task_id === t && y[key] === g);
        const done = x.filter((y) => y.completed).length;
        const avg = (k) => x.length ? fmt(x.reduce((s, y) => s + (y[k] || 0), 0) / x.length, 1) : '–';
        return `<td class="${done ? 'ok' : 'bad'}">${done}/${x.length}</td><td>${x.map((y) => fmt(y.score)).join(', ')}</td><td>${avg('actions')}</td><td>${avg('failed_actions')}</td><td>${avg('sim_time')}</td>
        <td>${x.map((y) => (y.memory_use.relevant_handbook_entries_at_start || []).join(',') || '–').join(' ')}</td>`; }).join('')}</tr>`).join('')}</table></div>`;
  }).join('') || '<span class="muted">no benchmark runs yet</span>';
}

// ------------------------------------------------------------------ boot
const h = await api('/api/health');
$('#health').textContent = `v${h.version} · config ${h.config_hash.slice(0, 10)} · ${h.agents} agents`;
await loadSessions();
const startTab = location.hash.slice(1);
if (startTab && startTab !== 'watch') showTab(startTab);
