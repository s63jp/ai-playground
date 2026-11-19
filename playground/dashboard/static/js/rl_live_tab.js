/* Live Neural Training tab for the main dashboard.
 *
 * All rendered state comes from the training process itself over
 * /api/rl_live/stream (SSE): one scene spec (the MuJoCo rig's bodies/geoms),
 * per-decision frames of the rig actually being trained, the trainer's own
 * iteration telemetry, and a READ-ONLY tail of the separate M6-B3 run.
 * Fallback when no spec exists yet: HumanoidView name-keyed rendering.
 */
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';
import { HumanoidView } from './humanoid_view.js';
import { createHumanoidV2 } from '/static/js/humanoid_v2.js?v=2';

const $ = (id) => document.getElementById(id);
const fmt = (v, n = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (typeof v === 'number' ? v.toFixed(n) : String(v));
const kvRows = (pairs) => `<table class="mono" style="width:100%;font-size:12px">${pairs
  .map(([k, v]) => `<tr><td class="muted">${k}</td><td style="text-align:right">${v}</td></tr>`)
  .join('')}</table>`;

// MuJoCo is z-up (three's default is y-up). Do NOT touch the global
// Object3D.DEFAULT_UP here: it silently re-orients every other dashboard
// viewer constructed after this module (the island renderer in app.js came
// out edge-on). The z-up convention stays local to THIS scene's camera.
/* ------------------------------------------------------------------ scene */
const holder = $('live-view');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1014);
const camera = new THREE.PerspectiveCamera(50, 1.4, 0.1, 300);
camera.up.set(0, 0, 1);                      // MuJoCo rig is z-up
camera.position.set(4.6, -4.6, 1.7);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
holder.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.8);
controls.maxPolarAngle = Math.PI / 2 - 0.02;
scene.add(new THREE.GridHelper(30, 30, 0x2a3341, 0x1a212b));
// lighting: the rig uses MeshStandardMaterial, which is BLACK without lights
// (z-up scene: key light from above +z, slightly ahead of the orbit view)
scene.add(new THREE.HemisphereLight(0xbdd2ee, 0x22262e, 0.9));
const keyLight = new THREE.DirectionalLight(0xfff4e0, 1.5);
keyLight.position.set(3.5, -5.5, 7.5);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0x9db9ff, 0.5);
fillLight.position.set(-3, 4.5, 2.5);
scene.add(fillLight);

const rigMeshes = new Map();
let humanoidFallback = new HumanoidView(scene);

/* ---- Humanoid V2 (Blender FBX, retargeted from live MuJoCo frames) ----
 * Default when it loads and verifies. ?rig=v1 keeps the primitive V1 rig.
 * Any V2 error at load or per frame falls back to V1 and says so. */
const v2Requested = new URLSearchParams(location.search).get('rig') !== 'v1';
let v2 = null;
const rigBodyNames = new Set();
function setRigPrimitivesVisible(visible) {
  for (const name of rigBodyNames) { const g = rigMeshes.get(name); if (g) g.visible = visible; }
}
function v2Fallback(reason) {
  console.error('humanoid V2 disabled:', reason);
  if (v2) { v2.dispose(); v2 = null; }
  window.__humanoidV2 = null;
  setRigPrimitivesVisible(true);
  flash(`V2 humanoid unavailable, showing V1: ${reason}`);
}
if (v2Requested) {
  createHumanoidV2(scene).then((h) => {
    v2 = h;
    window.__humanoidV2 = h;
    setRigPrimitivesVisible(false);
  }).catch((e) => v2Fallback(e.message));
}

function resize() {
  const w = holder.clientWidth, h = holder.clientHeight || 340;
  if (!w) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
new ResizeObserver(resize).observe(holder);

function geomMesh(g) {
  const rgba = (g.rgba && g.rgba.length >= 4) ? g.rgba : [0.7, 0.72, 0.78, 1];
  if (rgba[3] <= 0.02) return null;
  const s = g.size;
  let geo = null;
  switch (g.type) {
    case 'plane': geo = new THREE.PlaneGeometry(2 * (s[0] || 15), 2 * (s[1] || 15)); break;
    case 'sphere': geo = new THREE.SphereGeometry(s[0], 20, 14); break;
    case 'capsule': geo = new THREE.CapsuleGeometry(Math.max(s[0], 1e-4), Math.max(s[1] * 2, 1e-4), 8, 12); geo.rotateX(Math.PI / 2); break;  // Y-axis -> MuJoCo Z-axis
    case 'cylinder': geo = new THREE.CylinderGeometry(s[0], s[0], s[1] * 2, 16); geo.rotateX(Math.PI / 2); break;  // Y-axis -> MuJoCo Z-axis
    case 'box': geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
    case 'ellipsoid': geo = new THREE.SphereGeometry(1, 16, 12); break;
    default: return null;
  }
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
    transparent: rgba[3] < 0.98, opacity: rgba[3], roughness: 0.7, metalness: 0.05,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.fromArray(g.pos || [0, 0, 0]);
  if (g.quat && g.quat.length === 4) mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  if (g.type === 'ellipsoid') mesh.scale.set(s[0], s[1], s[2]);
  if (g.type === 'plane') mat.side = THREE.DoubleSide;
  return mesh;
}

function buildScene(spec) {
  for (const [, m] of rigMeshes) scene.remove(m);
  rigMeshes.clear();
  rigBodyNames.clear();
  if (humanoidFallback) { scene.remove(humanoidFallback.group); humanoidFallback = null; }
  let rig = 0, other = 0;
  for (const body of spec.bodies || []) {
    const group = new THREE.Group();
    let drawn = 0;
    for (const g of body.geoms || []) {
      const mesh = geomMesh(g);
      if (!mesh) continue;
      if (!body.rig) { mesh.material.transparent = true; mesh.material.opacity = Math.min(0.38, mesh.material.opacity); }
      group.add(mesh); drawn++;
    }
    if (!drawn) continue;
    group.position.fromArray(body.pos || [0, 0, 0]);
    if (body.quat && body.quat.length === 4) {
      group.quaternion.set(body.quat[1], body.quat[2], body.quat[3], body.quat[0]);
    } else {
      group.quaternion.identity();
    }
    if (body.rig) { rig++; rigBodyNames.add(body.name); } else other++;
    rigMeshes.set(body.name, group);
    if (body.rig && v2) group.visible = false;   // V2 replaces the V1 primitive rig
    scene.add(group);
  }
  $('live-scene-note').textContent =
    `· rig ${rig} bodies, island ${other} · meshes not drawn: ${spec.meshes_skipped ?? 0}`
    + (v2 ? ` · V2: ${v2.meshes.size} FBX meshes` : '');
}

function applyFrame(f) {
  if (f.t === 'reset') { if (v2) v2.reset(); return; }
  if (humanoidFallback) { scene.remove(humanoidFallback.group); humanoidFallback = null; }
  for (const [name, pose] of Object.entries(f.bodies || {})) {
    let obj = rigMeshes.get(name);
    if (!obj) { obj = new THREE.Group(); rigMeshes.set(name, obj); scene.add(obj); }
    obj.position.set(pose.p[0], pose.p[1], pose.p[2]);
    obj.quaternion.set(pose.q[1], pose.q[2], pose.q[3], pose.q[0]);
  }
  if (v2 && f.bodies) {
    try { v2.update(f); } catch (e) { v2Fallback(e.message); }
  }
  controls.target.lerp(new THREE.Vector3(f.root.pos[0], f.root.pos[1], 0.9), 0.05);
  $('live-episode').value = f.episode;
  $('live-decision').value = f.decision;
  $('live-upz').value = f.up_z.toFixed(2);
  $('live-contacts').value = `${fmt(f.contacts.left, 0)} / ${fmt(f.contacts.right, 0)} N`;
}

/* ------------------------------------------------------------------ sparks */
const seriesAxis = [], seriesReward = [], MAX_POINTS = 360;
function pushSeries(arr, v) {
  if (v === null || v === undefined) return;
  arr.push(v);
  if (arr.length > MAX_POINTS) arr.shift();
  if (arr.length > MAX_POINTS / 3 && arr.length % 3 === 0) {
    const d = []; for (let i = 0; i < arr.length; i += 3) d.push(arr[i]);
    arr.length = 0; arr.push(...d);
  }
}
function drawSpark(canvas, arr) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (arr.length < 2) return;
  let lo = Infinity, hi = -Infinity;
  for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-9) hi = lo + 1;
  ctx.beginPath();
  for (let i = 0; i < arr.length; i++) {
    const x = (i / (arr.length - 1)) * (w - 4) + 2;
    const y = h - 3 - ((arr[i] - lo) / (hi - lo)) * (h - 8);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.strokeStyle = '#5b8cff'; ctx.lineWidth = 1.6; ctx.stroke();
}

/* ----------------------------------------------------------------- panels */
function onIteration(rec) {
  $('live-tele-iter').textContent = `iteration ${rec.iteration} · step ${rec.total_valid_steps}`;
  $('live-telemetry').innerHTML = kvRows([
    ['mean reward', fmt(rec.mean_reward)],
    ['net disp / ep (m)', fmt(rec.roll_net_displacement_mean)],
    ['axis disp / ep (m)', fmt(rec.roll_axis_displacement_mean)],
    ['upright fraction', fmt(rec.roll_upright_fraction_mean)],
    ['falls / iteration', fmt(rec.roll_falls, 0)],
    ['effort (W)', fmt(rec.roll_effort_mean_w_mean, 1)],
    ['σ (action noise)', fmt(rec.sigma, 4)],
    ['value loss', fmt(rec.value_loss, 1)],
    ['approx KL', fmt(rec.approx_kl, 4)],
    ['clip fraction', fmt(rec.clip_frac, 3)],
  ]);
  pushSeries(seriesAxis, rec.roll_axis_displacement_mean ?? rec.roll_net_displacement_mean);
  pushSeries(seriesReward, rec.mean_reward);
  drawSpark($('live-spark-axis'), seriesAxis);
  drawSpark($('live-spark-reward'), seriesReward);
  if (rec.stage) $('live-stage').value = rec.stage;
  if (rec.iteration !== undefined) $('live-iter').value = rec.iteration;
  if (rec.total_valid_steps !== undefined) $('live-step').value = rec.total_valid_steps;
  if (rec.roll_comp_limb_tracking_mean !== undefined)
    $('live-s1-track').value = fmt(rec.roll_comp_limb_tracking_mean, 3);
}

let bannerT = null;
function flash(text) {
  const el = $('live-state');
  const base = el.dataset.base || '';
  el.textContent = `${base} · ${text}`;
  clearTimeout(bannerT);
  bannerT = setTimeout(() => { el.textContent = base; }, 4000);
}

let lastState = null;
function refreshControls() {
  const st = lastState && lastState.state;
  const en = {
    idle: { start: 1 }, starting: {}, running: { pause: 1, stop: 1 },
    paused: { resume: 1, stop: 1, save: 1 },
    stopped: { start: 1, save: +(lastState && lastState.config || 0) },
    error: { start: +(lastState && lastState.trainer_available || 0) },
  }[st] || {};
  $('live-start').disabled = !en.start;
  $('live-pause').disabled = !en.pause;
  $('live-resume').disabled = !en.resume;
  $('live-stop').disabled = !en.stop;
  $('live-save').disabled = !en.save;
  $('live-state').dataset.base = st || 'idle';
  $('live-state').textContent = st === 'running'
    ? `training · step ${lastState.total_valid_steps ?? 0} · stage ${lastState.stage ?? '—'}`
    : (st === 'error' && lastState.error) ? `error: ${lastState.error}` : `${st ?? 'booting'} — nothing training on this page yet`;
  const err = $('live-error');
  err.textContent = (st === 'error' && lastState && lastState.error) ? lastState.error : '';
  const cfg = lastState && lastState.config;
  if (cfg) $('live-run-name').value = `${cfg.name} · obj ${cfg.objective} · seed ${cfg.seed}`;
  if (cfg && cfg.s1) {
    $('live-s1-card').hidden = false;
    $('live-s1-group').value = (cfg.limb_stage_id || '').replace('m7_s1_', '') || '—';
  }
  // Evaluate needs a quiescent torch state and a saved checkpoint.
  const cks = (lastState && lastState.checkpoints) || [];
  $('live-eval').disabled = !(cfg && cfg.s1 && (st === 'stopped' || (st === 'paused' && cks.length)));
  $('live-trainer-note').textContent = lastState && lastState.trainer_available === false
    ? '· trainer modules not on this tree yet (B3 pending): viewer + telemetry only'
    : '';
}

/* --------------------------------------------------------------- controls */
async function post(path, body = {}) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!j.ok && j.error) flash(`${j.error.code}: ${j.error.message}`);
  return j;
}
/* ------------- architecture + S1 individual-limb wiring ------------- */
let OBJ = [];
const objMeta = (id) => OBJ.find((o) => o.id === id) || {};

function archHidden() {
  const v = $('live-arch').value;
  const raw = v === 'custom' ? $('live-arch-custom').value : v;
  return String(raw).split(',').map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n > 0);
}
function syncArchCustom() { $('live-arch-custom').parentElement.hidden = $('live-arch').value !== 'custom'; }
$('live-arch').onchange = syncArchCustom;
syncArchCustom();

function onObjectiveChange() {
  const m = objMeta($('live-objective').value);
  const isS1 = !!m.s1;
  $('live-warmrow').hidden = !isS1;
  $('live-s1-card').hidden = !isS1;
  if (isS1) $('live-s1-group').value = (m.limb_stage_id || '').replace('m7_s1_', '') || '—';
}
$('live-objective').onchange = onObjectiveChange;

function startBody() {
  const ppo = { hidden: archHidden() };
  const lr = Number($('live-lr').value);
  if (Number.isFinite(lr) && lr > 0) ppo.lr = lr;
  const body = {
    objective: $('live-objective').value,
    seed: Number($('live-seed').value),
    total_steps: Number($('live-steps').value),
    ppo_overrides: ppo,
  };
  if (objMeta($('live-objective').value).s1) {
    const warm = parseInt($('live-warmup').value, 10);
    if (Number.isFinite(warm) && warm >= 0) body.stage_budgets = { standing: warm };
  }
  return body;
}

$('live-start').onclick = () => post('/api/rl_live/start', startBody());
$('live-pause').onclick = () => post('/api/rl_live/pause');
$('live-resume').onclick = () => post('/api/rl_live/resume');
$('live-stop').onclick = () => { if (confirm('Stop training and save the final checkpoint?')) post('/api/rl_live/stop'); };
$('live-save').onclick = () => {
  const tag = prompt('Checkpoint tag', 'manual_1');
  if (tag !== null) post('/api/rl_live/save', { tag });
};

function latestCheckpointName() {
  const st = lastState && lastState.state;
  if (st === 'stopped') return 'ckpt_final.json';
  const cks = (lastState && lastState.checkpoints) || [];
  if (!cks.length) return 'ckpt_final.json';
  return (cks[cks.length - 1].path || '').split(/[\\/]/).pop() || 'ckpt_final.json';
}

function renderVerdict(v) {
  const el = $('live-s1-verdict');
  el.textContent = v.mastered ? 'MASTERED ✓' : 'not yet';
  el.style.color = v.mastered ? 'var(--good, #57c98b)' : 'var(--warn, #e2b93b)';
  const rows = Object.entries(v.joints || {}).map(([n, r]) =>
    [`${n} ${r.pass ? '✓' : '✗'}`, `${fmt(r.steady_abs_err_rad)} / ${fmt(r.band_rad)} rad`]);
  rows.push(['posture chest-up', `${fmt(v.posture.chest_up_z_min)} (≥${v.posture.band}) ${v.posture.pass ? '✓' : '✗'}`]);
  rows.push(['falls', `${v.falls.count} ${v.falls.pass ? '✓' : '✗'} · ${v.episodes} held-out episodes`]);
  $('live-s1-bands').innerHTML = kvRows(rows);
}

$('live-eval').onclick = async () => {
  const cfg = lastState && lastState.config;
  if (!cfg || !cfg.s1) return;
  $('live-s1-note').textContent = 'evaluating…';
  const r = await post('/api/rl_live/evaluate', {
    experiment: cfg.name, checkpoint: latestCheckpointName(),
    stage_id: cfg.limb_stage_id, episodes: 12,
  });
  if (r.ok && r.verdict) { renderVerdict(r.verdict); $('live-s1-note').textContent = ''; }
  else $('live-s1-note').textContent = (r.error && r.error.message) || 'eval failed';
};

/* -------------------------------------------------------------------- SSE */
const ev = new EventSource('/api/rl_live/stream');
ev.addEventListener('spec', (e) => buildScene(JSON.parse(e.data)));
ev.addEventListener('frame', (e) => { try { applyFrame(JSON.parse(e.data)); } catch (_) { /* partial */ } });
ev.addEventListener('reset', () => { if (v2) v2.reset(); });
ev.addEventListener('iteration', (e) => onIteration(JSON.parse(e.data)));
ev.addEventListener('b3_iteration', (e) => onB3(JSON.parse(e.data)));
ev.addEventListener('status', (e) => { lastState = JSON.parse(e.data); refreshControls(); });
ev.addEventListener('episode_end', (e) => {
  const d = JSON.parse(e.data);
  if (d.falls > 0) flash(`fell (${d.reason}) · ep ${d.episode} · ${fmt(d.net_displacement_m)} m`);
  else if (d.faulted_windows > 0) flash(`physics fault excluded · ep ${d.episode}`);
  else if (d.truncated) flash(`episode complete · ${fmt(d.net_displacement_m)} m net`);
});
ev.addEventListener('stage_transition', (e) => {
  const d = JSON.parse(e.data);
  flash(`curriculum → ${d.to_stage} (iteration ${d.iteration})`);
});
ev.addEventListener('saved', (e) => {
  const d = JSON.parse(e.data);
  flash(`checkpoint saved: ${d.path}`);
});
ev.onerror = () => { $('live-state').textContent = 'reconnecting…'; };
ev.onopen = () => bootSnapshot();

/* ------------------------------------------------------------------- B3 */
let b3First = null, b3Steps = null;
function onB3(rec) {
  if (rec.total_valid_steps !== b3Steps) {
    b3Steps = rec.total_valid_steps;
    b3First = Date.now();          // first observation of this fresh record
  }
  $('live-b3').innerHTML = kvRows([
    ['state', 'training'],
    ['steps', fmt(rec.total_valid_steps, 0)],
    ['iteration', fmt(rec.iteration, 0)],
    ['stage', rec.stage ?? '—'],
    ['steps/s', fmt(rec.steps_per_s, 1)],
    ['net disp / ep (m)', fmt(rec.roll_net_displacement_mean)],
    ['upright fraction', fmt(rec.roll_upright_fraction_mean)],
    ['σ', fmt(rec.sigma, 4)],
  ]);
}
setInterval(() => {
  if (b3First) $('live-b3-ago').textContent = `${Math.round((Date.now() - b3First) / 1000)}s ago`;
}, 2000);

async function pollB3() {
  try {
    const j = await fetch('/api/rl_b3/telemetry?limit=1').then(r => r.json());
    if (!j.available) { $('live-b3-ago').textContent = 'not configured'; return; }
    if (j.latest) onB3(j.latest);
  } catch (_e) { /* offline */ }
}

/* ------------------------------------------------------------------- boot */
async function bootSnapshot() {
  try {
    const objs = await fetch('/api/rl_live/objectives').then(r => r.json());
    OBJ = objs.objectives || [];
    const sel = $('live-objective');
    sel.innerHTML = '';
    for (const o of OBJ) {
      const op = document.createElement('option');
      op.value = o.id; op.textContent = o.label;
      sel.appendChild(op);
    }
    onObjectiveChange();
    const spec = await fetch('/api/rl_live/spec').then(r => r.json());
    if (spec && spec.available !== false) buildScene(spec);
    else if (humanoidFallback) $('live-scene-note').textContent = '· no scene yet — start a session to build the rig from the live model';
  } catch (_e) { /* offline: SSE status will inform */ }
  pollB3();
}

window.__rl_live_show = () => { resize(); bootSnapshot(); };
resize();

/* ------------------------------------------------------------ render loop */
/* The tab had NO render call at all: the renderer, rig and incoming frames
 * were all constructed, but the canvas never drew a single frame (a blank
 * page-background rectangle). Render every tick like the standalone viewer. */
function tick() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
