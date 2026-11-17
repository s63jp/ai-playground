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

const $ = (id) => document.getElementById(id);
const fmt = (v, n = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (typeof v === 'number' ? v.toFixed(n) : String(v));
const kvRows = (pairs) => `<table class="mono" style="width:100%;font-size:12px">${pairs
  .map(([k, v]) => `<tr><td class="muted">${k}</td><td style="text-align:right">${v}</td></tr>`)
  .join('')}</table>`;

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);      // MuJoCo is z-up (three: y-up default)

/* ------------------------------------------------------------------ scene */
const holder = $('live-view');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1014);
const camera = new THREE.PerspectiveCamera(50, 1.4, 0.1, 300);
camera.position.set(4.6, -4.6, 1.7);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
holder.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.8);
controls.maxPolarAngle = Math.PI / 2 - 0.02;
scene.add(new THREE.GridHelper(30, 30, 0x2a3341, 0x1a212b));

const rigMeshes = new Map();
let humanoidFallback = new HumanoidView(scene);
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
    case 'capsule': geo = new THREE.CapsuleGeometry(Math.max(s[0], 1e-4), Math.max(s[1] * 2, 1e-4), 8, 12); break;
    case 'cylinder': geo = new THREE.CylinderGeometry(s[0], s[0], s[1] * 2, 16); break;
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
    if (body.rig) rig++; else other++;
    rigMeshes.set(body.name, group);
    scene.add(group);
  }
  $('live-scene-note').textContent =
    `· rig ${rig} bodies, island ${other} · meshes not drawn: ${spec.meshes_skipped ?? 0}`;
}

function applyFrame(f) {
  if (humanoidFallback) { scene.remove(humanoidFallback.group); humanoidFallback = null; }
  for (const [name, pose] of Object.entries(f.bodies || {})) {
    let obj = rigMeshes.get(name);
    if (!obj) { obj = new THREE.Group(); rigMeshes.set(name, obj); scene.add(obj); }
    obj.position.set(pose.p[0], pose.p[1], pose.p[2]);
    obj.quaternion.set(pose.q[1], pose.q[2], pose.q[3], pose.q[0]);
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
$('live-start').onclick = () => post('/api/rl_live/start', {
  objective: $('live-objective').value, seed: Number($('live-seed').value),
  total_steps: Number($('live-steps').value),
});
$('live-pause').onclick = () => post('/api/rl_live/pause');
$('live-resume').onclick = () => post('/api/rl_live/resume');
$('live-stop').onclick = () => { if (confirm('Stop training and save the final checkpoint?')) post('/api/rl_live/stop'); };
$('live-save').onclick = () => {
  const tag = prompt('Checkpoint tag', 'manual_1');
  if (tag !== null) post('/api/rl_live/save', { tag });
};

/* -------------------------------------------------------------------- SSE */
const ev = new EventSource('/api/rl_live/stream');
ev.addEventListener('spec', (e) => buildScene(JSON.parse(e.data)));
ev.addEventListener('frame', (e) => { try { applyFrame(JSON.parse(e.data)); } catch (_) { /* partial */ } });
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
    const sel = $('live-objective');
    sel.innerHTML = '';
    for (const o of objs.objectives || []) {
      const op = document.createElement('option');
      op.value = o.id; op.textContent = o.label;
      sel.appendChild(op);
    }
    const spec = await fetch('/api/rl_live/spec').then(r => r.json());
    if (spec && spec.available !== false) buildScene(spec);
    else if (humanoidFallback) $('live-scene-note').textContent = '· no scene yet — start a session to build the rig from the live model';
  } catch (_e) { /* offline: SSE status will inform */ }
  pollB3();
}

window.__rl_live_show = () => { resize(); bootSnapshot(); };
resize();
