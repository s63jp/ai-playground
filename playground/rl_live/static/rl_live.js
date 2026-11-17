/* Live Neural Training viewer.
 *
 * Everything rendered here comes from the training process itself:
 *  - one scene spec (the MuJoCo model's bodies/geoms) after a session starts,
 *  - per-decision frames of the rig being trained (body world poses),
 *  - the trainer's own per-iteration telemetry records,
 *  - a read-only tail of the separate M6-B3 background run.
 *
 * Fallback: if no geom spec has arrived yet, HumanoidView (the existing
 * dashboard module) renders the name-keyed skeleton so the page is still
 * honest about what ran when.
 */
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';
import { HumanoidView } from '/static/js/humanoid_view.js';

const $ = (id) => document.getElementById(id);
const fmt = (v, n = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (typeof v === 'number' ? v.toFixed(n) : String(v));

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);           // MuJoCo is z-up

/* ------------------------------------------------------------------ scene */
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1014);
scene.fog = new THREE.Fog(0x0d1014, 18, 60);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 300);
camera.position.set(4.6, -4.6, 1.7);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
$('scene').appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.8);
controls.maxPolarAngle = Math.PI / 2 - 0.02;
let followRoot = false;
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const rigMeshes = new Map();          // body name -> Object3D
let humanoidFallback = new HumanoidView(scene);
const grid = new THREE.GridHelper(30, 30, 0x2a3341, 0x1a212b);
scene.add(grid);

function geomMesh(g) {
  const rgba = (g.rgba && g.rgba.length >= 4) ? g.rgba : [0.7, 0.72, 0.78, 1];
  const alpha = rgba[3];
  if (alpha <= 0.02) return null;
  let geo = null;
  const s = g.size;
  switch (g.type) {
    case 'plane':
      geo = new THREE.PlaneGeometry(2 * (s[0] || 15), 2 * (s[1] || 15));
      break;
    case 'sphere':
      geo = new THREE.SphereGeometry(s[0], 20, 14);
      break;
    case 'capsule':                         // [radius, half-len]
      geo = new THREE.CapsuleGeometry(Math.max(s[0], 1e-4), Math.max(s[1] * 2, 1e-4), 8, 12);
      break;
    case 'cylinder':
      geo = new THREE.CylinderGeometry(s[0], s[0], s[1] * 2, 16);
      break;
    case 'box':                             // half extents -> full
      geo = new THREE.BoxGeometry(2 * s[0] || 2e-4, 2 * s[1] || 2e-4, 2 * s[2] || 2e-4);
      break;
    case 'ellipsoid':
      geo = new THREE.SphereGeometry(1, 16, 12);
      break;
    default:
      return null;
  }
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
    transparent: alpha < 0.98, opacity: alpha,
    roughness: 0.7, metalness: 0.05,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.fromArray(g.pos || [0, 0, 0]);
  if (g.quat && g.quat.length === 4) {
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  }
  if (g.type === 'ellipsoid') mesh.scale.set(s[0], s[1], s[2]);
  if (g.type === 'plane') {
    mesh.receiveShadow = false;
    mat.side = THREE.DoubleSide;
  }
  return mesh;
}

function buildScene(spec) {
  for (const [, m] of rigMeshes) scene.remove(m);
  rigMeshes.clear();
  if (humanoidFallback) { scene.remove(humanoidFallback.group); humanoidFallback = null; }
  let rigCount = 0, otherCount = 0, skipped = spec.meshes_skipped || 0;
  for (const body of spec.bodies || []) {
    const group = new THREE.Group();
    let drawn = 0;
    for (const g of body.geoms || []) {
      const mesh = geomMesh(g);
      if (!mesh) continue;
      if (!body.rig) mesh.material.transparent = true, mesh.material.opacity = Math.min(0.38, mesh.material.opacity);
      group.add(mesh);
      drawn++;
    }
    if (!drawn) continue;
    group.position.fromArray(body.pos || [0, 0, 0]);
    if (body.quat && body.quat.length === 4) group.quaternion.set(body.quat[1], body.quat[2], body.quat[3], body.quat[0]);
    if (body.rig) rigCount++; else otherCount++;
    rigMeshes.set(body.name, group);
    scene.add(group);
  }
  grid.position.y = 0;
  $('sceneNote').textContent =
    `rig: ${rigCount} bodies · island: ${otherCount} bodies · meshes not drawn: ${skipped}` +
    (spec.joints ? ` · joints: ${spec.joints.length}` : '');
}

function applyFrame(frame) {
  if (humanoidFallback) {   // no geom spec yet (should not happen after spec)
    scene.remove(humanoidFallback.group);
    humanoidFallback = null;
  }
  const rootName = frame.bodies && Object.keys(frame.bodies)[0];
  for (const [name, pose] of Object.entries(frame.bodies || {})) {
    let obj = rigMeshes.get(name);
    if (!obj) {
      obj = new THREE.Group();
      rigMeshes.set(name, obj);
      scene.add(obj);
    }
    obj.position.set(pose.p[0], pose.p[1], pose.p[2]);
    obj.quaternion.set(pose.q[1], pose.q[2], pose.q[3], pose.q[0]);
  }
  if (followRoot && frame.root) {
    controls.target.lerp(new THREE.Vector3(frame.root.pos[0], frame.root.pos[1], 0.9), 0.06);
  }
}

/* -------------------------------------------------------------- telemetry */
const seriesAxis = [];
const seriesReward = [];
const MAX_POINTS = 360;

function pushSeries(arr, v) {
  if (v === null || v === undefined) return;
  arr.push(v);
  if (arr.length > MAX_POINTS) arr.shift();
  if (arr.length > MAX_POINTS / 3 && arr.length % 3 === 0) {
    const d = [];
    for (let i = 0; i < arr.length; i += 3) d.push(arr[i]);
    arr.length = 0;
    arr.push(...d);
  }
}

function drawSpark(canvas, arr) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (arr.length < 2) return;
  let lo = Infinity, hi = -Infinity;
  for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-9) { hi = lo + 1; }
  ctx.beginPath();
  for (let i = 0; i < arr.length; i++) {
    const x = (i / (arr.length - 1)) * (w - 4) + 2;
    const y = h - 3 - ((arr[i] - lo) / (hi - lo)) * (h - 8);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.strokeStyle = '#5b8cff';
  ctx.lineWidth = 1.6;
  ctx.stroke();
}

function onIteration(rec) {
  $('telemetryAgo').textContent = 'it ' + fmt(rec.iteration, 0);
  $('mReward').textContent = fmt(rec.mean_reward);
  $('mDisp').textContent = fmt(rec.roll_net_displacement_mean);
  $('mAxis').textContent = fmt(rec.roll_axis_displacement_mean);
  $('mUpright').textContent = fmt(rec.roll_upright_fraction_mean);
  $('mFalls').textContent = fmt(rec.roll_falls, 0);
  $('mEffort').textContent = fmt(rec.roll_effort_mean_w_mean, 1);
  $('mSigma').textContent = fmt(rec.sigma, 4);
  $('mVloss').textContent = fmt(rec.value_loss, 1);
  $('mKl').textContent = fmt(rec.approx_kl, 4);
  $('mClip').textContent = fmt(rec.clip_frac, 3);
  pushSeries(seriesAxis, rec.roll_axis_displacement_mean ?? rec.roll_net_displacement_mean);
  pushSeries(seriesReward, rec.mean_reward);
  drawSpark($('sparkAxis'), seriesAxis);
  drawSpark($('sparkReward'), seriesReward);
  if (rec.stage) $('liveStage').textContent = rec.stage;
  if (rec.iteration !== undefined) $('liveIter').textContent = rec.iteration;
  if (rec.total_valid_steps !== undefined) $('liveSteps').textContent = rec.total_valid_steps;
}

function flashBanner(text, kind = 'info') {
  const b = $('banner');
  b.textContent = text;
  b.style.borderColor = kind === 'bad' ? 'var(--bad)' : kind === 'good' ? 'var(--good)' : 'var(--edge)';
  b.style.opacity = 1;
  clearTimeout(b._t);
  b._t = setTimeout(() => { b.style.opacity = 0; }, kind === 'info' ? 4000 : 1500);
}

/* ------------------------------------------------------- controls & state */
let lastState = null;
const buttons = ['startBtn', 'pauseBtn', 'resumeBtn', 'stopBtn', 'saveBtn'];

function refreshControls() {
  const st = lastState && lastState.state;
  const map = {
    idle:      { start: true },
    starting:  {},
    running:   { pause: true, stop: true },
    paused:    { resume: true, stop: true, save: true },
    stopped:   { start: true, save: (lastState && (lastState.config)) ? true : false },
    error:     { start: true },
  };
  const en = map[st] || {};
  $('startBtn').disabled = !en.start;
  $('pauseBtn').disabled = !en.pause;
  $('resumeBtn').disabled = !en.resume;
  $('stopBtn').disabled = !en.stop;
  $('saveBtn').disabled = !en.save;
  const pill = $('pill');
  pill.textContent = st || 'booting';
  pill.className = 'pill ' + (st || 'stopped');
  const err = $('sessionError');
  if (st === 'error' && lastState && lastState.error) {
    err.textContent = lastState.error;
    err.hidden = false;
  } else err.hidden = true;
  const cfg = lastState && lastState.config;
  if (cfg) {
    $('runInfo').style.display = '';
    $('runName').textContent = `${cfg.name} · obj ${cfg.objective} · seed ${cfg.seed}`;
  }
}

async function post(path, body = {}) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!j.ok && j.error) flashBanner(`${j.error.code}: ${j.error.message}`, 'info');
  return j;
}

$('startBtn').onclick = () => post('/api/rl_live/start', {
  objective: $('objective').value,
  seed: Number($('seed').value),
  total_steps: Number($('steps').value),
});
$('pauseBtn').onclick = () => post('/api/rl_live/pause');
$('resumeBtn').onclick = () => post('/api/rl_live/resume');
$('stopBtn').onclick = () => {
  if (confirm('Stop training and save the final checkpoint?')) post('/api/rl_live/stop');
};
$('saveBtn').onclick = async () => {
  const tag = prompt('Checkpoint tag', 'manual_1');
  if (tag !== null) post('/api/rl_live/save', { tag });
};

/* ---------------------------------------------------------------- boot */
async function loadObjectives() {
  try {
    const j = await fetch('/api/rl_live/objectives').then(r => r.json());
    const sel = $('objective');
    sel.innerHTML = '';
    for (const o of j.objectives || []) {
      const opt = document.createElement('option');
      opt.value = o.id;
      opt.textContent = o.label;
      sel.appendChild(opt);
    }
    if (!j.trainer_available) {
      const e = $('sessionError');
      e.textContent = 'trainer modules not on this tree yet — telemetry and viewer work, training starts after the B3 commit lands';
      e.hidden = false;
      $('startBtn').disabled = true;
    }
  } catch (e) { /* offline: SSE state will inform */ }
}

async function bootSnapshot() {
  try {
    const st = await fetch('/api/rl_live/status').then(r => r.json());
    if (st.session) { lastState = st.session; refreshControls(); }
    const spec = await fetch('/api/rl_live/spec').then(r => r.json());
    if (spec && spec.available !== false) buildScene(spec);
    else if (lastState && lastState.state === 'running') {
      $('sceneNote').textContent = 'waiting for the first frames…';
    } else $('sceneNote').textContent = 'start a session to build the rig scene from the live model.';
  } catch (e) { $('pill').textContent = 'offline'; }
}

loadObjectives();
bootSnapshot();

/* ------------------------------------------------------------------ SSE */
const ev = new EventSource('/api/rl_live/stream');

ev.addEventListener('spec', (e) => buildScene(JSON.parse(e.data)));
ev.addEventListener('frame', (e) => applyFrame(JSON.parse(e.data)));
ev.addEventListener('iteration', (e) => onIteration(JSON.parse(e.data)));
ev.addEventListener('b3_iteration', (e) => onB3(JSON.parse(e.data)));
ev.addEventListener('status', (e) => {
  lastState = JSON.parse(e.data);
  refreshControls();
});
ev.addEventListener('episode_end', (e) => {
  const d = JSON.parse(e.data);
  $('liveEpisode').textContent = d.episode ?? '—';
  if (d.falls > 0) flashBanner(`fell (${d.reason}) · ep ${d.episode} · ${fmt(d.net_displacement_m)} m`, 'bad');
  else if (d.faulted_windows > 0) flashBanner(`physics fault excluded · ep ${d.episode}`, 'bad');
  else if (d.truncated) flashBanner(`episode complete · ${fmt(d.net_displacement_m)} m net`, 'good');
});
ev.addEventListener('stage_transition', (e) => {
  const d = JSON.parse(e.data);
  flashBanner(`curriculum → ${d.to_stage} (iteration ${d.iteration})`, 'good');
});
ev.addEventListener('saved', (e) => {
  const d = JSON.parse(e.data);
  flashBanner(`checkpoint saved: ${d.path} (${fmt(d.steps, 0)} steps)`, 'good');
});
ev.onerror = () => { $('pill').textContent = 'reconnecting…'; };
ev.onopen = () => { bootSnapshot(); };

/* ---------------------------------------------------------- B3 panel */
let b3Last = null;
function onB3(rec) {
  b3Last = Date.now();
  $('b3State').textContent = 'training';
  $('b3Steps').textContent = fmt(rec.total_valid_steps, 0);
  $('b3Iter').textContent = fmt(rec.iteration, 0);
  $('b3Stage').textContent = rec.stage ?? '—';
  $('b3Sps').textContent = fmt(rec.steps_per_s, 1);
  $('b3Disp').textContent = fmt(rec.roll_net_displacement_mean);
  $('b3Upright').textContent = fmt(rec.roll_upright_fraction_mean);
  $('b3Sigma').textContent = fmt(rec.sigma, 4);
}

async function pollB3() {
  try {
    const j = await fetch('/api/rl_b3/telemetry?limit=1').then(r => r.json());
    if (!j.available) {
      $('b3State').textContent = '—';
      $('b3Ago').textContent = 'not configured';
      return;
    }
    if (j.latest) onB3(j.latest);
    $('b3Ago').textContent = b3Last ? `${Math.round((Date.now() - b3Last) / 1000)}s ago` : '';
  } catch (e) { /* offline */ }
}
setInterval(pollB3, 5000);
pollB3();

/* -------------------------------------------------------- render loop */
function tick() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
