/* B3 checkpoint playback viewer.
 *
 * The page shows the REAL playback simulation: the loaded actor controls the
 * MuJoCo env, and this page consumes the same SSE feed the live-training
 * viewers do (spec -> rig, frame -> poses, reset/episode_end -> record).
 * Deterministic (tanh(mu)) and stochastic (sampled sigma) playback are
 * labelled explicitly; the mode switch applies at the next episode.
 */
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const fmt = (v, n = 3) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (typeof v === 'number' ? v.toFixed(n) : String(v));
const short = (s, n = 10) => (s ? String(s).slice(0, n) : '—');

// MuJoCo is z-up (three's default is y-up). The z-up convention stays local
// to THIS page's camera (no global Object3D.DEFAULT_UP mutation).
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1014);
scene.fog = new THREE.Fog(0x0d1014, 18, 60);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 300);
camera.up.set(0, 0, 1);
camera.position.set(3.3, -2.4, 1.6);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
$('scene').appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.85);
controls.maxPolarAngle = Math.PI / 2 - 0.02;
let followRoot = false;
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const rigMeshes = new Map();          // body name -> Object3D
const grid = new THREE.GridHelper(30, 30, 0x2a3341, 0x1a212b);
scene.add(grid);
scene.add(new THREE.HemisphereLight(0xbdd2ee, 0x22262e, 0.9));
const keyLight = new THREE.DirectionalLight(0xfff4e0, 1.5);
keyLight.position.set(3.5, -5.5, 7.5);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0x9db9ff, 0.5);
fillLight.position.set(-3, 4.5, 2.5);
scene.add(fillLight);

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
    case 'capsule':
      geo = new THREE.CapsuleGeometry(Math.max(s[0], 1e-4), Math.max(s[1] * 2, 1e-4), 8, 12);
      geo.rotateX(Math.PI / 2);             // three.js Y-axis -> MuJoCo Z-axis
      break;
    case 'cylinder':
      geo = new THREE.CylinderGeometry(s[0], s[0], s[1] * 2, 16);
      geo.rotateX(Math.PI / 2);             // three.js Y-axis -> MuJoCo Z-axis
      break;
    case 'box':
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
    mat.side = THREE.DoubleSide;
  }
  return mesh;
}

function buildScene(spec) {
  for (const [, m] of rigMeshes) scene.remove(m);
  rigMeshes.clear();
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
    `rig: ${rigCount} bodies · island: ${otherCount} · meshes not drawn: ${skipped}` +
    (spec.joints ? ` · joints: ${spec.joints.length}` : '');
}

/* ---------------------------------------------------------------- frames */
let epStartPos = null;
let epRootPos = null;
let epDecision = 0;

function applyFrame(frame) {
  epDecision = frame.decision ?? epDecision;
  for (const [name, pose] of Object.entries(frame.bodies || {})) {
    let obj = rigMeshes.get(name);
    if (!obj) {
      obj = new THREE.Group();
      rigMeshes.set(name, obj);   // a body the spec missed: still posed honestly
      scene.add(obj);
    }
    obj.position.set(pose.p[0], pose.p[1], pose.p[2]);
    obj.quaternion.set(pose.q[1], pose.q[2], pose.q[3], pose.q[0]);
  }
  const root = frame.root && frame.root.pos ? frame.root.pos : null;
  if (root) {
    if (epStartPos === null) epStartPos = [root[0], root[1]];
    epRootPos = [root[0], root[1]];
    updateLiveAxis(frame.target_velocity);
  }
  if (followRoot && root) {
    controls.target.lerp(new THREE.Vector3(root[0], root[1], 0.9), 0.06);
  }
  $('epDecision').textContent = fmt(epDecision, 0);
  $('epTime').textContent = fmt(frame.sim_t, 2) + ' s';
}

function updateLiveAxis(targetVel) {
  if (epStartPos === null || epRootPos === null) return;
  const tv = (targetVel || [0, 0.5, 0]);
  const h = Math.hypot(tv[0], tv[1]) || 1;
  const hx = tv[0] / h, hy = tv[1] / h;
  const dx = epRootPos[0] - epStartPos[0];
  const dy = epRootPos[1] - epStartPos[1];
  $('epAxis').textContent = fmt(dx * hx + dy * hy, 3) + ' m';
}

/* --------------------------------------------------------------- controls */
let lastState = null;

function refreshControls() {
  const st = lastState && lastState.state;
  const running = st === 'running';
  $('pauseBtn').disabled = !running;
  $('resumeBtn').disabled = !(lastState && st === 'paused');
  $('resetBtn').disabled = !(running || st === 'paused');
  const cfg = lastState && lastState.config;
  if (cfg && cfg.inference && cfg.inference !== 'deterministic') {
    $('modeBtn').textContent = 'To deterministic';
  } else {
    $('modeBtn').textContent = 'To stochastic';
  }
  $('modeBtn').disabled = !running || (lastState && lastState.mode_next);
  const badge = $('modeBadge');
  const mode = (lastState && lastState.mode) || 'deterministic';
  badge.className = 'modebadge ' + mode;
  badge.textContent = mode === 'deterministic'
    ? 'DETERMINISTIC · mean action tanh(μ)'
    : `STOCHASTIC · sampled · σ ${fmt(lastState && lastState.identity && lastState.identity.sigma_mean, 3)}`;
  const note = $('modeNote');
  if (lastState && lastState.mode_next && lastState.mode_next !== mode) {
    note.style.display = '';
    $('modePending').textContent =
      `switch to ${lastState.mode_next.toUpperCase()} applies at the next episode`;
  } else note.style.display = 'none';
  const pill = $('pill');
  pill.textContent = st || 'booting';
  pill.className = 'pill ' + (st || 'stopped');
  const err = $('sessionError');
  if ((st === 'error' || st === 'starting') && lastState && lastState.error) {
    err.textContent = lastState.error;
    err.hidden = false;
  } else err.hidden = true;
  if (lastState) $('speed').value = lastState.speed ?? 1;
}

function renderIdentity(idt) {
  if (!idt || !idt.content_sha256) {
    $('idCk').textContent = 'verifying…';
    return;
  }
  $('idCk').textContent = idt.checkpoint_name || '—';
  $('idSteps').textContent = `${fmt(idt.trained_steps, 0)} steps · run ${idt.run} · seed ${idt.seed}`;
  $('idSha').textContent = short(idt.content_sha256, 14) + ' ✓ VERIFIED';
  $('idContract').textContent =
    `obs ${idt.obs_dim} / act ${idt.action_dim} · rl_env/${idt.contract_version} ✓`;
  const actorOk = idt.actor_unmodified;
  $('idActor').textContent = (actorOk ? 'UNMODIFIED ' : 'CHANGED! ')
    + short(idt.actor_sha256, 8);
  $('idActor').style.color = actorOk ? 'var(--good)' : 'var(--bad)';
  const rig = idt.rig || {};
  $('idRig').textContent = rig.verified
    ? `B3-era · era sha ${short(rig.sha256, 12)}`
    : 'unverified';
  $('idHead').textContent = rig.compiled_head_mass_kg
    ? `${rig.compiled_head_mass_kg.toFixed(6)} kg @ +${rig.compiled_head_ipos[2]} m`
    : '—';
  $('idRigSha').textContent = short(rig.sha256, 14) + ' ✓';
  $('idEra').textContent = idt.era_note || '';
}

function renderStatus(st) {
  const totals = st.totals || {};
  $('totRow').textContent =
    `ep ${totals.episodes_started ?? '—'} · ok ${totals.completed ?? '—'} `
    + `· falls ${totals.falls ?? '—'} · faults ${totals.faulted ?? '—'}`;
  renderIdentity(st.identity || {});
  const le = st.last_episode;
  if (le) {
    $('lastHead').textContent = `#${le.episode} · seed ${le.seed} · ${le.mode}`;
    $('lastDur').textContent = `${fmt(le.sim_time_s, 2)} s · ${fmt(le.steps, 0)} decisions`;
    $('lastDisp').textContent = `${fmt(le.net_displacement_m)} / ${fmt(le.axis_displacement_m)} m`;
    $('lastFalls').textContent =
      `${le.falls} / ${le.faulted_windows}` + (le.truncated ? ' · complete' : '');
    $('lastWarn').textContent = (le.physics_warnings && le.physics_warnings.length)
      ? le.physics_warnings.join(', ') : 'none';
  }
  $('epNo').textContent = st.episode ? fmt(st.episode, 0) : '—';
}

function flashBanner(text, kind = 'info') {
  const b = $('banner');
  b.textContent = text;
  b.style.borderColor = kind === 'bad' ? 'var(--bad)' : kind === 'good' ? 'var(--good)' : 'var(--edge)';
  b.style.opacity = 1;
  clearTimeout(b._t);
  b._t = setTimeout(() => { b.style.opacity = 0; }, kind === 'info' ? 4000 : 1500);
}

async function post(path, body = {}) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!j.ok && j.error) flashBanner(`${j.error.code}: ${j.error.message}`, 'bad');
  return j;
}

$('pauseBtn').onclick = () => post('/api/rl_playback/pause');
$('resumeBtn').onclick = () => post('/api/rl_playback/resume');
$('resetBtn').onclick = () => post('/api/rl_playback/reset');
$('modeBtn').onclick = () => {
  const cfg = lastState && lastState.config;
  const current = (lastState && lastState.mode) || (cfg && cfg.inference) || 'deterministic';
  const want = current === 'deterministic' ? 'stochastic' : 'deterministic';
  flashBanner(`switching to ${want.toUpperCase()} at the next episode`, 'info');
  post('/api/rl_playback/mode', { mode: want });
};
$('speed').oninput = (e) => $('speedOut').textContent = Number(e.target.value).toFixed(2) + '×';
$('speed').onchange = (e) => post('/api/rl_playback/speed', { speed: Number(e.target.value) });

$('camSide').onclick = () => {
  camera.position.set(3.3, -2.4, 1.6);
  controls.target.set(0, 0, 0.85);
};
$('camFront').onclick = () => {
  camera.position.set(0.1, -4.6, 1.35);
  controls.target.set(0, 0, 0.9);
};
$('camFollow').onclick = (e) => {
  followRoot = !followRoot;
  e.target.classList.toggle('on', followRoot);
};

/* ------------------------------------------------------------------ boot */
async function bootSnapshot() {
  try {
    const st = await fetch('/api/rl_live/status').then(r => r.json());
    if (st.session) { lastState = st.session; refreshControls(); renderStatus(lastState); }
    const spec = await fetch('/api/rl_live/spec').then(r => r.json());
    if (spec && spec.available !== false) buildScene(spec);
    else $('sceneNote').textContent =
      'playback is building the verified B3-era rig — frames follow shortly';
  } catch (e) { $('pill').textContent = 'offline'; }
  const ago = $('idAgo');
  lastState && lastState.identity && lastState.identity.content_sha256
    ? (ago.textContent = '') : (ago.textContent = '');
}
bootSnapshot();

/* ------------------------------------------------------------------- SSE */
const ev = new EventSource('/api/rl_live/stream');

ev.addEventListener('spec', (e) => buildScene(JSON.parse(e.data)));
ev.addEventListener('frame', (e) => applyFrame(JSON.parse(e.data)));
ev.addEventListener('status', (e) => {
  lastState = JSON.parse(e.data);
  refreshControls();
  renderStatus(lastState);
});
ev.addEventListener('reset', (e) => {
  const d = JSON.parse(e.data);
  epStartPos = d.start_pos ? [d.start_pos[0], d.start_pos[1]] : null;
  $('epNo').textContent = fmt(d.episode, 0);
});
ev.addEventListener('episode_end', (e) => {
  const d = JSON.parse(e.data);
  if (d.falls > 0) flashBanner(`fell (${d.reason}) · ep ${d.episode} · ${fmt(d.net_displacement_m)} m`, 'bad');
  else if (d.faulted_windows > 0) flashBanner(`physics fault — window excluded · ep ${d.episode}`, 'bad');
  else if (d.truncated) flashBanner(`episode complete · ${fmt(d.net_displacement_m)} m net · ${fmt(d.sim_time_s, 1)} s`, 'good');
});
ev.onerror = () => { $('pill').textContent = 'reconnecting…'; };
ev.onopen = () => bootSnapshot();

/* --------------------------------------------------------- render loop */
function tick() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
