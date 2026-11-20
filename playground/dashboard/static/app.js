// AI Experimental Playground dashboard. Read-only: it never sends anything to a running agent.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const $ = (s) => document.querySelector(s);
// XSS policy: agents (language models) author free text that this page displays. Every value that can
// carry agent- or API-supplied text goes through esc() before being placed in HTML. Values interpolated
// without esc() are server-side enums, numbers, or ids validated server-side (agent ids: [a-z0-9_-]).
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
window.__bootErrors = [];
window.addEventListener('error', (e) => { window.__bootErrors.push(String(e.message || e) + ' @' + (e.filename || '') + ':' + (e.lineno || 0)); });
// Display sanitiser for "Agent's words" (reports/ADAPTER_TOOLCALL_LEAK.md): strip the chat-
// template tokens and embedded tool-call markup the model emits as text. Raw content stays in
// the database verbatim; the dashboard shows the agent's words, the template's.
const TEMPLATETOKEN = /<\|[^<>|]{1,24}\|>/g;
const FUNCBLOCK = /<function=[^<>]{0,80}>(?:.|\n)*?<\/function>|<function=[^<>]{0,80}>(?:.|\n)*/g;
const PARAMBLOCK = /<parameter=[^<>]{0,60}>(?:.|\n)*?(?:<\/parameter>|$)/g;
// A template-written call may also end with a bare tool_call close tag AFTER the
// </function> block (nemotron-3-nano-4b, sessions 79+; sessions 49-50 earlier) — the
// GGUF template writes "< .../tool_call ...>"-shaped text, and only the open side was
// covered by FUNCBLOCK. Built from char codes so this source never contains the raw
// sequence.
const CLOSETAG_RE = new RegExp(String.fromCharCode(60) + '\\/?' + 'tool_call' + String.fromCharCode(62), 'g');
const sanitizeWords = (s) => {
  let t = String(s ?? '').replace(TEMPLATETOKEN, '').replace(FUNCBLOCK, '').replace(PARAMBLOCK, '').replace(CLOSETAG_RE, '').trim();
  if (!t) t = '[tool-call markup — see Recent actions]';
  return t;
};
import { HumanoidView } from './js/humanoid_view.js';
import { CombatFX, makeMarker, makePellet, teamColour, _oid, isAccentSlug } from './js/combatfx.js?v=m62';

const api = async (p, opt) => { const r = await fetch(p, opt); if (!r.ok) throw new Error(`${p}: ${r.status} ${await r.text()}`); return r.json(); };

const fmt = (v, d = 2) => (typeof v === 'number' ? v.toFixed(d) : v ?? '–');

// ------------------------------------------------------------------ tabs
const PAGE_META = {
  watch: ['Observe', 'Watch & replay', 'Inspect the live world, replay sessions, and verify what agents claim against what the engine recorded.'],
  memory: ['Observe', 'Memory', 'Inspect persistent knowledge, evidence status, source files, and the researcher audit trail.'],
  analytics: ['Observe', 'Analytics', 'Explore session activity, behavioural signals, and independently verified outcomes over time.'],
  compare: ['Observe', 'Compare agents', 'Review all agents side by side, from memory quality and beliefs to benchmark completion.'],
  bench: ['Evaluate', 'Benchmarks', 'Compare task performance across experienced agents, fresh controls, and memory ablations.'],
  nn: ['Evaluate', 'Neural networks', 'Generate datasets, train models, and inspect quality, metrics, and model lineage.'],
  rl: ['Evaluate', 'RL environment', 'Inspect the consolidated locomotion environment contract, curriculum stages, and bounded measured probe episodes (no trainer yet).'],
  live: ['Evaluate', 'Live neural training', 'Watch a genuinely fresh network learn to walk in the real locomotion environment: the actual training simulation streamed live, the trainer\'s own telemetry, pause/resume/stop/save, and a read-only view of the background B3 run.'],
  playback: ['Evaluate', 'Checkpoint playback', 'Replay any saved brain driving the real MuJoCo simulation through its own inference — deterministic or stochastic, read-only, no PPO updates.'],
  lab: ['Evaluate', 'Run & models', 'Launch exploration runs, compare model behaviour, and audit session provenance.'],
  infra: ['System', 'Infrastructure', 'Monitor provider health, hardware utilisation, and live request queues.'],
  providers: ['System', 'Providers', 'Inspect discovered model providers, connectivity, capabilities, and available models.'],
  monitor: ['System', 'API monitor', 'Trace gateway requests, timing, token usage, and redacted request metadata.'],
  resources: ['System', 'Resources', 'Review measured hardware capacity and configure backend concurrency limits.'],
};
function showTab(name) {
  const b = document.querySelector(`#tabs button[data-tab="${name}"]`);
  if (!b) return;
  document.querySelectorAll('#tabs button').forEach((x) => {
    const active = x === b;
    x.classList.toggle('active', active);
    if (active) x.setAttribute('aria-current', 'page');
    else x.removeAttribute('aria-current');
  });
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === `tab-${name}`));
  const meta = PAGE_META[name];
  if (meta) {
    $('#page-kicker').textContent = meta[0].toUpperCase();
    $('#page-title').textContent = meta[1];
    $('#page-description').textContent = meta[2];
    document.title = `${meta[1]} · AI Playground`;
  }
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);   // deep link to a view
  window.scrollTo(0, 0);
  ({ memory: loadMemory, analytics: loadAnalytics, compare: loadCompare, bench: loadBench, nn: loadNn, rl: loadRl, lab: loadLab,
     live: () => window.__rl_live_show && window.__rl_live_show(),
     playback: () => { const f = $('#playback-frame'); if (f && f.src.endsWith('about:blank')) f.src = f.dataset.src; },
     infra: loadInfra, providers: loadProviders, monitor: loadMonitor, resources: loadResources }[name] || (() => {}))();
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

// Humanoid visualisation – created once after the world group
let humanoidView;
if (typeof HumanoidView === 'function') {
  humanoidView = new HumanoidView(world);
}
scene.add(world);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 400);
camera.position.set(-14, 18, 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(1, 1, 0);
// 2026-11-17 jank fixes: grazing-angle orbits showed the dark sea plane past the island
// edge as a broken void, and dipping below the floor plane hid the world. Clamp the orbit
// and fade the far sea into the background with scene fog.
scene.fog = new THREE.Fog(0x0d1014, 30, 110);
controls.maxPolarAngle = 1.45;   // ~83 deg — the camera never goes under/below the floor
controls.minPolarAngle = 0.12;
controls.minDistance = 2.5;
controls.maxDistance = 90;
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
const axes = new THREE.AxesHelper(1.2);   // a 3 m pole read as a weird tower in arena frames
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
// ------------------------------------------------------------------ combat visuals (M5.5)
// Markers/pellets/splats/trails are presentation over recorded physics: object meshes are
// classified from the recorded arena manifest (session config) + attribution events,
// equipment proportions come from the real EQUIPMENT registry, and no motion is invented.
const equipmentSpecs = {};                     // EQUIPMENT id -> registry entry (with dims_m)
let runModes = [];                             // backend-supported run modes for the lab selector
let equipList = [];                            // ordered EQUIPMENT entries (for dropdown labels)
try {
  const rm = await api('/api/run/modes');
  runModes = rm.modes || [];
  for (const e of (rm.equipment || [])) { equipmentSpecs[e.id] = e; equipList.push(e); }
} catch (e) { /* older backend still renders plain boxes */ }
const fx = new CombatFX(world);
let combatCtx = { match: null, teamOf: {}, pellets: {}, items: {}, _splatT: {}, _sid: null };
const itemTeam = {};                           // item object id -> team name (slug match)

function buildCombatCtx(ctx, sid) {
  // _sid binds the MANIFEST source: null here means "manifest not yet loaded from the
  // session detail" — refresh binds it once per session, then attribution accumulates.
  combatCtx = { match: ctx && ctx.match ? ctx.match : null, teamOf: {}, pellets: {}, items: {}, _splatT: {}, _sid: sid ?? null };
  for (const [t, aids] of Object.entries((ctx && ctx.match && ctx.match.teams) || {})) {
    for (const a of aids) combatCtx.teamOf[a] = t;
  }
  for (const it of ((ctx && ctx.match && ctx.match.items) || [])) combatCtx.items[it.id] = { item: it.item };
  // item ids are '<teamslug>_item_<k>' (M5 factory naming): match the slug to the team
  // names in the manifest so an item's marker wears its owning team's accent colour.
  for (const it of Object.keys(combatCtx.items)) {
    const slug = it.split('_item_')[0];
    for (const t of Object.keys((ctx && ctx.match && ctx.match.teams) || {})) {
      if (_oid(t) === slug) itemTeam[it] = t;
    }
  }
  window.__combatDebug = () => ({ manifestSid: combatCtx._sid, hasMatch: !!combatCtx.match,
    items: { ...combatCtx.items }, pelletCount: Object.keys(combatCtx.pellets).length,
    teams: { ...combatCtx.teamOf } });        // inspection hook for automated QA
}
function pelletColour(pid) {
  const owner = combatCtx.pellets[pid];
  const team = owner ? (combatCtx.teamOf[owner.agent] || owner.team) : (owner ? owner.team : null);
  return owner && (owner.agent || owner.team) ? teamColour(team || owner.agent) : 0xd8d8d8;
}
function itemColour(oid) {
  // Colour chain: manifest '<slug>_item_<k>' slug <-> team-name match ->, failing that, the
  // M5 factory naming contract (the recorded object id itself names the owning team slug)
  // -> neutral. Keeps legacy no-manifest duels (#120/#121-style) off the anonymous white.
  const slug = itemTeam[oid] || (/^[a-z]+_item_/i.test(oid) ? oid.split('_item_')[0] : null);
  return teamColour(slug || 'neutral');
}
function damageColour(src) {
  if (combatCtx.pellets[src]) return pelletColour(src);
  if (combatCtx.items[src]) return itemColour(src);   // melee strikes take the wielder's colour
  return null;                                        // unattributed hits stay uncoloured (honest)
}
function reinstateCombatMeshes() {
  // When attribution arrives after a plain block mesh exists (live polling), swap the
  // cached mesh for the real marker/pellet silhouette on the next show().
  for (const [id, m] of Object.entries(meshes)) {
    const wantMarker = combatCtx.items[id], wantPellet = combatCtx.pellets[id];
    if (wantMarker && m.userData.marker) continue;
    if (wantPellet && m.userData.pellet) continue;
    if (!wantMarker && !wantPellet) continue;
    world.remove(m);
    delete meshes[id];
  }
}
function combatEvents(evs) {
  // Paint effects respond ONLY to recorded physics events:
  // - pellet_launched  -> pellet attribution (object id -> shooter + item)
  // - weapon_reload    -> equipped-marker classification for legacy recordings
  // - agent_damaged    -> measured contact position -> splash particles in the shooter's colour
  // - collision        -> pellet-vs-surface contacts -> flat ground decals (throttled)
  for (const e of evs) {
    const d = e.data || {};
    if (e.type === 'pellet_launched') {
      if (d.object) combatCtx.pellets[d.object] = { agent: d.agent, item: d.item };
    } else if (e.type === 'weapon_reload') {
      if (d.object && d.item) {
        combatCtx.items[d.object] = { item: d.item };
        const slug = String(d.object).split('_item_')[0];
        if (slug) itemTeam[d.object] = Object.keys(combatCtx.teamOf).find((t) => _oid(t) === slug) || itemTeam[d.object];
      }
    } else if (e.type === 'equipment_equipped') {
      // Legacy recorders log the equip as a raw event (same payload shape as weapon_reload):
      // without it, pre-M5.5 duel items stay unclassified white template blocks (screenshots
      // 2026-11-17, session #120). Same classification as the action-result path below.
      if (d.object && d.item) {
        combatCtx.items[d.object] = { item: d.item };
        const slug = String(d.object).split('_item_')[0];
        if (slug) itemTeam[d.object] = Object.keys(combatCtx.teamOf).find((t) => _oid(t) === slug) || itemTeam[d.object];
      }
    } else if (e.type === 'agent_damaged') {
      const col = damageColour(d.source);
      if (col && Array.isArray(d.position) && d.position.length >= 3) {
        fx.burst(d.position[0], d.position[1], d.position[2], col, 7);
      }
    } else if (e.type === 'collision') {
      const objs = d.objects || [];
      let pid = null;
      for (const o of objs) if (combatCtx.pellets[o]) { pid = o; break; }
      if (!pid || (d.impulse || 0) < 0.3) continue;
      if (!Array.isArray(d.position) || d.position.length < 3 || d.position[2] > 0.12) continue;
      const okTime = (combatCtx._splatT[pid] == null) || (e.t - combatCtx._splatT[pid]) >= 0.4;
      if (!okTime) continue;
      combatCtx._splatT[pid] = e.t;
      fx.addSplat(d.position[0], d.position[1], d.position[2], pelletColour(pid), null, e.t); // a paint mark stamped at its recorded collision time
    }
  }
}
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
  // Combat objects (M5.5): equipment markers and pellets get real silhouettes; every
  // other generated object keeps the legacy block template.
  const itemEntry = combatCtx.items[id];
  if (itemEntry) {
    const es = equipmentSpecs[itemEntry.item] || { id: itemEntry.item };
    const col = itemColour(id);
    const g = makeMarker(es, col);
    g.userData.marker = itemEntry.item;
    meshes[id] = g;
    world.add(g);
    return g;
  }
  if (combatCtx.pellets[id]) {
    const owner = combatCtx.pellets[id];
    const radius = (equipmentSpecs[owner.item] || {}).projectile && equipmentSpecs[owner.item].projectile.radius;
    const g = makePellet(radius, pelletColour(id));
    g.userData.pellet = true;
    meshes[id] = g;
    world.add(g);
    return g;
  }
  const colour = spec.colours[id] || '#ffffff';
  // Generated object with no recorded classification (no manifest, no equip/reload evidence)
  // stays the block silhouette but at a COMPACT size: the 0.5 m template block read as a giant
  // tumbling white "moth" in replay close-ups; real equipment bodies are much smaller than that.
  meshes[id] = makeMesh({ ...blockTemplate, id, colour, dimensions: { x: 0.22, y: 0.22, z: 0.22 }, position: [0, 0, -50] });
  return meshes[id];
}
// ------------------------------------------------------------------ agent player model
// Procedural humanoid rig for the agent, built from the player-model asset (assets/humanoid_mujoco.xml):
// same joint layout, proportions and materials, scaled to the island's agent body (spec.agent_body).
// The physics agent stays the deterministic collision box — the rig is a faithful visual driven by the
// same pose stream (position per frame; heading and gait derived from velocity, like the pose replay).
const agentMesh = makeAgentRig();
// M5 arena replay: simple standing rigs for every OTHER recorded combatant. The shared
// world frames carry each agent's root position (frame.agents); the session's own agent
// keeps the full makeAgentRig figure. Values are internal ids/numbers only (no HTML).
const extraRigs = new Map();
const EXTRA_RIG_COLORS = [0xff7755, 0x55cc77, 0xddaa44, 0xcc66dd, 0x55dddd];
function ensureExtraRig(aid) {
  let rig = extraRigs.get(aid);
  if (rig) return rig;
  const group = new THREE.Group();
  // Team colour when the recorded arena manifest knows this combatant's team (M5.5);
  // otherwise the legacy deterministic hash colour.
  const idx = [...aid].reduce((a, c) => a + c.charCodeAt(0), 0) % EXTRA_RIG_COLORS.length;
  const color = (combatCtx.teamOf && combatCtx.teamOf[aid]) ? teamColour(combatCtx.teamOf[aid]) : EXTRA_RIG_COLORS[idx];
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.17, 0.55, 4, 10),
                              new THREE.MeshStandardMaterial({ color, roughness: 0.8 }));
  body.rotation.x = Math.PI / 2;   // build along +Z like every other mesh (the world group is z-up)
  body.position.set(0, 0, -0.18);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.11, 10, 10),
                              new THREE.MeshStandardMaterial({ color, roughness: 0.8 }));
  head.position.set(0, 0, 0.42);
  // A standing leg column (0.42 tall, near the ground) — the old 0.2-tall slab sat at mid-torso
  // height and the whole statue read as floating (screenshots 2026-11-17).
  const legs = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.09, 0.42),
                              new THREE.MeshStandardMaterial({ color: 0x3a4453, roughness: 0.85 }));
  legs.position.set(0, 0, -0.58);
  group.add(body, head, legs);
  world.add(group);
  rig = { group };
  extraRigs.set(aid, rig);
  return rig;
}
function makeAgentRig() {
  const H = spec.agent_body.height;                       // 1.6 m collision box height
  const skin = new THREE.MeshStandardMaterial({ color: 0xb3a69a, roughness: 0.9 });
  const shirt = new THREE.MeshStandardMaterial({ color: 0x5aa9ff, roughness: 0.7, emissive: 0x0b2a4a, emissiveIntensity: 0.4 });
  const trousers = new THREE.MeshStandardMaterial({ color: 0x3a4453, roughness: 0.85 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x223044, roughness: 0.6 });

  function capsule(r, len, mat) {
    const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 4, 10), mat);
    m.rotation.x = Math.PI / 2;                           // build along +Z (capsules are Y-aligned)
    return m;
  }
  const rig = new THREE.Group();
  const add = (mesh, pos) => { mesh.position.set(...pos); rig.add(mesh); return mesh; };
  // Torso/pelvis are capsules so the figure reads as a body from any angle (not a slab edge-on)
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.165, 0.26, 4, 12), shirt);
  torso.rotation.x = Math.PI / 2; add(torso, [0, 0, 0.36]);
  add(new THREE.Mesh(new THREE.SphereGeometry(0.135, 14, 10), shirt), [0, 0, 0.58]);          // chest
  const pelvis = new THREE.Mesh(new THREE.CapsuleGeometry(0.125, 0.10, 4, 12), trousers);
  pelvis.rotation.x = Math.PI / 2; add(pelvis, [0, 0, 0.10]);
  // Head + face marker on the front (+X is the model's forward, like the export)
  const head = add(new THREE.Mesh(new THREE.SphereGeometry(0.115, 18, 14), skin), [0, 0, 0.94]);
  add(new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.10, 0.05), dark), [0.095, 0, 0.96]);        // visor
  head.userData.name = 'head';
  // Arms: pivot groups at the shoulders so they swing around the walk gait
  function arm(side) {
    const sx = 0.20 * side;
    const shoulder = new THREE.Group(); shoulder.position.set(sx, 0, 0.56); rig.add(shoulder);
    const upper = capsule(0.04, 0.20, shirt); upper.position.set(0, 0, -0.13); shoulder.add(upper);
    const elbow = new THREE.Group(); elbow.position.set(0, 0, -0.26); shoulder.add(elbow);
    const fore = capsule(0.032, 0.16, skin); fore.position.set(0, 0, -0.11); elbow.add(fore);
    return { shoulder, elbow };
  }
  function leg(side) {
    const sx = 0.09 * side;
    const hip = new THREE.Group(); hip.position.set(sx, 0, 0.08); rig.add(hip);
    const thigh = capsule(0.055, 0.26, trousers); thigh.position.set(0, 0, -0.17); hip.add(thigh);
    const knee = new THREE.Group(); knee.position.set(0, 0, -0.34); hip.add(knee);
    const shin = capsule(0.04, 0.32, trousers); shin.position.set(0, 0, -0.19); knee.add(shin);
    const foot = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.18, 0.05), dark);
    foot.position.set(0.03, 0, -0.40); knee.add(foot);
    return { hip, knee };
  }
  const armL = arm(1), armR = arm(-1), legL = leg(1), legR = leg(-1);
  rig.userData = { armL, armR, legL, legR, totalHeight: H };
  window.__agentRig = rig;    // inspection hook for automated QA
  window.__view3d = { camera, controls, world };   // QA hook: camera control for automated checks
  window.__replay = { poses: () => poses, show, cur: () => cur };   // QA hook: replay access
  world.add(rig);
  return rig;
}

// Rig animation: heading/gait are derived from the same frame stream the replay uses (deterministic
// per frame pair, so scrubbing is stable). Forward is +X in model space; legs swing around the Y axis.
let rigHeading = 0;
function animateRig(p, prev) {
  let speed = 0, heading = rigHeading;
  if (prev) {
    const dt = Math.max(0.01, p.t - prev.t);
    const vx = (p.agent[0] - prev.agent[0]) / dt, vy = (p.agent[1] - prev.agent[1]) / dt;
    speed = Math.hypot(vx, vy);
    if (speed > 0.05) {
      const target = Math.atan2(vy, vx);
      let d = target - rigHeading;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      heading = rigHeading + d * 0.4;
    }
  }
  rigHeading = heading;
  const vz = prev ? (p.agent[2] - prev.agent[2]) / Math.max(0.01, p.t - prev.t) : 0;
  const airborne = vz > 0.35 ? 1 : (vz < -0.35 ? -1 : 0);   // rising tuck / landing prep
  const swing = Math.min(0.55, speed * 0.4);
  const phase = p.t * (2.6 + speed * 2.4);
  const s = Math.sin(phase);
  const u = agentMesh.userData;
  u.legL.hip.rotation.y = airborne > 0 ? -0.5 : -swing * s;
  u.legR.hip.rotation.y = airborne > 0 ? 0.15 : swing * s;
  u.legL.knee.rotation.y = airborne > 0 ? 0.9 : Math.max(0, 0.55 * swing * -s);
  u.legR.knee.rotation.y = airborne > 0 ? 1.15 : Math.max(0, 0.55 * swing * s);
  u.armL.shoulder.rotation.y = airborne === 0 ? swing * s : 0.55 * airborne;
  u.armR.shoulder.rotation.y = p.held ? -0.5 : (airborne === 0 ? -swing * s : -0.55 * airborne);
  u.armL.elbow.rotation.y = airborne === 0 ? -0.2 * Math.abs(s) : 0.6;
  u.armR.elbow.rotation.y = p.held ? -1.2 : (airborne === 0 ? -0.2 * Math.abs(s) : 0.6);
  applyTilt(agentMesh, heading, fallenTilt(p.agent[2]));
}

// 2026-11-17 jank fixes: a fallen combatant (incapacitated/lost balance — the MuJoCo torso
// root drops from ~0.8 m to ~0.15 m) rendered as an upright figure with its legs buried under
// the floor plane. Tilt the model down onto its recorded torso once the recorded root height
// says "lying": a derived presentation of the recorded pose, no invented motion.
function fallenTilt(rootZ) { return Math.max(0, Math.min(1, (0.5 - rootZ) / 0.24)); }
const _TILT_Q = new THREE.Quaternion(), _YAW_Q = new THREE.Quaternion();
const _TILT_AXIS = new THREE.Vector3(0, 1, 0), _YAW_AXIS = new THREE.Vector3(0, 0, 1);
function applyTilt(obj3d, heading, tilt) {
  if (tilt <= 0) { obj3d.rotation.set(0, 0, heading || 0); return; }
  obj3d.rotation.set(0, 0, 0);                       // clear the euler before composing
  _TILT_Q.setFromAxisAngle(_TILT_AXIS, 1.5 * tilt);  // face-down: model +Z (up) -> model +X (front)
  _YAW_Q.setFromAxisAngle(_YAW_AXIS, heading || 0);  // topple about ITS facing direction
  obj3d.quaternion.copy(_YAW_Q).multiply(_TILT_Q);
}

const trails = new THREE.Group(); world.add(trails);

function resize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();
let _lastFrameTs = performance.now();
(function loop() {
  requestAnimationFrame(loop);
  const now = performance.now();
  fx.tick(Math.min(0.05, (now - _lastFrameTs) / 1000));   // impact FX decay in real time
  _lastFrameTs = now;
  controls.update(); renderer.render(scene, camera);
})();

// ------------------------------------------------------------------ replay state
// Frames are stored as deltas with periodic keyframes; we reconstruct full poses on load.
let frames = [], poses = [], cur = 0, playing = false, sessionId = null, sessionAgent = null,
    lastSeen = { a: 0, e: 0, m: 0 };
let flights = [];

function rebuild(from = 0) {
  let state = from > 0 ? structuredClone(poses[from - 1]) : { objects: {}, present: null };
  for (let i = from; i < frames.length; i++) {
    const f = frames[i];
    if (f.key) { state = { objects: {}, present: f.present }; }
    for (const [id, p] of Object.entries(f.objects)) state.objects[id] = p;
    if (f.present) for (const id of Object.keys(state.objects)) if (!f.present.includes(id)) delete state.objects[id];
    poses[i] = { t: f.t, agent: f.agent, agents: f.agents || null, held: f.held, objects: { ...state.objects }, present: state.present };
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
    if (combatCtx.pellets[id]) fx.trail(id, q, pelletColour(id), cur);   // the pellet's REAL recorded path
  }
  fx.dropTrailsExcept(new Set(Object.keys(p.objects).filter((id) => combatCtx.pellets[id])));
  fx.timeFilter(p.t);   // decals only from collisions the replay has already reached
  agentMesh.position.set(...p.agent);
  // M5 arena: every other combatant recorded by the shared world gets its own rig.
  for (const [, rig] of extraRigs) rig.group.visible = false;
  if (p.agents && sessionAgent) {
    for (const [aid, xyz] of Object.entries(p.agents)) {
      if (aid === sessionAgent) continue;
      const rig = ensureExtraRig(aid);
      rig.group.visible = true;
      rig.group.position.set(xyz[0], xyz[1], xyz[2]);
      applyTilt(rig.group, 0, fallenTilt(xyz[2]));   // other combatants fall with their recorded pose too
    }
  }
  animateRig(p, poses[cur - 1] || null);
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

let arenaMatches = [];
const modeName = (m) => ({ 'arena_solo': '1v1 duel', 'arena_coop': 'team match' }[m] || m);
// All interpolated values below are esc()-wrapped per this file's XSS policy (see header).
async function loadArenaMatches(retries = 3) {
  try {
    arenaMatches = (await api('/api/arena')).matches;
    $('#w-arena').innerHTML = '<option value="">—</option>' + arenaMatches.map((m, i) =>
      `<option value="${i}">${esc(m.label.replace(/^arena:/, ''))} · ${m.sessions.length} ${m.sessions.length === 1 ? 'combatant' : 'combatants'} · ${esc(m.status)}</option>`).join('');
  } catch (e) {
    // startup race with the page's other boot fetches: quietly retry, never block the tab
    if (retries > 0) setTimeout(() => loadArenaMatches(retries - 1), 2000);
  }
}
$('#w-arena').onchange = (e) => {
  const m = arenaMatches[e.target.value];
  if (!m) return;
  $('#w-session').innerHTML = m.sessions.map((x) =>
    `<option value="${+x.id}">#${+x.id} ${esc(x.agent_id)} · ${esc(modeName(x.mode))} (${esc(x.status)})</option>`).join('');
  if (m.sessions.length) openSession(m.sessions[0].id);
};
loadArenaMatches();

async function loadSessions() {
  const a = $('#w-agent').value;
  const s = (await api(`/api/sessions?agent_id=${a}`)).sessions;
  $('#w-session').innerHTML = s.map((x) => `<option value="${+x.id}">#${+x.id} ${esc(modeName(x.mode))}${x.task_id ? ':' + esc(x.task_id) : ''} s${+x.session_number} (${esc(x.status)})</option>`).join('');
  if (s.length) await openSession(s[0].id);
}
async function openSession(id) {
  sessionId = +id; frames = []; poses = []; flights = []; lastSeen = { a: 0, e: 0, m: 0 };
  rigHeading = 0;
  trails.clear();
  fx.reset(); buildCombatCtx(null, null);    // combat attribution rebuilds from this session's log
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
    api(`/api/sessions/${sid}/events?after=${lastSeen.e}&types=flight_end,object_appeared,object_lost,collision,pellet_launched,agent_damaged,weapon_reload,equipment_equipped`),
    api(`/api/sessions/${sid}/messages?after=${lastSeen.m}`),
    api(`/api/sessions/${sid}`),
  ]);
  if (sid !== sessionId) return;
  const atEnd = cur >= poses.length - 1;
  sessionAgent = detail.agent_id;
  if (combatCtx._sid !== sid) buildCombatCtx(detail.arena_context, sid);   // the recorded match manifest
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
      $('#w-exps').insertAdjacentHTML('afterbegin', `<div class="row"><b>${esc(r.experiment_id)}</b> ${esc(a.args.hypothesis)}<br>
        <span class="small muted">result:</span> ${esc(a.args.actual_result)} · <span class="small">${a.args.supported ? 'supported' : 'not supported'}</span>
        · evidence <span class="${v.counts_as_evidence ? 'ok' : 'bad'}">${esc(v.verdict)}</span></div>`);
    }
    if (a.tool === 'equip' && r.success && (r.object || r.equipped) && r.item) {
      // equip results carry the real equipped_dict (the 'equipped' object id + EQUIPMENT
      // registry id): markers become classifiable even for sessions recorded before the
      // manifest existed. The M5 factory names items '<team>_item_<k>' — a team-accent slug
      // (red/blue/...) is honest recorded provenance, so the session agent wears that team
      // for pellets/extra rigs too, when nothing else already named it.
      const oid = r.object || r.equipped;
      combatCtx.items[oid] = { item: r.item };
      const slug = String(oid).split('_item_')[0];
      if (slug) {
        itemTeam[oid] = Object.keys(combatCtx.teamOf).find((t) => _oid(t) === slug) || itemTeam[oid];
        const word = slug.split('_')[0];
        if (isAccentSlug(word) && !combatCtx.teamOf[sessionAgent]) combatCtx.teamOf[sessionAgent] = word;
      }
      lastSeen.equipLearned = true;
    }
    if (a.tool === 'look_around' && r.success) {
      // Recorded neighbours carry their team name (arena factory naming) — legacy sessions
      // have no manifest teams, so the opponent rig/pellets otherwise hash-paint.
      for (const n of (r.agents_nearby || [])) {
        if (n && n.id && n.team && !combatCtx.teamOf[n.id]) combatCtx.teamOf[n.id] = String(n.team);
      }
    }
  }
  if (evs.events.length) { lastSeen.e = evs.events[evs.events.length - 1].n; combatEvents(evs.events); drawFlights(evs.events); }
  reinstateCombatMeshes();
  for (const m of msgs.messages) {
    lastSeen.m = m.id;
    if (m.role === 'assistant') $('#w-msgs').insertAdjacentHTML('afterbegin', `<div class="row say">“${esc(sanitizeWords(m.content))}”</div>`);
  }
  const st = detail.final_state;
  const s = detail.summary;
  $('#w-status').textContent = `${detail.status}${detail.end_reason ? ' · ' + detail.end_reason : ''} · ${detail.action_count} actions · sim ${fmt(detail.sim_time, 1)} s · ${detail.adapter}/${detail.model}`;
  // Enable/disable the stop button based on whether this session is running
  const stopBtn = $('#w-stop');
  if (stopBtn) stopBtn.disabled = (detail.status !== 'running');
  const last = poses[poses.length - 1];
  $('#w-state').innerHTML = last ? `<div class="mono small">body at [${last.agent.map((v) => fmt(v)).join(', ')}] · holding ${esc(last.held || 'nothing')}</div>
    ${s ? `<div class="small">behaviour: ${Object.entries(s.behaviour).map(([k, v]) => `${k.replace(/_/g, ' ')} <b>${v}</b>`).join(' · ')}</div>` : ''}` : '';
  const recs = await api(`/api/sessions/${sid}/records?kind=goal`);
  const goals = {};
  for (const g of recs.records) goals[g.key] = { ...(goals[g.key] || {}), ...g.data };
  let sessionHtml = Object.entries(goals).map(([k, g]) => `<div class="row"><b>${esc(k)}</b> ${esc(g.title)} <span class="pill">${esc(g.status)}</span>
    ${g.kind && g.kind !== 'objective' ? `<span class="pill small">${esc(g.kind)}</span>` : ''}
    ${g.independently_verified === true ? '<span class="ok small">verified by physics</span>' : g.independently_verified === false ? '<span class="bad small">claim NOT verified</span>' : ''}</div>`).join('');
  // Also show the agent's STANDING open goals from memory, so a fresh session isn't shown as "none set"
  // when the agent carries unfinished objectives across sessions (audit 2026-11-13).
  let openHtml = '';
  try {
    const ag = $('#w-agent').value;
    if (ag) {
      const g = await api(`/api/agents/${encodeURIComponent(ag)}/goals`);
      openHtml = g.open_goals.map((x) => `<div class="row muted"><b>${esc(x.id)}</b> ${esc(x.title)} <span class="pill small">${esc(x.kind || 'objective')}</span>
        ${x.next_step ? `<br><span class="small">next: ${esc(x.next_step)}</span>` : ''}</div>`).join('');
      if (openHtml) openHtml = '<div class="small muted">standing objectives (from memory):</div>' + openHtml;
    }
  } catch (e) { /* non-fatal */ }
  $('#w-goals').innerHTML = (sessionHtml + openHtml) || '<span class="muted">none set</span>';
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

// Stop session button: sends a stop signal to the running session and any active run job
$('#w-stop')?.addEventListener('click', async () => {
  const btn = $('#w-stop');
  if (!sessionId || btn.disabled) return;
  btn.disabled = true;
  btn.textContent = 'Stopping…';
  try {
    await api(`/api/sessions/${sessionId}/stop`, { method: 'POST' });
    // Also stop any active run job that might be driving this session
    try {
      const active = await api('/api/run/active');
      for (const [jid, status] of Object.entries(active.jobs || {})) {
        if (status === 'running') await api(`/api/run/jobs/${jid}/stop`, { method: 'POST' });
      }
    } catch (e) { /* non-fatal */ }
    $('#w-status').textContent = 'Stop requested — session will end at the next safe point.';
    setTimeout(() => { btn.textContent = 'Stop'; }, 2000);
  } catch (err) {
    $('#w-status').textContent = String(err);
    btn.disabled = false;
    btn.textContent = 'Stop';
  }
});

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
  $('#m-stats').textContent = `${s.sessions_completed} sessions · ${s.discoveries_total} handbook entries `
    + `(${s.discoveries.confirmed} confirmed, ${s.discoveries.probable} probable, ${s.discoveries.hypothesis} hypotheses, `
    + `${s.discoveries.disproven} disproven, ${s.discoveries.unverified ?? 0} unverified, ${s.discoveries.challenged} challenged`
    + (s.records_removed ? `, ${s.records_removed} removed` : '')
    + `) · ${s.experiments_verified}/${s.experiments} experiments pass the evidence check (a pass is not a proof)`;
  // Rendering follows this file's established convention: every interpolated
  // value passes through esc() (innerHTML is only used to compose the layout;
  // no raw record text reaches the DOM unescaped).
  const disc = Object.fromEntries(memData.discoveries.map((d) => [d.id, d]));
  const exps = Object.fromEntries(memData.experiments.map((e) => [e.id, e]));
  $('#m-items').innerHTML = memData.items.map((i) => {
    const d = i.kind === 'discovery' ? disc[i.id] : null;
    const e = i.kind === 'experiment' ? exps[i.id] : null;
    let extra = '';
    if (d) {
      const ev = (d.evidence || []).map((id) => {
        const x = exps[id];
        if (!x) return id;
        const sup = x.supported === true ? 'supports' : x.supported === false ? 'contradicts' : 'read by agent only';
        return `${id} (${sup})`;
      });
      const ch = (d.history || []).filter((h) => h.kind === 'challenge')
        .map((h) => `challenge by ${esc(h.by ?? 'researcher')}: “${esc(h.reason || '')}”`);
      const last = [...(d.history || [])].reverse().find((h) => h.to && h.from !== undefined && h.kind !== 'challenge');
      const overlayNote = (d.status === 'challenged' || d.status === 'unverified') && d.evidence_status
        ? `<span class="pill ${esc(d.evidence_status)}" title="fresh evidence would grant this, but only a researcher resolves the overlay">evidence would grant: ${esc(d.evidence_status)}</span>` : '';
      extra = `<div class="small muted">origin: ${esc(d.agent_id || a)} in session ${esc(d.created_session)}${d.session_id ? ` (db #${esc(d.session_id)})` : ''}`
        + ` · requested: ${esc(d.requested_status || '—')}`
        + ` ${overlayNote}</div>`
        + `<div class="small muted">evidence: ${ev.map(esc).join(', ') || 'none'}</div>`
        + `<div class="small muted">status reason: ${esc(d.status_reason || '—')}</div>`
        + `<div class="small muted">challenge record: ${ch.length ? ch.join(' · ') : 'none'}</div>`
        + `<div class="small muted">last status change: ${last ? `${esc(last.from || 'new')} → <b>${esc(last.to)}</b> by ${esc(last.by || 'verification system')}${last.at ? ' at ' + new Date(last.at * 1000).toLocaleString() : ''}` : 'unchanged'}</div>`;
    } else if (e) {
      const o = e.outcome || {};
      const oc = e.verification || {};
      extra = `<div class="small muted">agent: ${esc(e.agent_id || a)} · session ${esc(e.session)}${e.session_id ? ` (db #${esc(e.session_id)})` : ''}`
        + `${e.discovery_id ? ` · tests <b>${esc(e.discovery_id)}</b>` : ''}</div>`
        + `<div class="small muted">evidence check: <b class="pill ${o.evidence_check === 'passed' ? 'confirmed' : 'disproven'}">${esc(o.evidence_check ?? (oc.verified ? 'passed' : 'failed'))}</b>`
        + ` (${esc(oc.verdict || '—')}) · scientific conclusion: <b>${esc(o.scientific ?? 'inconclusive')}</b> — these are separate things</div>`
        + `<div class="small muted">conditions: ${esc(e.conditions || '—')}</div>`
        + `<div class="small muted">observed: ${esc(String(e.actual_result || '—')).slice(0, 220)}</div>`
        + `<div class="small muted">cited actions: ${esc((e.action_refs || []).join(', ')) || '—'}${(e.action_refs || []).length ? ' (engine-checked)' : ''}</div>`;
    }
    return `<div class="row"><span class="mono">${i.id}</span>
    <span class="pill ${i.status || ''}">${i.kind}${i.status ? ': ' + i.status : ''}</span> ${esc((i.text || '').slice(0, 180))}
    ${i.flags && i.flags.challenged ? '<span class="warn small">challenged</span>' : ''}${extra}
    <div>${i.kind === 'discovery' ? `<button class="btn" data-op="challenge" data-id="${i.id}">challenge</button>
      <button class="btn" data-op="mark_unverified" data-id="${i.id}">mark unverified</button>
      ${i.status === 'challenged' ? `<button class="btn" data-op="resolve_challenge" data-id="${i.id}">resolve challenge</button>` : ''}
      ${i.status === 'unverified' ? `<button class="btn" data-op="reverify" data-id="${i.id}">re-verify</button>` : ''}` : ''}
      <button class="btn" data-op="remove" data-id="${i.id}">remove</button></div></div>`;
  }).join('');
  $('#m-items').querySelectorAll('button').forEach((b) => b.onclick = async () => {
    const reason = prompt(`Reason for ${b.dataset.op} ${b.dataset.id}?`);
    if (reason === null) return;
    try {
      const r = await api(`/api/agents/${a}/memory/${b.dataset.id}/${b.dataset.op}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reason }) });
      if (r && r.error) alert(`Researcher action failed: ${r.error}`);
    } catch (err) {
      // Backend rejections (unknown item bad status, ...) must be visible, not silent.
      alert(`Researcher action failed: ${err && err.message ? err.message : err}`);
    }
    loadMemory();
  });
  $('#m-audit').innerHTML = memData.audit.map((x) => `<div class="row small">${new Date(x.ts * 1000).toLocaleString()} · ${esc(x.actor)} · <b>${esc(x.op)}</b> ${esc(x.item_id)} ${esc(x.reason || '')}</div>`).join('') || '<span class="muted">no researcher interventions recorded in this agent\'s audit trail</span>';
  renderFile();
}
$('#m-agent').onchange = loadMemory;

// ------------------------------------------------------------------ analytics tab
async function loadAnalytics() {
  const a = $('#a-agent').value;
  const tl = (await api(`/api/analytics${a ? '?agent_id=' + a : ''}`)).timeline.filter((t) => t.mode === 'exploration');
  const max = Math.max(1, ...tl.map((t) => t.action_count || 0));
  const totalActions = tl.reduce((sum, t) => sum + (t.action_count || 0), 0);
  const evidencePassed = tl.reduce((sum, t) => sum + (t.experiments_ok || 0), 0);
  const evidenceTotal = tl.reduce((sum, t) => sum + (t.experiments_ok || 0) + (t.experiments_bad || 0), 0);
  const claimsVerified = tl.reduce((sum, t) => sum + (t.claims_verified || 0), 0);
  const claimsTotal = tl.reduce((sum, t) => sum + (t.claims_verified || 0) + (t.claims_rejected || 0), 0);
  const summary = [
    ['Exploration sessions', tl.length],
    ['Recorded actions', totalActions.toLocaleString()],
    ['Evidence checks passed', `${evidencePassed} / ${evidenceTotal}`],
    ['Claims verified', `${claimsVerified} / ${claimsTotal}`],
  ].map(([label, value]) => `<div class="metric-box"><div class="label">${label}</div><div class="value">${value}</div></div>`).join('');
  $('#a-chart').innerHTML = `<div class="metrics-grid">${summary}</div>` + (tl.length ? `
    <div class="chart-heading"><strong>Actions per exploration session</strong><span class="chart-meta">0–${max.toLocaleString()} actions · hover or focus a bar for detail</span></div>
    <div class="chart-stage">
      <div class="bars" role="group" aria-label="Actions per exploration session">${tl.map((t) => {
        const label = `#${t.id} · ${esc(t.agent_id)} · ${t.action_count || 0} actions`;
        return `<div role="img" tabindex="0" aria-label="${label}" data-session="${t.id}" data-agent="${esc(t.agent_id)}" data-actions="${t.action_count || 0}" style="height:${100 * (t.action_count || 0) / max}%"></div>`;
      }).join('')}</div>
      <div class="chart-tooltip" role="tooltip" hidden></div>
    </div>
    <div class="chart-axis"><span>Oldest session</span><span>Most recent</span></div>
    <div class="small muted chart-note">One bar per recorded exploration session. Full values and verification outcomes are in the table below.</div>`
    : '<div class="muted">No exploration sessions recorded for this selection yet.</div>');
  const chart = $('#a-chart .chart-stage');
  if (chart) {
    const tooltip = chart.querySelector('.chart-tooltip');
    const showTip = (bar) => {
      const value = document.createElement('strong');
      const label = document.createElement('span');
      value.textContent = `${bar.dataset.actions} actions`;
      label.textContent = `#${bar.dataset.session} · ${bar.dataset.agent}`;
      tooltip.replaceChildren(value, label);
      tooltip.hidden = false;
      const half = tooltip.offsetWidth / 2;
      const x = bar.offsetLeft + bar.offsetWidth / 2;
      tooltip.style.left = `${Math.max(half + 8, Math.min(chart.clientWidth - half - 8, x))}px`;
    };
    chart.querySelectorAll('.bars > div').forEach((bar) => {
      bar.addEventListener('pointerenter', () => showTip(bar));
      bar.addEventListener('focus', () => showTip(bar));
      bar.addEventListener('pointerleave', () => { tooltip.hidden = true; });
      bar.addEventListener('blur', () => { tooltip.hidden = true; });
    });
  }
  $('#a-table').innerHTML = `<table><tr><th>session</th><th>agent</th><th>#</th><th>actions</th><th>failed</th><th>sim s</th>
    <th>experiments ok/bad</th><th>claims ok/rejected</th><th>handbook</th><th>goals</th><th>pretend</th><th>repeat+vary</th><th>own sake</th><th>end</th></tr>
    ${tl.map((t) => `<tr><td>${t.id}</td><td>${esc(t.agent_id)}</td><td>${t.session_number}</td><td>${t.action_count}</td><td>${t.failed_actions ?? '–'}</td><td>${fmt(t.sim_time, 0)}</td>
    <td>${t.experiments_ok ?? '–'}/${t.experiments_bad ?? '–'}</td><td>${t.claims_verified ?? '–'}/${t.claims_rejected ?? '–'}</td>
    <td><span title="frozen snapshot taken when each session closed; open the Memory tab for current statuses">${t.discoveries ? Object.entries(t.discoveries).map(([k, v]) => v ? `${v} ${k}` : '').filter(Boolean).join(', ') : '–'}</span></td>
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
    <div class="small muted">memory quality = share of recorded experiments whose cited evidence passed the engine's integrity check (a check that confirms the actions were real and the numbers traced — not that the claim itself is true).</div>`;
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

// ------------------------------------------------------------------ neural networks tab
let nnColumns = null, nnJobPoll = null;
const nnDatasetRows = {};   // rows count per dataset seen last, to flash real refreshes on the tab

// Which registry the NN tab is pointed at. 'researcher' is the shared lab; 'agent' + an agent id is
// that agent's own networks (per-agent selection, audit 2026-11-13 NN lab stage).
const nnSel = { scope: 'researcher', owner: 'researcher' };

function nnQuery() {
  return `scope=${encodeURIComponent(nnSel.scope)}&owner=${encodeURIComponent(nnSel.owner)}`;
}

// Populate the owner dropdown once, and wire the scope/owner selectors to reload the tab.
(function wireNnScope() {
  const ownerSel = $('#nn-owner');
  if (ownerSel && !ownerSel.options.length) {
    ownerSel.innerHTML = agents.map((a) => `<option value="${esc(a)}">${esc(a)}</option>`).join('');
  }
  const scopeSel = $('#nn-scope');
  const wrap = $('#nn-owner-wrap');
  function sync() {
    nnSel.scope = scopeSel.value;
    if (nnSel.scope === 'agent') {
      wrap.style.display = '';
      nnSel.owner = ownerSel.value || agents[0] || 'researcher';
    } else {
      wrap.style.display = 'none';
      nnSel.owner = 'researcher';
    }
    loadNn();
  }
  scopeSel?.addEventListener('change', sync);
  ownerSel?.addEventListener('change', sync);
})();

async function loadDatasetQuality(datasetName) {
  const el = $('#nn-quality');
  if (!el) return;
  if (!datasetName) { el.textContent = 'select a dataset'; return; }
  $('#nn-quality-name').textContent = `(${datasetName})`;
  try {
    const q = await api(`/api/ml/datasets/${encodeURIComponent(datasetName)}/quality?${nnQuery()}`);
    const parts = [];
    parts.push(`<div><b>${q.rows}</b> rows · kind ${esc(q.kind || '–')} · ${q.stale ? '<span class="warn">stale</span>' : 'fresh'}`
      + (q.age_minutes != null ? ` · updated ${q.age_minutes < 1 ? 'just now' : Math.round(q.age_minutes) + ' min ago'}` : '') + '</div>');
    if (q.coverage && Object.keys(q.coverage).length) {
      parts.push('<div>log coverage: ' + Object.entries(q.coverage)
        .map(([t, n]) => `${esc(t)} <b>${n}</b> acts`).join(' · ') + '</div>');
    }
    if (q.balance) {
      const b = q.balance, tot = Math.max(1, b.class_1 + b.class_0);
      parts.push(`<div>balance (${esc(b.column)}): 1 → <b>${b.class_1}</b> (${Math.round(100 * b.class_1 / tot)}%) · 0 → <b>${b.class_0}</b> (${Math.round(100 * b.class_0 / tot)}%) `
        + `<span class="bar-mini"><span style="width:${Math.round(100 * b.class_1 / tot)}%"></span></span></div>`);
    }
    if (q.warnings && q.warnings.length) {
      parts.push(q.warnings.map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join(''));
    } else {
      parts.push('<div class="ok">no quality warnings</div>');
    }
    parts.push(`<div class="muted">lineage: sha <span class="mono">${esc((q.rows_sha256 || '–').slice(0, 12))}</span>${q.source ? ' · source ' + esc(q.source) : ''}</div>`);
    el.innerHTML = parts.join('');
  } catch (e) {
    el.innerHTML = `<span class="bad">${esc(String(e))}</span>`;
  }
}

// Leaderboard: persistent per-model matrix, click a column header to sort
let lbSort = { key: 'best_test_r2', dir: -1 };
async function loadLeaderboard() {
  const el = $('#nn-leaderboard');
  if (!el) return;
  try {
    const r = await api('/api/ml/leaderboard');
    const rows = (r.by_model || []).slice();
    if (!rows.length) { el.innerHTML = '<span class="muted">no profiles recorded yet — train a model or run a comparison</span>'; return; }
    rows.sort((a, b) => {
      const ka = a[lbSort.key], kb = b[lbSort.key];
      const va = ka === undefined || ka === null ? -Infinity : ka;
      const vb = kb === undefined || kb === null ? -Infinity : kb;
      return (va < vb ? -1 : va > vb ? 1 : 0) * lbSort.dir;
    });
    const cols = [
      { key: 'owner', label: 'owner' }, { key: 'name', label: 'model' },
      { key: 'best_test_r2', label: 'best R²' }, { key: 'best_test_rmse', label: 'best RMSE' },
      { key: 'beats_linear', label: 'beats linear' },
      { key: 'trend_r2', label: 'trend' }, { key: 'train_count', label: 'trains' },
      { key: 'compare_count', label: 'compares' }, { key: 'last_ts', label: 'last run' }];
    el.innerHTML = '<table class="lb"><tr>' + cols.map((c) =>
      `<th class="sortable" data-k="${c.key}">${esc(c.label)}${lbSort.key === c.key ? (lbSort.dir < 0 ? ' ▾' : ' ▴') : ''}</th>`).join('') + '</tr>' +
      rows.map((m) => {
        const trend = m.trend_r2 == null ? '–' : (m.trend_r2 > 0 ? `<span class="ok">▲${m.trend_r2}</span>` : m.trend_r2 < 0 ? `<span class="bad">▼${Math.abs(m.trend_r2)}</span>` : '=0');
        const last = m.last_ts ? new Date(m.last_ts * 1000).toLocaleString([], { year: '2-digit', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '–';
        const cmp = m.latest_compare ? `${m.latest_compare.actions_mean ?? '–'} acts · ${m.latest_compare.distinct_tools_mean ?? '–'} tools` : '–';
        return `<tr><td class="small">${esc(m.owner)}</td><td>${esc(m.name)}</td>` +
          `<td>${m.best_test_r2 == null ? '–' : m.best_test_r2.toFixed(3)}</td>` +
          `<td>${m.best_test_rmse == null ? '–' : m.best_test_rmse.toFixed(3)}</td>` +
          `<td>${m.beats_linear ? '<span class="ok">yes</span>' : '<span class="bad">no</span>'}</td>` +
          `<td>${trend}</td><td>${m.train_count}</td><td>${m.compare_count}<div class="muted small mono">${esc(cmp)}</div></td>` +
          `<td class="small muted">${esc(last)}</td></tr>`;
      }).join('') + '</table>';
    el.querySelectorAll('th.sortable').forEach((th) => th.onclick = () => {
      const k = th.dataset.k;
      const isText = (k === 'owner' || k === 'name');
      const dflt = isText ? 1 : -1;
      lbSort = (lbSort.key === k) ? { key: k, dir: -lbSort.dir } : { key: k, dir: dflt };
      loadLeaderboard();
    });
  } catch (e) {
    el.innerHTML = `<span class="bad">${esc(String(e))}</span>`;
  }
}

async function loadColumnsForDataset(datasetName) {
  try {
    const q = datasetName ? `?dataset=${encodeURIComponent(datasetName)}&${nnQuery()}` : `?${nnQuery()}`;
    const cols = await api(`/api/ml/columns${q}`);
    nnColumns = cols;
    const inputsDiv = $('#nn-inputs');
    if (inputsDiv) inputsDiv.innerHTML = Object.entries(cols.inputs).map(([k, v]) =>
      `<label title="${esc(v)}"><input type="checkbox" name="input" value="${esc(k)}">${esc(k)}</label>`
    ).join('');
    const targetSel = $('#nn-target');
    if (targetSel) targetSel.innerHTML = Object.entries(cols.targets).map(([k, v]) =>
      `<option value="${esc(k)}" title="${esc(v)}">${esc(k)}</option>`
    ).join('');
  } catch (e) { console.error('loadColumnsForDataset:', e); }
}

// ------------------------------------------------------------------ RL environment (M6-B B2)
// Read-only: the authoritative contract, curriculum stages, and bounded
// measured probe episodes. There is no trainer yet (M6-B B3) — nothing here
// renders a learning claim; scripted-gait probe legs are benchmarks.
const RL_COMPONENTS = ['progress', 'velocity_tracking', 'upright', 'alive',
                       'effort', 'smoothness', 'stability', 'fall'];
async function loadRl() {
  const status = $('#rl-status');
  try {
    const c = await api('/api/rl/contract');
    $('#rl-obs-head').textContent = `float32[${c.obs_dim}] · layout v${c.observation.layout_version}`;
    $('#rl-act-head').textContent = `14 normalized joints · v${c.action.layout_version}`;
    $('#rl-obs').innerHTML = `<table><tr><th>#</th><th>slice</th><th>field</th><th>unit</th><th>note</th></tr>${c.observation.sections.map((s, i) =>
      `<tr><td>${i + 1}</td><td>[${s.slice[0]},${s.slice[1]})</td><td><b>${esc(s.name)}</b></td><td>${esc(s.unit)}</td><td class="small">${esc(s.note)}</td></tr>`).join('')}</table>` +
      `<div class="small muted">${esc(c.observation.joint_order.join(', '))}</div>`;
    $('#rl-act').innerHTML =
      `<div class="small">encoding: ${esc(c.action.encoding)} · mapping: ${esc(c.action.mapping)}</div>` +
      `<table><tr><th>joint</th><th>range (rad)</th><th>kp / kd</th></tr>${c.action.joints.map((j) =>
        `<tr><td>${esc(j.name)}</td><td>${fmt(j.range_rad[0], 4)} … ${fmt(j.range_rad[1], 4)}</td><td>${fmt(j.kp, 1)} / ${fmt(j.kd, 2)}</td></tr>`).join('')}</table>` +
      `<div class="small">passive conventional PD (outside the action space): ${c.action.passive_conventional_pd.map((p) => `${esc(p.name)} → ${fmt(p.target_rad, 1)} (${fmt(p.kp, 1)}/${fmt(p.kd, 1)})`).join(', ')}</div>` +
      `<div class="small muted">decimation: policy every ${fmt(c.action.control_period_ms, 0)} ms (PD at ${fmt(c.action.pd_control_hz, 0)} Hz) · actuation: ${esc(c.action.actuation_mode)} · tagged "${esc(c.action.source_tag)}"</div>`;
    const curriculum = await api('/api/rl/curriculum');
    $('#rl-curriculum').innerHTML = `<table><tr><th>stage</th><th>name</th><th>target v*</th><th>upright assist</th><th>duration (decisions)</th><th>status</th></tr>${curriculum.stages.map((s) =>
      `<tr><td><b>${esc(s.id)}</b></td><td>${esc(s.name)}</td><td>${s.phase.target_velocity ? `[${s.phase.target_velocity.map((v) => fmt(v, 2)).join(', ')}]` : '–'}</td><td>${s.phase.upright_assist != null ? fmt(s.phase.upright_assist, 1) : '–'}</td><td>${s.duration_decisions || '–'}</td><td class="${s.env_ready ? 'ok' : ''}">${s.env_ready ? 'ready' : 'reserved (needs env-side hook)'}</td></tr>`).join('')}</table>` +
      `<div class="small muted">${esc(curriculum.note)}</div>`;
    status.textContent = `contract v${c.contract_version}`;
    status.className = 'muted small';
  } catch (e) {
    status.textContent = `contract unavailable: ${e.message}`;
    status.className = 'bad small';
  }
}

async function runRlProbe() {
  const st = $('#rl-probe-status');
  const btn = $('#rl-run');
  btn.disabled = true;
  st.textContent = 'running bounded episode…';
  try {
    const body = { leg: $('#rl-leg').value, seconds: Number($('#rl-seconds').value || 8), speed: 1.0 };
    const r = await api('/api/rl/probe', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const s = r.summary;
    $('#rl-probe-result').innerHTML =
      `<div class="small muted">${esc(r.note)}</div>` +
      `<table><tr><th>leg</th><th>steps</th><th>sim s</th><th>net displacement (m)</th><th>path (m)</th><th>progress paid (m)</th><th>upright frac</th><th>effort mean (W)</th><th>falls</th><th>faulted windows</th><th>reward total</th></tr>` +
      `<tr><td>${esc(s.leg)}</td><td>${s.steps}</td><td>${fmt(s.sim_time_s)}</td><td>${fmt(s.net_displacement_m, 4)}</td><td>${fmt(s.path_len_m)}</td><td>${fmt(s.heading_progress_rewarded_m)}</td><td>${fmt(s.upright_fraction, 4)}</td><td>${fmt(s.effort_mean_w, 1)}</td><td>${s.falls}</td><td>${s.faulted_windows}</td><td>${fmt(s.reward_total, 1)}</td></tr></table>` +
      `<div class="small">per-component sums: ${RL_COMPONENTS.map((k) => `${esc(k)} ${fmt((s.components || {})[k], 2)}`).join(' · ')}</div>`;
    st.textContent = 'measured via episode_summary() (engine measurements only)';
    st.className = 'muted small';
  } catch (e) {
    st.textContent = `probe failed: ${e.message}`;
    st.className = 'bad small';
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('rl-refresh').onclick = loadRl;
document.getElementById('rl-run').onclick = runRlProbe;

async function loadNn() {
  loadBehaviourKinds();
  // Status
  try {
    const st = await api('/api/ml/status');
    const parts = [];
    if (st.available) {
      parts.push(`PyTorch ${esc(st.version)}`);
      if (st.xpu) parts.push(`XPU: ${esc(st.xpu_device)}`);
      else if (st.cuda) parts.push(`CUDA: ${esc(st.cuda_device)}`);
      else parts.push('CPU only');
    } else {
      parts.push(`<span class="warn">${esc(st.reason)}</span>`);
    }
    $('#nn-status').innerHTML = parts.join(' · ');
  } catch (e) {
    $('#nn-status').textContent = String(e);
  }
  // Columns (for form) + quality — load dataset-specific data when a dataset is selected
  if (!window._nnColumnsWired) {
    window._nnColumnsWired = true;
    document.addEventListener('change', (e) => {
      if (e.target && e.target.id === 'nn-ds-select') loadColumnsForDataset(e.target.value);
      if (e.target && e.target.id === 'nn-ds-select') loadDatasetQuality(e.target.value);
    });
  }
  // Datasets (scoped to the selected registry) — with dataset-age + refresh flash
  try {
    const ds = (await api(`/api/ml/datasets?${nnQuery()}`)).datasets;
    const changed = [];
    $('#nn-datasets').innerHTML = ds.length ? `<table><tr><th>name</th><th>rows</th><th>source</th><th>updated</th></tr>
      ${ds.map((d) => {
        const prev = nnDatasetRows[d.name];
        const changedNow = prev !== undefined && prev !== d.rows;
        if (changedNow) changed.push(d.name);
        nnDatasetRows[d.name] = d.rows;
        const ageMin = d.updated_at ? Math.max(0, (Date.now() / 1000 - d.updated_at) / 60) : null;
        const ageTxt = ageMin === null ? '–' : (ageMin < 1 ? 'just now' : ageMin < 60 ? `${Math.round(ageMin)} min ago`
          : ageMin < 60 * 24 ? `${Math.round(ageMin / 60)} h ago` : `${Math.round(ageMin / 1440)} d ago`);
        return `<tr${changedNow ? ' class="ds-flash"' : ''}><td>${esc(d.name)}${changedNow ? ' <span class="ok small">updated</span>' : ''}</td><td>${d.rows}</td><td>${esc(d.source)}</td><td class="small muted" title="dataset age">${esc(ageTxt)}</td></tr>`;
      }).join('')}</table>` : '<span class="muted">no datasets</span>';
    if (changed.length) $('#nn-status').insertAdjacentHTML('beforeend',
      ` · <span class="ok">datasets refreshed: ${changed.map(esc).join(', ')}</span>`);
    const dsSel2 = $('#nn-ds-select');
    const prevVal = dsSel2.value;
    dsSel2.innerHTML = ds.map((d) => `<option value="${esc(d.name)}">${esc(d.name)} (${d.rows} rows)</option>`).join('');
    if (prevVal && ds.some((d) => d.name === prevVal)) dsSel2.value = prevVal;
    // Load columns + quality for the currently selected dataset (or defaults)
    await loadColumnsForDataset(dsSel2.value);
    loadDatasetQuality(dsSel2.value);
  } catch (e) {
    $('#nn-datasets').innerHTML = `<span class="bad">${esc(String(e))}</span>`;
  }
  loadLeaderboard();
  // Models (the /api/ml/models scan returns every scope; filter to the selected registry)
  try {
    let models = (await api('/api/ml/models')).models;
    if (nnSel.scope === 'agent') models = models.filter((m) => m.owner === nnSel.owner);
    else models = models.filter((m) => (m.scope || 'researcher') === 'researcher');
    if (!models.length) {
      $('#nn-models').innerHTML = '<span class="muted">no models in this registry</span>';
    } else {
      $('#nn-models').innerHTML = `<table><tr><th>scope</th><th>owner</th><th>name</th><th>v</th><th>target</th><th>inputs</th><th>params</th><th>test RMSE</th><th>linear RMSE</th><th>beats linear</th><th>closed loop</th></tr>
        ${models.map((m) => `<tr class="model-row" data-scope="${esc(m.scope || 'researcher')}" data-owner="${esc(m.owner)}" data-name="${esc(m.name)}" data-version="${m.version}">
          <td class="pill">${esc(m.scope || 'researcher')}</td><td>${esc(m.owner)}</td><td>${esc(m.name)}</td><td>${m.version}</td>
          <td>${esc(m.target)}</td><td class="small">${(m.inputs || []).map(esc).join(', ')}</td>
          <td>${m.parameters}</td><td>${fmt(m.test_rmse, 4)}</td><td>${fmt(m.linear_rmse, 4)}</td>
          <td class="${m.beats_linear ? 'ok' : 'bad'}">${m.beats_linear ? 'yes' : 'no'}</td>
          <td>${m.closed_loop ? `${m.closed_loop.controllers?.network?.hit_rate * 100 || '–'}%` : '–'}</td></tr>`).join('')}</table>`;
      document.querySelectorAll('.model-row').forEach((row) => {
        row.onclick = () => loadModelDetail(row.dataset.scope, row.dataset.owner, row.dataset.name, +row.dataset.version);
      });
    }
  } catch (e) {
    $('#nn-models').innerHTML = `<span class="bad">${esc(String(e))}</span>`;
  }
}

async function loadModelDetail(scope, owner, name, version) {
  try {
    const m = await api(`/api/ml/models/${scope}/${owner}/${name}?version=${version}`);
    $('#nn-detail-card').style.display = 'block';
    // Loss curve SVG
    const hist = m.history || [];
    const maxLoss = Math.max(...hist.map((h) => Math.max(h.train_loss, h.val_loss))) || 1;
    const w = 400, h = 150, pad = 30;
    const xScale = (i) => pad + (i / (hist.length - 1 || 1)) * (w - 2 * pad);
    const yScale = (v) => h - pad - (v / maxLoss) * (h - 2 * pad);
    const trainLine = hist.map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(i)},${yScale(p.train_loss)}`).join(' ');
    const valLine = hist.map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(i)},${yScale(p.val_loss)}`).join(' ');
    const lossCurveSvg = `<svg viewBox="0 0 ${w} ${h}" class="loss-curve">
      <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#3a3a3a"/>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" stroke="#3a3a3a"/>
      <path d="${trainLine}" fill="none" stroke="#5aa9ff" stroke-width="1.5"/>
      <path d="${valLine}" fill="none" stroke="#e8b04b" stroke-width="1.5"/>
      <text x="${w - pad}" y="${h - 5}" fill="#8b93a1" font-size="9" text-anchor="end">epoch ${hist.length}</text>
      <text x="${pad}" y="${pad - 5}" fill="#8b93a1" font-size="9">loss ${fmt(maxLoss, 3)}</text>
      <text x="${w - 80}" y="15" fill="#5aa9ff" font-size="9">train</text>
      <text x="${w - 40}" y="15" fill="#e8b04b" font-size="9">val</text>
    </svg>`;
    // Metrics
    const mt = m.metrics || {};
    const metricsHtml = `<div class="metrics-grid">
      <div class="metric-box"><div class="label">Test RMSE</div><div class="value">${fmt(mt.test?.rmse, 4)}</div></div>
      <div class="metric-box"><div class="label">Test R²</div><div class="value">${fmt(mt.test?.r2, 4)}</div></div>
      <div class="metric-box"><div class="label">Test MAE</div><div class="value">${fmt(mt.test?.mae, 4)}</div></div>
      <div class="metric-box"><div class="label">Linear RMSE</div><div class="value">${fmt(mt.baseline_linear?.rmse, 4)}</div></div>
      <div class="metric-box"><div class="label">Mean RMSE</div><div class="value">${fmt(mt.baseline_mean?.rmse, 4)}</div></div>
      <div class="metric-box"><div class="label">Beats linear</div><div class="value ${mt.beats_linear_baseline ? 'ok' : 'bad'}">${mt.beats_linear_baseline ? 'Yes' : 'No'}</div></div>
    </div>`;
    // Checks
    const ch = m.checks || {};
    const checksHtml = `<div class="small muted">
      gradients finite: <b class="${ch.gradients_finite ? 'ok' : 'bad'}">${ch.gradients_finite}</b> ·
      weight change: ${fmt(ch.weight_change_l2, 4)} · first train loss: ${fmt(ch.first_train_loss, 4)} · best val loss: ${fmt(ch.best_val_loss, 4)}
    </div>`;
    // Test examples
    const exHtml = (m.test_examples || []).length ? `<table class="small"><tr><th>inputs</th><th>actual</th><th>predicted</th><th>error</th></tr>
      ${m.test_examples.map((e) => `<tr><td>${Object.entries(e.inputs).map(([k, v]) => `${k}=${fmt(v, 2)}`).join(', ')}</td>
        <td>${fmt(e.actual, 3)}</td><td>${fmt(e.predicted, 3)}</td><td>${fmt(Math.abs(e.actual - e.predicted), 3)}</td></tr>`).join('')}</table>` : '';
    // Closed-loop
    let clHtml = '';
    if (m.closed_loop) {
      const cl = m.closed_loop;
      const ctrl = cl.controllers || {};
      clHtml = `<h4>Closed-loop evaluation</h4>
        <div class="small muted">elevation ${cl.elevation_deg}° · tolerance ${cl.tolerance} m · ${cl.targets?.length || 0} targets</div>
        <table class="small"><tr><th>controller</th><th>hits</th><th>hit rate</th><th>mean error</th></tr>
        ${Object.entries(ctrl).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.hits}/${v.trials}</td><td>${fmt(v.hit_rate * 100, 1)}%</td><td>${fmt(v.mean_abs_error, 3)}</td></tr>`).join('')}</table>`;
    }
    // Evaluate buttons. 'Run as controller' puts the network in the body and runs a benchmark task,
    // judged by the task's own verifier (the 'test my network in the world' button, audit 2026-11-13).
    const evalHtml = `<div style="margin-top:10px;">
      <button class="btn" id="nn-eval-fresh">Fresh engine test</button>
      <button class="btn" id="nn-eval-cl">Closed-loop test</button>
      <button class="btn" id="nn-run-ctrl">Run as controller</button>
      <select id="nn-ctrl-task" title="benchmark task"><option value="throw_height">throw_height</option></select>
      <button class="btn" id="nn-clv-btn" title="same task, same conditions: this network AS the brain vs the scripted reference policy, both judged by the task's own verifier">vs scripted baseline</button>
      <div id="nn-clv-out" class="small" style="margin-top:6px"></div>
    </div>`;

    $('#nn-detail').innerHTML = `
      <div class="small muted">v${m.version} · ${esc(m.spec?.name)} → ${esc(m.spec?.target)} · ${m.parameters} params · ${m.epochs_run} epochs · ${fmt(m.seconds, 1)}s on ${esc(m.device)}</div>
      <div class="small">inputs: ${(m.spec?.inputs || []).map(esc).join(', ')} · hidden: [${(m.spec?.hidden || []).join(', ')}]</div>
      ${lossCurveSvg}
      ${metricsHtml}
      ${checksHtml}
      ${exHtml ? '<h4>Test examples</h4>' + exHtml : ''}
      ${clHtml}
      ${evalHtml}
    `;
    // Eval button handlers
    $('#nn-eval-fresh').onclick = async () => {
      try {
        const r = await api('/api/ml/evaluate', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ owner, name, version, kind: 'fresh', n: 100 })
        });
        pollJob(r.job_id);
      } catch (e) { alert(String(e)); }
    };
    $('#nn-eval-cl').onclick = async () => {
      try {
        const r = await api('/api/ml/evaluate', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ owner, name, version, kind: 'closed_loop' })
        });
        pollJob(r.job_id);
      } catch (e) { alert(String(e)); }
    };
    $('#nn-run-ctrl').onclick = async () => {
      try {
        const task_id = $('#nn-ctrl-task').value;
        const r = await api('/api/ml/controller', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ owner, name, version, task_id })
        });
        pollJob(r.job_id);
      } catch (e) { alert(String(e)); }
    };
    // Closed-loop vs scripted: the network AS the brain vs the scripted reference, same task + verifier.
    $('#nn-clv-btn').onclick = async () => {
      const out = $('#nn-clv-out');
      try {
        out.textContent = 'Running both sides (neural first, then scripted baseline)…';
        const task_id = $('#nn-ctrl-task').value;
        const r = await api('/api/ml/closed-loop-compare', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ owner, name, version, task_id })
        });
        const started = Date.now();
        const timer = setInterval(async () => {
          try {
            const j = await api(`/api/ml/jobs/${r.job_id}`);
            if (j.status === 'done') {
              clearInterval(timer);
              const s = j.result_summary, v = s.verdict;
              out.innerHTML = `<table><tr><th></th><th>network</th><th>scripted</th></tr>
                <tr><td>completed</td><td>${s.model.completed ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>'}</td><td>${s.baseline.completed ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>'}</td></tr>
                <tr><td>score</td><td><b>${s.model.score ?? '–'}</b></td><td><b>${s.baseline.score ?? '–'}</b></td></tr>
                <tr><td>accuracy error</td><td>${s.model.accuracy ?? '–'}</td><td>${s.baseline.accuracy ?? '–'}</td></tr>
                <tr><td>actions</td><td>${s.model.actions ?? '–'}</td><td>${s.baseline.actions ?? '–'}</td></tr>
                <tr><td>failed</td><td>${s.model.failed_actions ?? '–'}</td><td>${s.baseline.failed_actions ?? '–'}</td></tr>
                </table><div>${v.beats_baseline === true ? '<span class="ok">✔ The network BEAT the scripted baseline.' : v.beats_baseline === false ? '<span class="warn">✖ The scripted baseline scored higher.</span>' : '<span class="muted">Neither side produced a score to compare.</span>'}${v.score_delta != null ? ` (Δ ${v.score_delta})` : ''}</div>`;
            } else if (j.status === 'error') {
              clearInterval(timer);
              out.innerHTML = `<span class="bad">${esc(j.error || 'error')}</span>`;
            } else {
              out.textContent = `Running (${Math.round((Date.now() - started) / 1000)} s)…`;
            }
          } catch (e2) { clearInterval(timer); out.innerHTML = `<span class="bad">${esc(String(e2))}</span>`; }
        }, 1200);
      } catch (e) { out.innerHTML = `<span class="bad">${esc(String(e))}</span>`; }
    };
  } catch (e) {
    $('#nn-detail-card').style.display = 'block';
    $('#nn-detail').innerHTML = `<span class="bad">${esc(String(e))}</span>`;
  }
}

function pollJob(jobId) {
  $('#nn-job-card').style.display = 'block';
  $('#nn-job').innerHTML = `<span class="muted">Job ${esc(jobId)}: polling...</span>`;
  if (nnJobPoll) clearInterval(nnJobPoll);
  nnJobPoll = setInterval(async () => {
    try {
      const j = await api(`/api/ml/jobs/${jobId}`);
      if (j.status === 'done') {
        clearInterval(nnJobPoll);
        nnJobPoll = null;
        $('#nn-job').innerHTML = `<span class="ok">Job ${esc(jobId)}: done</span><pre class="small">${esc(JSON.stringify(j.result_summary, null, 2))}</pre>`;
        loadNn();  // refresh models
      } else if (j.status === 'error') {
        clearInterval(nnJobPoll);
        nnJobPoll = null;
        $('#nn-job').innerHTML = `<span class="bad">Job ${esc(jobId)}: error</span><pre class="small">${esc(j.error)}</pre>`;
      } else {
        const p = j.progress;
        $('#nn-job').innerHTML = `<span class="muted">Job ${esc(jobId)}: ${esc(j.status)}`
          + (p && p.done != null ? ` · ${p.done}/${p.of} rows` : '') + '</span>';
      }
    } catch (e) {
      clearInterval(nnJobPoll);
      nnJobPoll = null;
      $('#nn-job').innerHTML = `<span class="bad">${esc(String(e))}</span>`;
    }
  }, 1000);
}

// Train form handler
document.getElementById('nn-train-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = $('#nn-name').value.trim();
  const dataset = $('#nn-ds-select').value;
  const target = $('#nn-target').value;
  const inputs = [...document.querySelectorAll('#nn-inputs input:checked')].map((i) => i.value);
  const hidden = $('#nn-hidden').value.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => n > 0);
  const epochs = parseInt($('#nn-epochs').value, 10) || 300;
  const device = $('#nn-device').value;
  if (!name || !inputs.length) {
    $('#nn-train-status').innerHTML = '<span class="bad">Name and at least one input required</span>';
    return;
  }
  try {
    $('#nn-train-status').textContent = 'Submitting...';
    const r = await api('/api/ml/train', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset,
        spec: { name, inputs, target, hidden },
        train: { epochs, device },
        scope: nnSel.scope,
        owner: nnSel.owner
      })
    });
    $('#nn-train-status').textContent = `Job ${r.job_id} started`;
    pollJob(r.job_id);
  } catch (err) {
    $('#nn-train-status').innerHTML = `<span class="bad">${esc(String(err))}</span>`;
  }
});

// Generate-dataset form handler: builds training data by running the real world, into the selected registry.
document.getElementById('nn-gen-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const kind = $('#nn-gen-kind')?.value || 'throw';
  const name = $('#nn-gen-name').value.trim();
  const n = parseInt($('#nn-gen-n').value, 10) || 1500;
  const seed = parseInt($('#nn-gen-seed').value, 10) || 7;
  if (!name) {
    $('#nn-gen-status').innerHTML = '<span class="bad">Name required</span>';
    return;
  }
  try {
    $('#nn-gen-status').textContent = 'Submitting…';
    const r = await api('/api/ml/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, name, n, seed, scope: nnSel.scope, owner: nnSel.owner })
    });
    $('#nn-gen-status').textContent = `Job ${r.job_id} started`;
    pollJob(r.job_id);
  } catch (err) {
    $('#nn-gen-status').innerHTML = `<span class="bad">${esc(String(err))}</span>`;
  }
});

// ------------------------------------------------------------------ Lab: run, models, compare, ML switch, provenance
let labInit = false, labAdapters = [];
const MODE_DEFAULTS = {                                // only when /api/run/modes is unavailable
  modes: [], note: 'run-mode registry unavailable — restart the dashboard server',
};
function labMode() { return $('#lab-run-mode').value || 'exploration'; }
function labSelect(sel, values, preselect) {
  $(sel).innerHTML = values.map((v) =>
    `<option value="${esc(v)}" ${v === preselect ? 'selected' : ''}>${esc(v)}</option>`).join('');
}
async function populateLabAdapters() {
  try {
    const ad = await api('/api/adapters');
    labAdapters = ad.adapters.map((a) => a.name);
    const oc = ad.adapters.filter((a) => a.type === 'openai_compat');
    $('#lab-adapter').innerHTML = oc.map((a) => `<option value="${esc(a.name)}">${esc(a.name)}</option>`).join('');
  } catch (e) { labAdapters = ['scripted']; }
}
function combatantRow(cls, defaults) {
  const adapters = ['scripted', ...labAdapters.filter((a) => a !== 'scripted')];
  return `<div class="row small"><input class="${cls} cid" value="${esc(defaults.id)}" placeholder="combatant id" size="14"> `
    + `<select class="${cls} cad">${adapters.map((a) => `<option value="${esc(a)}" ${a === (defaults.adapter || 'scripted') ? 'selected' : ''}>${esc(a)}</option>`).join('')}</select> `
    + `<input class="${cls} cmodel" placeholder="model (provider default)" size="18"></div>`;
}
function buildArenaTeamsUi() {
  const host = $('#lab-arena-teams');
  if (!host) return;
  const solo = labMode() === 'arena_solo';
  const counts = { a: +($('#lab-arena-a-count')?.value || 1), b: +($('#lab-arena-b-count')?.value || 1) };
  const names = { a: $('#lab-arena-a-name')?.value || 'Red', b: $('#lab-arena-b-name')?.value || 'Blue' };
  // preserve typed rows while keeping the count in sync
  const read = (letter) => [...host.querySelectorAll(`#lab-arena-${letter}-rows .cid`)].map((i) => ({ id: i.value,
    adapter: i.closest('.row')?.querySelector('.cad')?.value, model: i.closest('.row')?.querySelector('.cmodel')?.value }));
  const prevA = read('a'), prevB = read('b');
  const defaultsA = []; const defaultsB = [];
  for (let i = 0; i < counts.a; i++) defaultsA.push(prevA[i] || { id: i === 0 ? 'agent_arena_r1' : `agent_arena_r${i + 1}` });
  for (let i = 0; i < counts.b; i++) defaultsB.push(prevB[i] || { id: `agent_arena_b${i + 1}` });
  host.innerHTML = `
    <div class="form-row team-a">
      <b>Team A</b>
      <label>name <input id="lab-arena-a-name" value="${esc(names.a)}" size="8"></label>
      <label>combatants <input type="number" id="lab-arena-a-count" value="${counts.a}" min="1" max="4" ${solo ? 'disabled' : ''}></label>
    </div>
    <div id="lab-arena-a-rows">${defaultsA.map((d) => combatantRow('team-a-c', d)).join('')}</div>
    <div class="form-row">
      <label>Team A equipment <select id="lab-arena-a-equip" multiple size="3">
        ${equipList.map((e) => `<option value="${esc(e.id)}" data-name="${esc(e.name)}" ${e.id === 'paintball_rifle' ? 'selected' : ''}>${esc(e.name || e.id)}${e.combat_class ? ' (' + esc(e.combat_class) + ')' : ''}</option>`).join('')}
      </select></label>
    </div>
    <div class="form-row team-b">
      <b>Team B</b>
      <label>name <input id="lab-arena-b-name" value="${esc(names.b)}" size="8"></label>
      <label>combatants <input type="number" id="lab-arena-b-count" value="${counts.b}" min="1" max="4" ${solo ? 'disabled' : ''}></label>
    </div>
    <div id="lab-arena-b-rows">${defaultsB.map((d) => combatantRow('team-b-c', d)).join('')}</div>
    <div class="form-row">
      <label>Team B equipment <select id="lab-arena-b-equip" multiple size="3">
        ${equipList.map((e) => `<option value="${esc(e.id)}" ${e.id === 'paintball_pistol' ? 'selected' : ''}>${esc(e.name || e.id)}${e.combat_class ? ' (' + esc(e.combat_class) + ')' : ''}</option>`).join('')}
      </select></label>
    </div>`;
  ['a', 'b'].forEach((letter) => {
    const c = $(`#lab-arena-${letter}-count`);
    if (c) c.onchange = buildArenaTeamsUi;             // re-render rows with preserved values
    const nm = $(`#lab-arena-${letter}-name`);
    if (nm) nm.oninput = null;                         // typed team names persist via the rebuild read
  });
}
function refreshLabModeUi() {
  if (!$('#lab-run-mode')) return;
  const m = labMode();
  const meta = runModes.find((x) => x.id === m) || MODE_DEFAULTS.modes.find((x) => x.id === m);
  $('#lab-run-mode-note').textContent = meta ? meta.note : (runModes.length ? '' : MODE_DEFAULTS.note);
  const solo = m === 'arena_solo', coop = m === 'arena_coop', bench = m === 'benchmark', explore = m === 'exploration';
  $('#lab-run-explore-fields').hidden = !explore;
  $('#lab-run-explore-extra').hidden = !explore;
  $('#lab-run-bench-fields').hidden = !bench;
  $('#lab-run-arena-fields').hidden = !(solo || coop);
  if (solo || coop) {
    buildArenaTeamsUi();
  }
}
function populateBenchTasks(tasks) {
  // tasks = /api/benchmarks/tasks payload (task ids are the backend's real TASKS keys)
  $('#lab-run-bench-task').innerHTML = (tasks || []).map((t) =>
    `<option value="${esc(t.id)}">${esc(t.name || t.id)}</option>`).join('');
}
async function initRunModes() {
  if (!$('#lab-run-mode')) return;
  if (runModes.length) {
    $('#lab-run-mode').innerHTML = runModes.map((m) =>
      `<option value="${esc(m.id)}">${esc(m.label || m.id)}</option>`).join('');
    const benchAgents = (await api('/api/agents')).agents.map((a) => a.agent_id);
    labSelect('#lab-run-bench-agent', benchAgents, benchAgents[0]);
    try {
      const bt = (await api('/api/benchmarks/tasks')).tasks || [];
      populateBenchTasks(bt.sort((a, b) => String(a.id).localeCompare(String(b.id))));
    } catch (e) { populateBenchTasks([]); }
  } else {
    $('#lab-run-mode').innerHTML = `<option value="">${esc(MODE_DEFAULTS.note)}</option>`;
  }
  $('#lab-run-mode').onchange = refreshLabModeUi;
  refreshLabModeUi();
}
async function loadLab() {
  if (!labInit) {
    labInit = true;
    // provider + agent dropdowns (the run-mode/equipment registry is loaded ONCE at page boot)
    try {
      await populateLabAdapters();
      const ags = await api('/api/agents');
      const agOpts = ags.agents.map((a) => `<option value="${esc(a.agent_id)}">${esc(a.agent_id)}</option>`).join('');
      $('#lab-run-agent').innerHTML = agOpts;
    } catch (e) { /* non-fatal */ }
    await initRunModes();
    labFillBenchAdapter();
  }
  loadMlConfig();
}
function labFillBenchAdapter() {
  const adapters = ['scripted', ...labAdapters.filter((a) => a !== 'scripted')];
  const el = $('#lab-run-bench-adapter');
  if (el) el.innerHTML = adapters.map((a) => `<option value="${esc(a)}" ${a === 'scripted' ? 'selected' : ''}>${esc(a)}</option>`).join('');
}
function arenaTeamPayload(letter) {
  const name = ($(`#lab-arena-${letter}-name`)?.value || '').trim();
  const rows = [...document.querySelectorAll(`#lab-arena-${letter}-rows .row`)].map((r) => ({
    id: (r.querySelector('.cid')?.value || '').trim(),
    adapter: (r.querySelector('.cad')?.value || 'scripted').trim(),
    model: (r.querySelector('.cmodel')?.value || '').trim() || null,
  }));
  const equipment = [...($(`#lab-arena-${letter}-equip`)?.selectedOptions || [])].map((o) => o.value);
  return { name, agents: rows, equipment };
}
function validateArenaForm() {
  const solo = labMode() === 'arena_solo';
  const teams = [arenaTeamPayload('a'), arenaTeamPayload('b')];
  for (const [i, t] of teams.entries()) {
    if (!t.name) return `Team ${i === 0 ? 'A' : 'B'} needs a name`;
    for (const c of t.agents) {
      if (!/^[a-z0-9][a-z0-9_\-]{0,39}$/i.test(c.id)) return `Team ${t.name}: combatant id '${c.id}' is invalid (letters/digits/_/-)`;
    }
  }
  if (teams[0].name.toLowerCase() === teams[1].name.toLowerCase()) return 'Team names must differ';
  const dup = teams[0].agents.map((c) => c.id).filter((id) => teams[1].agents.some((c) => c.id === id));
  if (dup.length) return `Combatant ids must be unique across teams (duplicate: ${dup[0]})`;
  const dur = +$('#lab-arena-duration').value;
  if (!(dur >= 2 && dur <= 600)) return 'Duration must be 2-600 s';
  const target = +$('#lab-arena-target').value;
  if (!(target >= 1 && target <= 999)) return 'Score target must be 1-999';
  const pellets = +$('#lab-arena-pellets').value;
  if (!(pellets >= 1 && pellets <= 96)) return 'Pellet pool must be 1-96';
  if (solo && (teams[0].agents.length !== 1 || teams[1].agents.length !== 1)) return 'arena duel (1v1) needs exactly one combatant per team';
  return null;
}


async function loadMlConfig() {
  try {
    const c = await api('/api/ml/config');
    $('#lab-ml-global').checked = c.enabled;
    $('#lab-ml-status').textContent = c.enabled ? 'nn_* tools offered to agents' : 'nn_* tools hidden';
    $('#lab-ml-agents').innerHTML = Object.entries(c.resolved).map(([a, on]) =>
      `<label class="row small"><input type="checkbox" class="lab-ml-agent" data-agent="${esc(a)}" ${on ? 'checked' : ''}> ${esc(a)}</label>`).join('');
    document.querySelectorAll('.lab-ml-agent').forEach((cb) => cb.onchange = async () => {
      await api('/api/ml/config', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent: cb.dataset.agent, agent_enabled: cb.checked }) });
      loadMlConfig();
    });
  } catch (e) { $('#lab-ml-status').innerHTML = `<span class="bad">${esc(String(e))}</span>`; }
}
$('#lab-ml-global')?.addEventListener('change', async (e) => {
  await api('/api/ml/config', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: e.target.checked }) });
  loadMlConfig();
});

$('#lab-list-models')?.addEventListener('click', async () => {
  const adapter = $('#lab-adapter').value;
  $('#lab-models-status').textContent = 'Querying…';
  try {
    const r = await api(`/api/models?adapter=${encodeURIComponent(adapter)}`);
    $('#lab-models-status').textContent = `${r.models.length} model(s) at ${esc(r.base_url)}`;
    $('#lab-models').innerHTML = r.models.length
      ? `<table><tr><th>id</th><th>state</th><th>ctx</th><th></th></tr>${r.models.map((m) =>
          `<tr><td class="mono small">${esc(m.id)}</td><td>${esc(m.state || '')}</td><td>${m.context ?? '–'}</td>` +
          `<td><button class="btn small lab-pick" data-id="${esc(m.id)}">use</button></td></tr>`).join('')}</table>`
      : '<span class="muted">No models (server offline, no model loaded, or auth needed).</span>';
    document.querySelectorAll('.lab-pick').forEach((b) => b.onclick = () => {
      $('#lab-run-model').value = b.dataset.id;
      if (!$('#lab-cmp-models').value.includes(b.dataset.id)) {
        $('#lab-cmp-models').value = ($('#lab-cmp-models').value ? $('#lab-cmp-models').value + ', ' : '') + b.dataset.id;
      }
    });
  } catch (e) { $('#lab-models-status').innerHTML = `<span class="bad">${esc(String(e))}</span>`; }
});

async function pollRunJob(jobId, statusEl, onDone) {
  const tick = async () => {
    try {
      const j = await api(`/api/run/jobs/${jobId}`);
      const prog = (j.progress || []).reduce((n, p) => n + (Array.isArray(p.sessions) ? p.sessions.length : 1), 0);
      statusEl.textContent = `Job ${jobId}: ${j.status} · ${prog} session(s) so far`;
      if (j.status === 'done') { onDone(j); return; }
      if (j.status === 'error') { statusEl.innerHTML = `<span class="bad">${esc(j.error || 'error')}</span>`; return; }
      setTimeout(tick, 2000);
    } catch (e) { statusEl.innerHTML = `<span class="bad">${esc(String(e))}</span>`; }
  };
  tick();
}

let labRunJob = null; // id of the job launched most recently from this form
$('#lab-run-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const el = $('#lab-run-status');
  const mode = labMode();
  try {
    let r;
    if (mode === 'arena_solo' || mode === 'arena_coop') {
      const invalid = validateArenaForm();
      if (invalid) { el.innerHTML = `<span class="bad">${esc(invalid)}</span>`; return; }
      r = await api('/api/run/arena', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode, name: $('#lab-arena-name').value.trim() || null,
          rules: { duration_s: +$('#lab-arena-duration').value, pellets: +$('#lab-arena-pellets').value,
            score_target: +$('#lab-arena-target').value, friendly_fire: !!$('#lab-arena-ff').checked },
          teams: [arenaTeamPayload('a'), arenaTeamPayload('b')],
        }) });
    } else if (mode === 'benchmark') {
      const task = $('#lab-run-bench-task').value;
      const agentId = $('#lab-run-bench-agent').value;
      if (!task) { el.innerHTML = '<span class="bad">No tasks available (backend registry empty)</span>'; return; }
      if (!agentId) { el.innerHTML = '<span class="bad">Pick an agent for the task run</span>'; return; }
      r = await api('/api/run/benchmark', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: task, agent_id: agentId, adapter: $('#lab-run-bench-adapter').value,
          model: $('#lab-run-bench-model').value.trim() || null }) });
    } else {
      if (mode && mode !== 'exploration') { el.innerHTML = `<span class="bad">Unknown mode '${esc(mode)}'</span>`; return; }
      r = await api('/api/run/explore', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_id: $('#lab-run-agent').value, adapter: $('#lab-adapter').value,
          model: $('#lab-run-model').value.trim() || null,
          sessions: parseInt($('#lab-run-sessions').value, 10) || 1,
          max_steps: parseInt($('#lab-run-steps').value, 10) || null,
          concurrency: parseInt($('#lab-run-concurrency').value, 10) || 1,
          max_continuations: parseInt($('#lab-run-cont').value, 10) || 0 }) });
    }
    labRunJob = r.job_id;
    const stopBtn = $('#lab-run-stop');
    if (stopBtn) stopBtn.disabled = false;
    pollRunJob(r.job_id, el, (j) => {
      if (mode === 'arena_solo' || mode === 'arena_coop') {
        const res = j.result || {};
        el.innerHTML = `<span class="ok">Match ${esc(mode)} done — ${esc(res.end_reason || '?')}`
          + `${res.winner ? ' · winner ' + esc(res.winner) : ''} · team scores `
          + Object.entries(res.team_stats || {}).map(([t, s]) => `${esc(t)} ${s}`).join('/') + '</span>'
          + (res.sessions || []).map((s) => `<div class="small">#${esc(s.session_id)} ${esc(s.agent_id)} (${esc(s.team)})</div>`).join('')
          + `<div class="small muted">Open the Arena selector in Watch &amp; Replay to watch or open the recorded sessions.</div>`;
        loadArenaMatches();
      } else if (mode === 'benchmark') {
      const res = j.result || {};
      el.innerHTML = `<span class="ok">Task ${esc(res.task_id || '')} done — session #${esc(res.session_id || '?')} </span>`
        + `<span class="small muted">${res.action_count != null ? res.action_count + ' actions · ' : ''}end ${esc(res.end_reason || res.error || '')}</span>`;
      } else {
        el.innerHTML = `<span class="ok">Done.</span> ` + (j.progress || []).map((p) =>
          `s${esc(p.session_id)}: ${esc(p.actions)} acts, ${esc(p.end_reason)}${p.continue_requested ? ' · asked to continue' : ''}${p.error ? `<br><span class="bad small">${esc(String(p.error))}</span>` : ''}`).join(' · ');
      }
      loadSessions();
    });
  } catch (err) { el.innerHTML = `<span class="bad">${esc(String(err))}</span>`; }
});

$('#lab-run-stop')?.addEventListener('click', async () => {
  if (!labRunJob) return;
  const btn = $('#lab-run-stop');
  btn.disabled = true;
  btn.textContent = 'Stopping…';
  try {
    await api(`/api/run/jobs/${labRunJob}/stop`, { method: 'POST' });
    setTimeout(() => { btn.textContent = 'Stop run'; }, 2000);
  } catch (err) {
    $('#lab-run-status').innerHTML = `<span class="bad">${esc(String(err))}</span>`;
    btn.disabled = false;
    btn.textContent = 'Stop run';
  }
});

$('#lab-cmp-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const models = $('#lab-cmp-models').value.split(',').map((s) => s.trim()).filter(Boolean);
  if (!models.length) { $('#lab-cmp-status').innerHTML = '<span class="bad">Enter at least one model id</span>'; return; }
  const ctx = parseInt($('#lab-cmp-ctx').value, 10);
  const body = { models, adapter: $('#lab-adapter').value,
    agents: parseInt($('#lab-cmp-agents').value, 10) || 2,
    max_steps: parseInt($('#lab-cmp-steps').value, 10) || 60,
    context_tokens: Number.isFinite(ctx) ? ctx : null };
  const el = $('#lab-cmp-status');
  el.textContent = 'Launching comparison…';
  try {
    const r = await api('/api/run/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body) });
    pollRunJob(r.job_id, el, (j) => {
      el.innerHTML = '<span class="ok">Comparison done.</span>';
      const bm = j.result || {};
      const keys = ['sessions', 'actions_mean', 'distinct_tools_mean', 'throw_share_mean',
        'observation_share_mean', 'goals_set_mean', 'experiments_mean', 'budget_used_mean'];
      $('#lab-cmp-results').innerHTML = `<table><tr><th>model</th>${keys.map((k) => `<th>${k.replace(/_mean$/, '').replace(/_/g, ' ')}</th>`).join('')}</tr>` +
        Object.entries(bm).map(([m, a]) => `<tr><td class="mono small">${esc(m)}</td>${keys.map((k) => `<td>${a[k] ?? '–'}</td>`).join('')}</tr>`).join('') + '</table>';
    });
  } catch (err) { el.innerHTML = `<span class="bad">${esc(String(err))}</span>`; }
});

$('#lab-prov-run')?.addEventListener('click', async () => {
  const el = $('#lab-prov');
  el.textContent = 'Checking…';
  try {
    const r = await api('/api/audit/provenance');
    if (r.available === false) { el.innerHTML = `<span class="muted">${esc(r.error || r.note || 'unavailable')}</span>`; return; }
    const m = r.mismatches || [];
    el.innerHTML = m.length
      ? m.map((s) => `<div class="row"><b>session ${s.session_id}</b> (${esc(s.agent_id)}): row says '${esc(s.row_adapter)}', actually ran ${esc((s.ran || []).join(', '))}` +
          (s.items ? `<br><span class="small muted">affected: ${esc(JSON.stringify(s.items))}</span>` : '') + '</div>').join('')
      : '<span class="ok">No provenance mismatches: every session ran under a single matching adapter.</span>';
  } catch (e) { el.innerHTML = `<span class="bad">${esc(String(e))}</span>`; }
});

// Behaviour-dataset form: build climb/jump/locomotion/manipulation/placement/affordance datasets from
// the recorded log, into the selected registry (audit 2026-11-13). Not just throws.
let behKinds = null;
async function loadBehaviourKinds() {
  if (behKinds) return;
  try {
    const r = await api('/api/ml/dataset-kinds');
    behKinds = r.kinds.filter((k) => k.name !== 'throw');  // throws have their own generate form
    const sel = $('#nn-beh-kind');
    if (sel) {
      sel.innerHTML = behKinds.map((k) => `<option value="${esc(k.name)}">${esc(k.name)}</option>`).join('');
      const showDesc = () => { const k = behKinds.find((x) => x.name === sel.value); $('#nn-beh-desc').textContent = k ? k.description : ''; };
      sel.onchange = showDesc; showDesc();
    }
  } catch (e) { /* non-fatal */ }
}
document.getElementById('nn-beh-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const kind = $('#nn-beh-kind').value;
  const name = $('#nn-beh-name').value.trim();
  if (!name) { $('#nn-beh-status').innerHTML = '<span class="bad">Name required</span>'; return; }
  try {
    $('#nn-beh-status').textContent = 'Building…';
    const r = await api('/api/ml/build-behaviour', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, kind, scope: nnSel.scope, owner: nnSel.owner })
    });
    $('#nn-beh-status').textContent = `Job ${r.job_id} started`;
    pollJob(r.job_id);
  } catch (err) { $('#nn-beh-status').innerHTML = `<span class="bad">${esc(String(err))}</span>`; }
});

// Kill switch: one click stops every running session and run job
$('#panic')?.addEventListener('click', async () => {
  const btn = $('#panic');
  btn.disabled = true;
  btn.textContent = 'Stopping…';
  try {
    const r = await api('/api/panic', { method: 'POST' });
    const parts = [];
    if (r.sessions_signalled) parts.push(`${r.sessions_signalled} session${r.sessions_signalled === 1 ? '' : 's'}`);
    if (r.run_jobs_stopping) parts.push(`${r.run_jobs_stopping} run job${r.run_jobs_stopping === 1 ? '' : 's'}`);
    $('#health').textContent = parts.length ? `kill switch: stopping ${parts.join(' + ')}` : $('#health').textContent;
    setTimeout(() => { btn.disabled = false; btn.textContent = 'Kill switch'; loadSessions(); }, 1500);
  } catch (err) {
    btn.textContent = 'Kill switch';
    btn.disabled = false;
    $('#health').textContent = String(err);
  }
});

// ------------------------------------------------------------------ live console (SSE, all sessions)
const lcState = { es: null, cols: new Map() };   // running, session columns by sid
function lcColFor(sid, meta = null) {
  let col = lcState.cols.get(sid);
  if (!col) {
    const grid = $('#live-console');
    const box = document.createElement('div');
    box.className = 'lc-col';
    box.innerHTML = `<h4>#${sid}</h4><div class="lc-evs"></div>`;
    grid.appendChild(box);
    col = { title: box.querySelector('h4'), evs: box.querySelector('.lc-evs'), n: 0 };
    lcState.cols.set(sid, col);
    while (grid.children.length > 6) grid.children[0].remove();   // keep the newest 6 sessions
  }
  if (meta) col.title.innerHTML = `#${sid} <span class="small muted">${esc(meta.agent_id || '')} · ${esc(meta.mode || '')}</span> ` +
    (meta.status === 'running' ? '<span class="pill-ok">●</span>' : `<span class="pill-end">${esc(meta.status || '')}</span>`);
  return col;
}
function lcPush(sid, text, bad = false, meta = null) {
  const col = lcColFor(sid, meta);
  const div = document.createElement('div');
  div.className = 'lc-ev' + (bad ? ' bad' : '');
  div.textContent = text;
  col.evs.appendChild(div);
  col.n++;
  if (col.n > 40) { col.evs.children[0].remove(); col.n--; }
  col.evs.scrollTop = col.evs.scrollHeight;
}
$('#lc-toggle')?.addEventListener('click', () => {
  const btn = $('#lc-toggle');
  if (lcState.es) {
    lcState.es.close(); lcState.es = null;
    btn.textContent = 'Start';
    $('#live-console').innerHTML = '<span class="muted small">Stopped.</span>';
    lcState.cols.clear();
    return;
  }
  btn.textContent = 'Stop';
  $('#live-console').innerHTML = '';
  lcState.cols.clear();
  lcState.es = new EventSource('/api/stream/events');
  lcState.es.onmessage = (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    if (ev.type === 'session') {
      const line = ev.status === 'completed' ? `session ended (${ev.end_reason || ev.status})`
                                             : `session ${ev.status} (${ev.agent_id}, #${ev.session_number})`;
      lcPush(ev.session_id, line, false, { agent_id: ev.agent_id, mode: ev.mode, status: ev.status });
      lcColFor(ev.session_id, ev);
    } else if (ev.type === 'action') {
      const args = Object.entries(ev.args || {}).slice(0, 2)
        .map(([k, v]) => `${k}=${typeof v === 'number' ? (Math.round(v * 100) / 100) : v}`).join(' ');
      lcPush(ev.session_id, `${ev.tool}(${args})${ev.success ? '' : ' ✗' + (ev.error_code || '')}`,
             !ev.success, { status: 'running', agent_id: '' });
    }
  };
  lcState.es.onerror = () => {
    if (lcState.es && lcState.es.readyState === EventSource.CLOSED) {
      lcState.es = null; btn.textContent = 'Start';
    }
  };
});

// ------------------------------------------------------------------ Infrastructure: overview, providers, monitor, resources
// Every value rendered here passes through esc() (app.js header XSS policy); numeric fields are
// server-side enums or numbers and are formatted before interpolation.
const num = (v, d = 2) => (typeof v === 'number' && isFinite(v) ? (+v).toFixed(d) : '–');
const kvRows = (obj) => Object.entries(obj || {}).map(([k, v]) =>
  `<tr><td class="muted small">${esc(String(k))}</td><td>${typeof v === 'number' ? num(v, 1)
  : (v === null || v === undefined || typeof v === 'object' ? '–' : esc(String(v)))}</td></tr>`).join('');
const reqRow = (r) => `<tr><td class="mono small">${esc(r.request_id?.slice(0, 8))}…</td>` +
  `<td class="small"><button class="btn small mon-open" data-id="${esc(r.request_id)}">${esc(`${r.provider || ''}/${r.model || ''}`)}</button></td>` +
  `<td class="small">${esc(r.agent || '–')}</td><td>${esc(r.status || '–')}</td>` +
  `<td>${r.http_status ?? '–'}</td><td>${num(r.latency_ms, 1)}ms</td><td>${r.total_tokens ?? '–'}</td></tr>`;

async function renderInfra() {
  const [pv, res, q] = await Promise.all([
    api('/api/gateway/providers?models=false'),
    api('/api/gateway/resources'),
    api('/api/gateway/queue'),
  ]);
  let reqs = { requests: [] };
  try { reqs = await api('/api/gateway/requests?limit=12'); } catch (e) { /* monitor off */ }
  const gpu = (res.gpus || [])[0] || {};
  $('#inf-tiles').innerHTML = [
    ['CPU', `${num(res.cpu, 0)}%`],
    ['RAM', `${num(res.memory?.percent, 0)}%`],
    ['GPU', gpu.name ? `${esc(String(gpu.name))} · ${num(gpu.utilization_percent ?? gpu.utilization, 0)}%` : 'n/a'],
    ['Providers', `${pv.providers.length} · ${pv.providers.filter((p) => p.health?.reachable).length} reachable`],
    ['Queue', `${num(q?.queue_depth, 0)} waiting`],
    ['Requests', `${num((reqs.requests || []).length, 0)} recent`],
  ].map(([k, v]) => `<div class="metric-box tile"><div class="label">${k}</div><div class="value">${v}</div></div>`).join('');
  $('#inf-providers').innerHTML = pv.providers.map((p) =>
    `<div class="row"><b>${esc(p.name)}</b> <span class="muted small">${esc(p.kind)}</span> · ` +
    (p.health?.reachable ? `<span class="pill-ok">reachable ${num(p.health.latency_ms, 0)}ms</span>` : '<span class="pill-end">offline</span>') +
    ` · ${p.default_models?.length ?? 0} model(s)</div>`).join('');
  $('#inf-hw').innerHTML = `<table>${kvRows({ platform: res.platform, cpu_percent: res.cpu })}` +
    `</table><div class="small muted">GPU: ${esc(JSON.stringify(res.gpus?.[0] ?? {})).slice(0, 300)}</div>`;
  $('#inf-queue').innerHTML = (q?.queued || []).length
    ? q.queued.map((w) => `<div class="row small">#${w.seq} ${esc(w.model || '?')} for ${esc(w.agent || '–')} (prio ${w.priority})</div>`).join('')
    : '<span class="muted small">Queue empty.</span>';
  $('#inf-recent').innerHTML = (reqs.requests || []).length
    ? `<table><tr><th>id</th><th>provider/model</th><th>agent</th><th>status</th><th>http</th><th>ms</th><th>tok</th></tr>${reqs.requests.map(reqRow).join('')}</table>`
    : '<span class="muted small">No gateway requests yet.</span>';
  $('#inf-updated').textContent = `updated ${new Date().toLocaleTimeString()}`;
}

let infraTimer = null;
async function loadInfra() {
  await renderInfra();
  if (infraTimer) clearInterval(infraTimer);
  infraTimer = setInterval(() => { if ($('#tab-infra')?.classList.contains('active')) renderInfra(); }, 5000);
}

// -- Provider catalogue (LM Studio and friends land here; models via the per-provider detail).
function providerMeta(p) {
  return [
    ['kind', p.kind], ['base_url', p.base_url], ['key', (p.key_present ? 'present' : 'none')],
    ['api key env', p.api_key_env], ['models', p.default_models?.length ?? 0], ['enabled', String(p.enabled)],
  ];
}
async function loadProviders() {
  const el = $('#pv-list');
  try {
    $('#pv-refresh').onclick = loadProviders;
    const r = await api('/api/gateway/providers?models=false');
    el.innerHTML = r.providers.length
      ? r.providers.map((p) => `<div class="row"><button class="btn small pv-open" data-name="${esc(p.name)}">open</button> ` +
        `<b>${esc(p.name)}</b> <span class="muted small">${esc(p.kind)} · ${esc(p.base_url || '')}</span> ` +
        (p.health?.reachable ? `<span class="pill-ok">reachable ${num(p.health.latency_ms, 0)}ms</span>` : '<span class="pill-end">offline</span>') +
        ` · ${p.default_models?.length ?? 0} model(s)</div>`).join('')
      : '<span class="muted">No providers configured (config-derived: adapters in playground config).</span>';
    document.querySelectorAll('.pv-open').forEach((b) => b.onclick = async () => {
      const name = b.dataset.name;
      $('#pv-detail-card').style.display = 'block';
      $('#pv-detail').innerHTML = '<span class="muted">Loading…</span>';
      try {
        const [d, m] = await Promise.all([
          api(`/api/gateway/providers/${encodeURIComponent(name)}`),
          api(`/api/gateway/providers/${encodeURIComponent(name)}/models`),
        ]);
        $('#pv-detail').innerHTML = `<table>${kvRows(Object.fromEntries(providerMeta(d)))}</table>` +
          `<h4 class="small">Models on the server (${(m.models || []).length}, ${m.loaded?.length ?? 0} loaded)</h4>` +
          ((m.models || []).length
            ? `<table><tr><th>id</th><th>state</th><th>ctx</th></tr>` +
              m.models.map((mo) => { const lo = (m.loaded || []).find((x) => (x.id || x) === mo.id) || {};
                return `<tr><td class="mono small">${esc(mo.id)}</td><td>${esc(mo.state || lo.state || 'not loaded')}</td><td>${mo.context ?? lo.context ?? '–'}</td></tr>`; }).join('') + '</table>'
            : '<span class="muted small">No models found at the provider endpoint.</span>');
      } catch (e) {
        $('#pv-detail').innerHTML = `<span class="bad">${esc(String(e))}</span>`;
      }
    });
    $('#pv-status').textContent = `${r.providers.length} provider(s), ` +
      `${r.providers.filter((p) => p.health?.reachable).length} reachable`;
  } catch (e) {
    el.innerHTML = `<span class="bad">${esc(String(e))}</span>`;
  }
}

// -- API monitor: gateway request history with filters and per-request detail.
async function loadMonitor() {
  const btn = $('#mon-refresh');
  if (btn.dataset.wired) return;
  btn.dataset.wired = '1';
  btn.onclick = async () => {
    const p = new URLSearchParams();
    if ($('#mon-f-provider').value) p.set('provider', $('#mon-f-provider').value);
    if ($('#mon-f-model').value) p.set('model', $('#mon-f-model').value);
    if ($('#mon-f-agent').value) p.set('agent', $('#mon-f-agent').value);
    if ($('#mon-f-status').value) p.set('status', $('#mon-f-status').value);
    if ($('#mon-f-failed').checked) p.set('failed', 'true');
    p.set('limit', '200');
    try {
      const r = await api(`/api/gateway/requests?${p}`);
      $('#mon-stats').textContent = `${(r.requests || []).length} request(s)` +
        (r.stats ? ` · ${esc(JSON.stringify(r.stats)).slice(0, 120)}` : '');
      $('#mon-table').innerHTML = (r.requests || []).length
        ? `<table><tr><th>id</th><th>provider/model</th><th>agent</th><th>status</th><th>http</th><th>ms</th><th>tok</th></tr>${r.requests.map(reqRow).join('')}</table>`
        : '<span class="muted">No requests match those filters.</span>';
      document.querySelectorAll('.mon-open').forEach((b) => b.onclick = async () => {
        try {
          const d = await api(`/api/gateway/requests/${encodeURIComponent(b.dataset.id)}`);
          $('#mon-detail-card').querySelector('h3').textContent = `Request ${d.request_id}`;
          $('#mon-detail').innerHTML =
            `<table>${kvRows(Object.fromEntries(Object.entries(d).filter(([, v]) => typeof v !== 'object' || v === null)))}</table>` +
            `<div class="small muted">request/response bodies are redacted+truncated by the gateway on capture.</div>`;
        } catch (e) { $('#mon-detail').innerHTML = `<span class="bad">${esc(String(e))}</span>`; }
      });
    } catch (e) {
      $('#mon-table').innerHTML = `<span class="bad">${esc(String(e))}</span>`;
    }
  };
}

// -- Resources: measured hardware + live concurrency limits (also rendered into the infra tab).
async function loadResources() {
  const [res, lim, q] = await Promise.all([
    api('/api/gateway/resources'),
    api('/api/gateway/concurrency'),
    api('/api/gateway/queue'),
  ]);
  $('#res-hw').innerHTML = `<table>${kvRows({ platform: res.platform, cpu_percent: res.cpu,
    memory: res.memory?.percent ?? res.memory, gpus: (res.gpus || []).length })}
    </table>` + (res.gpus || []).map((g) =>
    `<div class="small muted">${esc(String(g.name))}: util ${num(g.utilization_percent ?? g.utilization, 0)}%` +
    (g.total_memory ? ` · VRAM ${num((g.used_memory ?? 0), 1)}/${num(g.total_memory, 1)}GB` : '') + '</div>').join('');
  $('#rc-global').value = lim.max_concurrent ?? 0;
  $('#rc-provider').value = lim.provider_default ?? 0;
  $('#rc-model').value = lim.model_default ?? 0;
  $('#rc-agent').value = lim.agent_default ?? 0;
  $('#rc-queue').value = lim.queue_max ?? 0;
  $('#rc-priority').value = lim.default_priority ?? 5;
  const ov = Object.entries(lim.per_provider || {}).map(([k, v]) => `${k}=${v}`).join(', ')
    + Object.entries(lim.per_model || {}).map(([k, v]) => ` model:${k}=${v}`).join(', ');
  $('#rc-overrides').textContent = ov ? `overrides: ${ov}` : 'no per-provider/model overrides';
  $('#res-queue').innerHTML = (q?.queued || []).length
    ? q.queued.map((w) => `<div class="row small">#${w.seq} ${esc(w.model || '?')} for ${esc(w.agent || '–')} (prio ${w.priority})</div>`).join('')
    : '<span class="muted small">Queue empty.</span>';
}
$('#res-conc-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    await api('/api/gateway/concurrency', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        max_concurrent: parseInt($('#rc-global').value, 10) || 0,
        provider_default: parseInt($('#rc-provider').value, 10) || 0,
        model_default: parseInt($('#rc-model').value, 10) || 0,
        agent_default: parseInt($('#rc-agent').value, 10) || 0,
        queue_max: parseInt($('#rc-queue').value, 10) || 0,
        default_priority: parseInt($('#rc-priority').value, 10) || 5,
      }) });
    $('#rc-status').textContent = 'applied.';
    loadResources();
  } catch (err) { $('#rc-status').innerHTML = `<span class="bad">${esc(String(err))}</span>`; }
});

// ------------------------------------------------------------------ boot
const h = await api('/api/health');
$('#health').textContent = `Online · v${h.version} · ${h.agents} agents`;
$('#health').title = `Configuration ${h.config_hash} · ${h.db}`;
await loadSessions();
const startTab = location.hash.slice(1);
if (startTab && startTab !== 'watch') showTab(startTab);
