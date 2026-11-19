/* Humanoid V2 viewer: loads the Blender FBX once, bakes each of the 35 meshes into
 * Blender world space, and drives them from live MuJoCo frames through the retarget
 * port in humanoid_v2_math.js. No second animation system, no scripted motion.
 */
import * as THREE from 'three';
import { FBXLoader } from '/static/vendor/three-fbx/loaders/FBXLoader.js';
import { preparePlan, captureRest, retarget, validateFrame } from '/static/js/humanoid_v2_math.js';

const BASE = '/static/assets/humanoid_v2/';
const BOUNDS_TOL = 0.005;          // metres: baked bbox must match the Blender golden bounds

let assetsPromise = null;          // FBX + plan are fetched and parsed ONCE per page

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

function loadAssets() {
  if (!assetsPromise) {
    assetsPromise = Promise.all([
      fetchJson(BASE + 'retarget_plan.json'),
      fetchJson(BASE + 'blender_mesh_bounds.json'),
      new FBXLoader().loadAsync(BASE + 'humanoid_robot_rigged.fbx'),
    ]).then(([plan, bounds, fbx]) => ({ plan: preparePlan(plan), bounds, fbx }));
    assetsPromise.catch(() => { assetsPromise = null; });   // allow a retry after failure
  }
  return assetsPromise;
}

function uniquePositionCount(geo) {
  const pos = geo.attributes.position.array;
  const seen = new Set();
  for (let i = 0; i < pos.length; i += 3) {
    seen.add(`${Math.round(pos[i] * 1e5)},${Math.round(pos[i + 1] * 1e5)},${Math.round(pos[i + 2] * 1e5)}`);
  }
  return seen.size;
}

export async function createHumanoidV2(scene) {
  const { plan, bounds, fbx } = await loadAssets();
  fbx.updateMatrixWorld(true);

  // index the FBX meshes by their exported names; fail closed on any mismatch
  const found = new Map();
  fbx.traverse((o) => {
    if (!o.isMesh) return;
    if (found.has(o.name)) throw new Error(`duplicate FBX mesh name: ${o.name}`);
    found.set(o.name, o);
  });
  const want = Object.keys(plan.meshes);
  const missing = want.filter((n) => !found.has(n));
  const extra = [...found.keys()].filter((n) => !plan.meshes[n]);
  if (missing.length || extra.length) {
    throw new Error(`V2 FBX mesh names mismatch: missing=[${missing}] extra=[${extra}]`);
  }

  const group = new THREE.Group();
  group.name = 'humanoid_v2';
  scene.add(group);
  const meshes = new Map();
  const bakeReport = [];
  const failures = [];
  for (const name of want) {
    const src = found.get(name);
    const geo = src.geometry.clone();
    geo.deleteAttribute('skinIndex');          // rest pose only: no skinning
    geo.deleteAttribute('skinWeight');
    geo.applyMatrix4(src.matrixWorld);         // bake FBX node transform -> Blender world
    geo.computeBoundingBox();
    const gold = bounds.meshes[name];
    const bb = geo.boundingBox;
    const got = [bb.min.x, bb.min.y, bb.min.z, bb.max.x, bb.max.y, bb.max.z];
    const want6 = [...gold.min, ...gold.max];
    const dev = Math.max(...got.map((v, i) => Math.abs(v - want6[i])));
    // three.js FBX geometry is non-indexed (split at normal/UV seams); Blender's count is
    // its welded vertex count, so compare unique positions.
    const nUnique = uniquePositionCount(geo);
    bakeReport.push({ name, dev, nUnique, goldVerts: gold.n_verts });
    if (dev > BOUNDS_TOL || nUnique !== gold.n_verts) {
      failures.push(`${name}: bounds dev ${dev.toFixed(4)} m, unique verts ${nUnique}/${gold.n_verts}`);
    }
    const mesh = new THREE.Mesh(geo, src.material);
    mesh.name = name;
    mesh.matrixAutoUpdate = false;
    group.add(mesh);
    meshes.set(name, mesh);
  }
  if (failures.length) {
    group.clear();
    scene.remove(group);
    throw new Error(`V2 mesh check failed (${failures.length}/${want.length}): ${failures.join('; ')}`);
  }

  let rest = null;

  const api = {
    group,
    meshes,
    plan,
    bakeReport,
    restEpisode: () => (rest ? rest.episode : null),
    /** Apply one live frame. Recaptures the rest pose at each new episode / first frame. */
    update(frame) {
      validateFrame(frame, plan);
      if (rest === null || (frame.episode ?? null) !== rest.episode) {
        rest = captureRest(frame, plan);
      }
      const out = retarget(frame, rest, plan);
      for (const [name, mesh] of meshes) {
        const f = out.objects[name];
        mesh.matrix.set(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
          f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]);
        mesh.matrixWorld.copy(mesh.matrix);    // group is identity, so world == local
      }
    },
    reset() { rest = null; },
    dispose() {
      scene.remove(group);
      for (const mesh of meshes.values()) mesh.geometry.dispose();
      meshes.clear();
    },
  };
  return api;
}
