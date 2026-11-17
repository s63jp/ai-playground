// CombatFX (M5.5) — arena combat visuals driven ONLY by recorded physics data.
//
// Frame convention: children of the z-up world group (the dashboard rotates that group
// to convert MuJoCo z-up into three.js y-up), so every mesh below is built along local
// +Z (up) with +Y as the marker's long axis — the same axes the MuJoCo box bodies use.
// All geometry here is presentation only: positions come from recorded frame objects
// (the real physics bodies), colours from the recorded arena manifest + pellet
// attribution events. No motion is invented: trails are replayed recorded poses.
import * as THREE from 'three';

export const TEAM_COLORS = [
  { re: /red/i, c: 0xe0463f }, { re: /blue/i, c: 0x3f7fe0 }, { re: /green/i, c: 0x3fae60 },
  { re: /orange/i, c: 0xe0873a }, { re: /purple|violet/i, c: 0x9a5ae0 }, { re: /cyan|teal/i, c: 0x35b5b5 },
  { re: /yellow/i, c: 0xd9c33a }, { re: /white/i, c: 0xd8d8d8 }, { re: /black|dark/i, c: 0x777788 },
];

export function _oid(name) {
  return String(name || '').toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '') || 'team';
}

export function teamColour(team) {
  const key = String(team || '');
  for (const t of TEAM_COLORS) if (t.re.test(key)) return t.c;
  let h = 0;                                        // deterministic hash fallback
  for (const ch of key) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return new THREE.Color().setHSL((h % 360) / 360, 0.62, 0.52).getHex();
}

// The paintball marker (M5.5 Part 1). `spec` = the equipment registry entry served by
// /api/run/modes (dims_m [x width, y length, z height], combat_class, grip), `accent` =
// team colour. Proportions follow the registered dims; local +Y is the muzzle direction.
export function makeMarker(spec, accent) {
  const dims = (spec && spec.dims_m) || [0.05, 0.19, 0.12];
  const w = Math.max(0.02, dims[0]), L = Math.max(0.06, dims[1]), Ht = Math.max(0.04, dims[2]);
  const cls = (spec && spec.combat_class) || 'ranged';
  const accentMat = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.45, metalness: 0.35 });
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x23272e, roughness: 0.55, metalness: 0.55 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x14161a, roughness: 0.7 });
  const g = new THREE.Group();
  const add = (geo, mat, pos, rot) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(...pos);
    if (rot) m.rotation.set(...rot);
    g.add(m);
    return m;
  };
  if (cls === 'shield') {
    add(new THREE.BoxGeometry(w, L * 0.35, Ht), accentMat, [0, 0, 0]);                       // facing plate
    add(new THREE.BoxGeometry(w * 0.3, L * 0.3, Ht * 0.6), bodyMat, [-w * 0.3, 0, 0]);       // frame rib
    add(new THREE.CylinderGeometry(0.016, 0.016, L * 0.28), darkMat, [w * 0.22, 0, -Ht * 0.3]); // handle
  } else if (cls === 'melee' || (cls === 'throwable' && L >= 0.5)) {
    add(new THREE.CylinderGeometry(w * 0.19, w * 0.19, L, 10), darkMat, [0, 0, 0], [Math.PI / 2, 0, 0]); // rod
    add(new THREE.CylinderGeometry(w * 0.14, w * 0.14, L * 0.12, 10), accentMat, [0, L * 0.20, 0], [Math.PI / 2, 0, 0]); // tip band
    add(new THREE.CylinderGeometry(w * 0.15, w * 0.15, L * 0.18, 10), accentMat, [0, -L * 0.36, 0], [Math.PI / 2, 0, 0]); // grip wrap
  } else if (cls === 'throwable') {
    add(new THREE.SphereGeometry(w * 0.5, 14, 10), accentMat, [0, 0, 0]);                     // shell
    add(new THREE.CylinderGeometry(0.008, 0.008, w * 0.35), darkMat, [0, 0, w * 0.48]);       // spout
  } else {
    // ranged marker: receiver + barrel + hopper + grip + trigger guard (+ tank for long guns)
    const long = L >= 0.4;
    const barrelLen = long ? L * 0.34 : L * 0.30;
    const recvLen = long ? L * 0.46 : L * 0.42;
    // receiver along the middle (box centre == the MuJoCo body origin)
    add(new THREE.BoxGeometry(w * 0.72, recvLen, Ht * 0.52), bodyMat, [0, -L * 0.04, 0]);
    // barrel cylinder (three cylinders are y-axis-aligned already — this IS the local +Y axis)
    add(new THREE.CylinderGeometry(w * 0.22, w * 0.22, barrelLen, 10), darkMat,
      [0, L * 0.50 - barrelLen / 2, Ht * 0.08]);
    // muzzle ring in the team colour
    add(new THREE.CylinderGeometry(w * 0.26, w * 0.26, L * 0.05, 10), accentMat,
      [0, L * 0.50 - L * 0.025, Ht * 0.08]);
    if (spec && spec.id === 'dual_paintball_pistols' || (w >= 0.08 && L < 0.4)) {
      // dual pistols: a mirrored second barrel above the first
      add(new THREE.CylinderGeometry(w * 0.20, w * 0.20, barrelLen, 10), darkMat,
        [0, L * 0.50 - barrelLen / 2, Ht * 0.28]);
    }
    // hopper / loader sitting on top of the receiver (the paintball silhouette)
    const hopper = add(new THREE.SphereGeometry(Ht * 0.30, 12, 10), accentMat, [0, L * 0.02, Ht * 0.60]);
    hopper.scale.set(0.8, 1.0, 0.85);
    // grip: tilted box below the receiver, rear side
    add(new THREE.BoxGeometry(w * 0.55, w * 0.4, Ht * 0.62), darkMat,
      [0, -L * 0.16 - w * 0.02, -Ht * 0.42], [0.9, 0, 0]);
    // trigger guard
    const guard = add(new THREE.TorusGeometry(Ht * 0.14, w * 0.055, 6, 14), darkMat,
      [0, -L * 0.10, -Ht * 0.22], [0, Math.PI / 2, 0]);
    guard.scale.set(1, 1, 0.9);
    if (long) add(new THREE.CylinderGeometry(w * 0.30, w * 0.30, L * 0.22, 10), accentMat,
      [0, -L * 0.44, -Ht * 0.14]);   // propellant tank
  }
  g.userData.marker = (spec && spec.id) || 'marker';
  return g;
}

// A paintball: glossy sphere in the shooter's team colour. The real projectile body
// radius is 0.02 m (0.04 for grenade shells); the visual is scaled up 1.6x so a 2 cm
// paintball is readable at arena zoom, while every position stays the recorded pose.
export function makePellet(radiusM, colourHex) {
  const r = Math.max(0.022, 1.6 * (radiusM || 0.02));
  const mat = new THREE.MeshStandardMaterial({
    color: colourHex, roughness: 0.18, metalness: 0.05,
    emissive: colourHex, emissiveIntensity: 0.22, transparent: true, opacity: 0.95,
  });
  return new THREE.Mesh(new THREE.SphereGeometry(r, 14, 10), mat);
}

// Flat paint decals on the ground plane (the only surface where a decal lies correct
// without a recorded surface normal). Instanced per colour, recycled FIFO.

// Short-lived impact particles (a paint splash reads as a 3D event even without a
// surface normal). One InstancedMesh pool, per-instance colour via setColorAt.
class ParticlePool {
  constructor(world, capacity = 260) {
    this.world = world;
    this.capacity = capacity;
    this.geometry = new THREE.SphereGeometry(0.012, 6, 5);
    this.mesh = new THREE.InstancedMesh(this.geometry,
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9, depthWrite: false }), capacity);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    for (let i = 0; i < capacity; i++) {
      this.mesh.setMatrixAt(i, new THREE.Matrix4().makeTranslation(0, 0, -50));
      this.mesh.setColorAt(i, new THREE.Color(0xffffff));
    }
    world.add(this.mesh);
    this.alive = new Map();   // pool index -> {v: THREE.Vector3, life, age}
    this.next = 0;
  }
  burst(x, y, z, colHex, n = 8) {
    const c = new THREE.Color(colHex);
    for (let k = 0; k < n; k++) {
      const idx = this.next;
      this.next = (this.next + 1) % this.capacity;
      const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI * 0.5;
      const sp = 1.1 + Math.random() * 1.4;             // m/s
      const v = new THREE.Vector3(Math.cos(th) * Math.sin(ph) * sp,
        Math.sin(th) * Math.sin(ph) * sp, Math.cos(ph) * sp * 0.6 + 0.4);
      this.alive.set(idx, { v, life: 0.35 + Math.random() * 0.25, age: 0 });
      this.mesh.setColorAt(idx, c);
      this._put(idx, x, y, z);
    }
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
  _put(idx, x, y, z) {
    this.mesh.setMatrixAt(idx, new THREE.Matrix4().makeTranslation(x, y, z));
    this.mesh.instanceMatrix.needsUpdate = true;
  }
  tick(dt) {
    if (!this.alive.size) return;
    for (const [idx, p] of this.alive) {
      p.age += dt;
      if (p.age >= p.life) {
        this.alive.delete(idx);
        this.mesh.setMatrixAt(idx, new THREE.Matrix4().makeTranslation(0, 0, -50));
        this.mesh.instanceMatrix.needsUpdate = true;
        continue;
      }
      // gravity -9.8 m/s^2 along MuJoCo -z in the world group's local frame
      p.v.z -= 9.8 * dt * 0.55;
      p.v.x *= 1 - 1.6 * dt; p.v.y *= 1 - 1.6 * dt;
      const m = new THREE.Matrix4();
      this.mesh.getMatrixAt(idx, m);
      const pos = new THREE.Vector3().setFromMatrixPosition(m).addScaledVector(p.v, dt);
      this.mesh.setMatrixAt(idx, new THREE.Matrix4().makeTranslation(pos.x, pos.y, pos.z));
      this.mesh.instanceMatrix.needsUpdate = true;
    }
  }
  reset() {
    this.alive.clear();
    for (let i = 0; i < this.capacity; i++) this.mesh.setMatrixAt(i, new THREE.Matrix4().makeTranslation(0, 0, -50));
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

export class CombatFX {
  constructor(world) {
    this.world = world;
    this.parts = new ParticlePool(world);
    this.trails = new Map();       // pellet id -> {pts, line} of recorded pose samples
    // Splat root: rotate the plane set's normal (+Y in plane space) onto local +Z, so
    // every decal lies FLAT on the ground plane without per-instance rotations.
    this.splatRoot = new THREE.Group();
    this.splatRoot.rotation.x = Math.PI / 2;
    world.add(this.splatRoot);
    this._plane = new THREE.PlaneGeometry(1, 1);
    this.splats = new Map();                   // colourHex -> {mesh, count, next, t: Float32Array}
    this.splatCapacity = 256;
  }
  addSplat(x, y, z, colHex, sizeM, simT) {
    sizeM = sizeM || (0.09 + Math.abs(((x * 12.9898 + y * 78.233) % 1)) * 0.05);
    let p = this.splats.get(colHex);
    if (!p) {
      const mat = new THREE.MeshBasicMaterial({ color: colHex, transparent: true, opacity: 0.7, depthWrite: false });
      const mesh = new THREE.InstancedMesh(this._plane, mat, this.splatCapacity);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      for (let i = 0; i < this.splatCapacity; i++) mesh.setMatrixAt(i, new THREE.Matrix4().makeTranslation(0, 0, -50));
      this.splatRoot.add(mesh);
      p = { mesh, count: 0, next: 0, t: new Float32Array(this.splatCapacity).fill(Infinity) };
      this.splats.set(colHex, p);
    }
    const idx = p.count < this.splatCapacity ? p.count : p.next;
    const m = new THREE.Matrix4().makeRotationZ(Math.abs(((x * 12.9898 + y * 78.233) % 1)) * Math.PI * 2)
      .premultiply(new THREE.Matrix4().makeScale(sizeM, sizeM, 1));
    m.setPosition(x, y, Math.min(z, 0.12) + 0.008);
    p.mesh.setMatrixAt(idx, m);
    p.t[idx] = simT == null ? -Infinity : simT;    // the collision's recorded sim time
    if (p.count < this.splatCapacity) p.count += 1;
    else p.next = (p.next + 1) % this.splatCapacity;
    p.mesh.instanceMatrix.needsUpdate = true;
  }
  // Replay honesty: a decal is visible only once the replay has reached the frame its
  // recording collision happened in. Hidden instances park under the world.
  timeFilter(simT) {
    for (const [, p] of this.splats) {
      p.meshHidden = p.meshHidden || new Uint8Array(this.splatCapacity);
      p.pose = p.pose || [];
      let changed = false;
      for (let i = 0; i < this.splatCapacity; i++) {
        const active = Number.isFinite(p.t[i]) && p.t[i] <= simT;   // used slot: got a real sim time
        const hidden = p.meshHidden[i] === 1;
        if (active && hidden) {
          if (p.pose[i]) p.mesh.setMatrixAt(i, p.pose[i]);
          p.meshHidden[i] = 0;
          changed = true;
        } else if (!active && !hidden) {
          p.pose[i] = new THREE.Matrix4();
          p.mesh.getMatrixAt(i, p.pose[i]);
          p.mesh.setMatrixAt(i, new THREE.Matrix4().makeTranslation(0, 0, -50));
          p.meshHidden[i] = 1;
          changed = true;
        }
      }
      if (changed) p.mesh.instanceMatrix.needsUpdate = true;
    }
  }
  burst(x, y, z, colHex, n) { this.parts.burst(x, y, z, colHex, n || 8); }
  tick(dt) { this.parts.tick(dt); }
  trail(pid, posArr, colHex, frameIdx) {
    // Recorded-pose polyline, index-accurate: entries carry the frame index they came
    // from; scrubbing backwards trims everything beyond the shown frame, so the trail
    // is always exactly the pellet's recorded path up to the current frame (64 cap).
    let t = this.trails.get(pid);
    if (!t) {
      const mat = new THREE.LineBasicMaterial({ color: colHex, transparent: true, opacity: 0.5 });
      t = { pts: [], lastIdx: -1, line: new THREE.Line(new THREE.BufferGeometry(), mat) };
      t.line.frustumCulled = false;
      this.world.add(t.line);
      this.trails.set(pid, t);
    }
    const v = new THREE.Vector3(posArr[0], posArr[1], posArr[2]);
    if (t.pts.length && t.pts[t.pts.length - 1][0] > frameIdx) {
      // scrubbed backwards: drop samples beyond the shown frame first
      t.pts = t.pts.filter((e) => e[0] <= frameIdx);
    }
    if (frameIdx > t.lastIdx && (!t.pts.length || t.pts[t.pts.length - 1][1].distanceToSquared(v) > 1e-8)) {
      t.pts.push([frameIdx, v]);
      if (t.pts.length > 64) t.pts.shift();
      t.lastIdx = frameIdx;
      t.line.geometry.setFromPoints(t.pts.map((e) => e[1]));
    }
    t.line.visible = true;
    return t.line;
  }
  dropTrailsExcept(activeIds) {
    for (const [pid, t] of this.trails) if (!activeIds.has(pid)) t.line.visible = false;
  }
  reset() {
    this.parts.reset();
    for (const [, p] of this.splats) {
      for (let i = 0; i < this.splatCapacity; i++) p.mesh.setMatrixAt(i, new THREE.Matrix4().makeTranslation(0, 0, -50));
      p.count = 0; p.next = 0;
      p.mesh.instanceMatrix.needsUpdate = true;
    }
    for (const [, t] of this.trails) { t.line.geometry.dispose(); this.world.remove(t.line); }
    this.trails.clear();
  }
}
