/* Humanoid V2 retargeting, browser side. Dependency-free (no three.js) so it runs
 * under node for the parity test. Exact port of playground/rl_live/humanoid_v2_retarget.py:
 *   reset:  T0[b] = pose(frame) at reset; A = translate(mean(T0[driver].p - head_world))
 *   frame:  D[b] = pose(frame[b]) * inv(T0[b]);  object[mesh] = D[driver(mesh)] * A
 *           bone[b] = D[driver(b)] * A * rest_world[b]
 * 4x4 matrices are flat row-major arrays of 16 numbers. Applied once per frame, no state
 * carried between frames besides the reset capture.
 */

export function poseMat(p, q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(n > 1e-9)) throw new Error('quaternion has zero norm');
  const w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), p[0],
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), p[1],
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), p[2],
    0, 0, 0, 1,
  ];
}

export function mul(A, B) {
  const C = new Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += A[i * 4 + k] * B[k * 4 + j];
      C[i * 4 + j] = s;
    }
  }
  return C;
}

export function invPose(T) {
  // rigid transform inverse: [R t]^-1 = [R^T  -R^T t]
  const r = [T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]];
  const t = [T[3], T[7], T[11]];
  const RT = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
  const nt = [
    -(RT[0] * t[0] + RT[1] * t[1] + RT[2] * t[2]),
    -(RT[3] * t[0] + RT[4] * t[1] + RT[5] * t[2]),
    -(RT[6] * t[0] + RT[7] * t[1] + RT[8] * t[2]),
  ];
  return [RT[0], RT[1], RT[2], nt[0], RT[3], RT[4], RT[5], nt[1], RT[6], RT[7], RT[8], nt[2], 0, 0, 0, 1];
}

export function translate(v) {
  return [1, 0, 0, v[0], 0, 1, 0, v[1], 0, 0, 1, v[2], 0, 0, 0, 1];
}

/** Fail closed: frame must hold exactly the plan's rig bodies, each with finite p and q. */
export function validateFrame(frame, plan) {
  if (!frame || typeof frame !== 'object' || !frame.bodies || typeof frame.bodies !== 'object') {
    throw new Error('frame has no bodies map');
  }
  const want = new Set(plan.rig_bodies);
  const got = new Set(Object.keys(frame.bodies));
  const missing = plan.rig_bodies.filter((n) => !got.has(n));
  const unexpected = [...got].filter((n) => !want.has(n));
  if (missing.length || unexpected.length) {
    throw new Error(`frame body set mismatch: missing=[${missing}] unexpected=[${unexpected}]`);
  }
  for (const n of plan.rig_bodies) {
    const b = frame.bodies[n];
    if (!b || !Array.isArray(b.p) || !Array.isArray(b.q) || b.p.length !== 3 || b.q.length !== 4) {
      throw new Error(`${n}: pose needs p[3] and q[4]`);
    }
    for (const v of [...b.p, ...b.q]) if (!Number.isFinite(v)) throw new Error(`${n}: non-finite pose value`);
  }
}

/** Reset capture: T0 for every rig body and the single rest alignment A. */
export function captureRest(frame, plan) {
  validateFrame(frame, plan);
  const T0 = {};
  for (const n of plan.rig_bodies) T0[n] = poseMat(frame.bodies[n].p, frame.bodies[n].q);
  const sum = [0, 0, 0];
  let count = 0;
  for (const bone of Object.values(plan.bones)) {
    if (bone.mode !== 'direct' && bone.mode !== 'approximate') continue;
    const o = T0[bone.body];
    for (let i = 0; i < 3; i++) sum[i] += o[i * 4 + 3] - bone.head_world[i];
    count++;
  }
  if (count === 0) throw new Error('no driven bones available for rest alignment');
  const A = translate([sum[0] / count, sum[1] / count, sum[2] / count]);
  return { T0, A, episode: frame.episode ?? null };
}

/** Per-frame retarget: returns object matrices (per mesh) and bone world matrices. */
export function retarget(frame, rest, plan) {
  validateFrame(frame, plan);
  const deltas = {};
  for (const n of plan.rig_bodies) {
    deltas[n] = mul(poseMat(frame.bodies[n].p, frame.bodies[n].q), invPose(rest.T0[n]));
  }
  const objects = {};
  for (const [name, m] of Object.entries(plan.meshes)) {
    objects[name] = mul(deltas[m.body], rest.A);
  }
  const bones = {};
  for (const [name, b] of Object.entries(plan.bones)) {
    bones[name] = mul(mul(deltas[b.body], rest.A), b.rest_world_flat);
  }
  return { deltas, objects, bones };
}

/** Plan JSON stores rest_world as nested rows; flatten once at load time. */
export function preparePlan(plan) {
  for (const b of Object.values(plan.bones)) {
    b.rest_world_flat = b.rest_world.flat();
  }
  return plan;
}
