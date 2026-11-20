// Node parity test: JS retarget port must reproduce the Python outputs (fixture is
// generated from real LocoEnv frames by playground/rl_live/humanoid_v2/export_viewer_assets.py).
// Run: node --experimental-default-type=module tests/js/humanoid_v2_math.test.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import { preparePlan, captureRest, retarget, validateFrame } from '../../playground/dashboard/static/js/humanoid_v2_math.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');
const fixture = JSON.parse(readFileSync(path.join(root, 'tests/fixtures/humanoid_v2_viewer_parity.json'), 'utf8'));
const plan = preparePlan(JSON.parse(readFileSync(
  path.join(root, 'playground/dashboard/static/assets/humanoid_v2/retarget_plan.json'), 'utf8')));

function maxAbsDiff(a, b) {
  assert.equal(a.length, 16, 'matrix must have 16 entries');
  let m = 0;
  for (let i = 0; i < 16; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// the live payload nests the body map under `bodies`; the fixture stores that map as-is
const wrap = (c) => ({ bodies: c.bodies, episode: 0 });
const reset = fixture.cases[0];
const rest = captureRest(wrap(reset), plan);
let worst = 0;
let checked = 0;
for (const c of fixture.cases) {
  const out = retarget(wrap(c), rest, plan);
  for (const [n, exp] of Object.entries(c.expected.deltas)) worst = Math.max(worst, maxAbsDiff(out.deltas[n], exp.flat()));
  for (const [n, exp] of Object.entries(c.expected.object_matrix)) {
    const d = maxAbsDiff(out.objects[n], exp.flat()); worst = Math.max(worst, d); checked++;
  }
  for (const [n, exp] of Object.entries(c.expected.bone_world)) worst = Math.max(worst, maxAbsDiff(out.bones[n], exp.flat()));
  assert.ok(worst < 1e-9, `${c.label}: parity drift ${worst}`);
}
assert.equal(Object.keys(plan.meshes).length, 35, 'plan must carry 35 meshes');
assert.equal(checked, 35 * fixture.cases.length, 'every mesh checked in every case');
console.log(`PARITY ok: ${fixture.cases.length} frames, ${checked} mesh matrices, max abs diff ${worst.toExponential(2)}`);

// fail-closed checks
const bad = wrap(JSON.parse(JSON.stringify(reset)));
delete bad.bodies.torso;
assert.throws(() => validateFrame(bad, plan), /missing=\[torso\]/);
const extra = wrap(JSON.parse(JSON.stringify(reset)));
extra.bodies['agent_001:torso'] = extra.bodies.torso;
assert.throws(() => validateFrame(extra, plan), /unexpected=\[agent_001:torso\]/);
const nan = wrap(JSON.parse(JSON.stringify(reset)));
nan.bodies.head.p[2] = NaN;
assert.throws(() => validateFrame(nan, plan), /non-finite/);
console.log('FAIL-CLOSED ok: missing, prefixed/unexpected and non-finite frames rejected');

// no accumulation: 500 repeated applications of the same frame give identical output
const first = retarget(wrap(fixture.cases[2]), rest, plan).objects;
let drift = 0;
for (let k = 0; k < 500; k++) {
  const again = retarget(wrap(fixture.cases[2]), rest, plan).objects;
  for (const n of Object.keys(first)) drift = Math.max(drift, maxAbsDiff(again[n], first[n]));
}
assert.equal(drift, 0, 'repeated frames must not drift');
console.log(`NO-DRIFT ok: 500 repeats, max diff ${drift}`);
