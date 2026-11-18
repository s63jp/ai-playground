# Anthropometric + gravity-curriculum investigation — audit and design (roadmap)

Jamie roadmap item 2026-11-18 (extended mid-session with the head design
caution and the perturbation stage), held SEPARATE from the running
`b4_posture1` (no interruption; read-only audit done alongside, training
lineage deferred until the b4_posture1 gate and Jamie's sequencing call).
Jamie's thesis, recorded: use the physics environment to teach the agent to
handle its own body, rather than continually compensating for poor posture
through reward engineering — the agent should learn to control a physical
body, not exploit a reward formula; and the approach must generalize to other
agents (creatures, robots, vehicles) with their own physical characteristics
and training environments.

## 1-2. Mass audit: the rig vs human proportions (measured this session)

Script `ai-playground-audit/b4_mass_audit.py` → `b4_mass_audit.json`
(compiled-model read, world scene props excluded: 20 island blocks + rock +
ring ≈ 114 kg are environment bodies, not the agent):

**Humanoid total: 47.05 kg** (≈ 1.6 m figure).

| group | kg | % of humanoid | human reference (Winter literature %) | reading |
|---|---|---|---|---|
| root column (`agent` slide-root body) | 10.0 | **21.3 %** | — none | artificial: a tenth of the body's mass is actuated root structure |
| pelvis/torso box | 7.0 | 14.9 % | pelvis ≈ 14.2 % | ✓ on proportion |
| chest/upper trunk (`spine_L2`) | 7.07 | **15.0 %** | abdomen+thorax ≈ 35.5 % | **~42 % of human proportion — the upper trunk is under-massed** |
| head+neck | 1.8 (B4 fix) | 3.8 % | ≈ 8.1 % | **~half proportion** (deliberate B4 safety choice) |
| arms (both) | 5.30 | 11.3 % | ≈ 10.0 % | ✓ near proportion |
| legs (both) | 15.88 | 33.7 % | thigh+shank+foot ≈ 32.2 % | ✓ near proportion; per-segment: thigh 9.2 %, shank 5.3 %, foot+ankle 2.4 % of body |

Findings (measured):

1. The leg/pelvis//arm proportioning is broadly human-like — the B4 head fix
   brought the head within reach and the legs carry human-plausible mass.
2. The upper trunk (chest) at 15 % vs human 35.5 % concentrates too little
   mass above the pelvis — combined with the 10 kg root column, the rig's
   vertical inertia distribution is less top-heavy than a human's. One
   measured consequence: dynamic pitching of the chest (the spine_L2
   hierarchy) carries less momentum than a human's would.
3. The head at 3.8 % is HALF human proportion — and the B4 measurement shows
   a full human-proportion head is mechanically HOLDABLE on the current neck:
   ~3.8 kg @ 0.13-0.19 m com ⇒ gravity moment ≈ 4.8-7.1 N·m vs the kp=8
   neck's measured 12.57 N·m capacity (the original failure was the 6.37 kg
   head's COM at +0.26 m ⇒ 16.25 N·m). Restoring human head proportion is
   therefore FEASIBLE without regressing the B4 fix — a candidate
   anthropometric correction for this lineage (Jamie decision; it is an
   asset change and gets its own lineage separation).
4. Trunk+head above pelvis ≈ 30.9 % (human ≈ 57.8 %) — the most fundamental
   anthropometric divergence; any balance/posture difficulty on this rig is
   structurally easier than the human problem. Worth stating plainly: this
   rig is NOT a human analogue in its inertia distribution, and gravity
   curriculum cannot fix that — segment mass is an asset-layer property.

## 2b. The head as a coupled mass + COM + inertia design space (Jamie's caution)

Jamie's caution honoured explicitly: the 6.37 kg head was not wrong merely by
mass — its COM (+0.26 m above the hinges) AND inertia made the neck's
measured 12.57 N·m capacity (kp=8, full range) the binding constraint, and
the B4 fix chose a light head to solve mechanical stability. Whether 1.8 kg
@ +0.13 m is an appropriate LONG-TERM physical model is a separate question
from whether it was a good stability fix. The design space, measured/analytic
(ALL rows must be neutral-gate probe verified before any lineage — the
analytic numbers are estimates from m·g·d and Winter segment tables):

| head model | mass | COM above hinge | diaginertia | hinge-gravity moment | % of neck capacity (12.57 N·m) | hinge inertia (I + m·d²) |
|---|---|---|---|---|---|---|
| original (measured, UNSTABLE) | 6.37 kg | +0.26 m | ≈ 0.036 | **16.25 N·m** | **129 %** | 0.512 kg·m² |
| B4 fix (measured, holds) | 1.8 kg | +0.13 m | 0.010/0.011/0.009 | 2.30 N·m | 18 % | 0.0404 kg·m² |
| human-proportion candidate A | ≈ 3.8 kg (8.1 % of 47 kg) | +0.13 m (atlas-adjacent) | scaled ≈ 0.023 (Winter) | ≈ 4.85 N·m | ≈ 39 % | ≈ 0.087 kg·m² |
| same mass, original COM | 3.8 kg | +0.26 m | — | ≈ 9.70 N·m | ≈ 77 % — marginal, no dynamic margin | ≈ 0.32 kg·m² |

Readings: a human-MASS head at a realistic (atlas-adjacent) COM holds with
~60 % margin and roughly doubles the head hinge inertia (note: that ALSO
moves the Issue-#2 synthetic-QACC threshold further from reach — the
light-head hinge inertia drop is what narrowed it). A human-mass head at the
ORIGINAL +0.26 m COM does NOT have a dynamic margin — mass and COM must be
chosen together. The long-term physical model should therefore be a
**human-proportion head at a human-realistic COM with Winter-scaled inertia**,
verified by the same neutral-gate + hostile-replay probes the B4 fix used.
The same coupled treatment applies to the chest (mass, COM, inertia) when
correcting the trunk toward proportion.

## 3. Gravity configuration options (inspected)

- MuJoCo: `opt.gravity = (0, 0, −9.81)` single 3-vector; no per-body gravity.
- RL env: base gravity captured once at build (`env.py:99`); curriculum
  phases REWRITE gravity from that captured base — `set_phase` with
  `gravity_scale` (absent → leave as-is) is non-cumulative by design
  (the pre-M6 bug of re-multiplying gravity is documented in curriculum.py);
  `reset()` re-applies the current scale (`env.py:306`).
- All `DEFAULT_STAGES` phases already carry `gravity_scale: 1.0` — the knob
  is live, exercised, and bounded-non-cumulative.
- Combat/trajectory code reads a separate `world.gravity` config (projectile
  ballistics) — independent of the RL env's MuJoCo gravity path.

## 4. Progressive gravity training: feasibility = YES, contract unchanged

`gravity_scale` is already a per-stage curriculum parameter applied through
the measured `set_phase` seam — a gravity ladder is curriculum PHASE DATA
(the exact B3 stage-direction-correction pattern: data correction with
measured evidence, no obs/action change, contract version untouched,
checkpoint fingerprints unchanged). Design sketch (for Jamie's call):

- Ladder options: (a) ride the existing stage ladder — each stage's phase
  carries its own `gravity_scale` (standing @ reduced g → controlled →
  sustained @ 1.0), so gravity eases with task difficulty; (b) a dedicated
  gravity dimension advanced on its own measured schedule inside the standing
  stage, then full-g walking stages (cleaner attribution: gravity eases
  BALANCE learning specifically, where the measured posture failures live).
- **Jamie's recommended 4-step ladder (mid-session addition) fits the
  curriculum's reserved stages**: reduced g (learn basic balance/posture with
  less load) → intermediate g (support increasing load) → normal 9.81 m/s²
  (demonstrate learned posture under realistic conditions) → perturbations
  (recovery without falling). Steps 3 and 4 map onto the existing sustained
  stage and the curriculum's RESERVED disturbance stage (stages 4-6 sit in
  DEFAULT_STAGES with env_ready=False awaiting enablement — the machinery
  Jamie's step 4 needs is already staged, unimplemented content only).
- Jamie's caution honoured in the acceptance design: reduced gravity changes
  contact forces and movement dynamics — success at 0.5 g has NOT
  automatically learned 1 g standing. The final evaluation is AT 1 g.
- Items the design phase must measure before a lineage starts (all engine
  measurements, no new machinery): fall-rule semantics under sub-g (the
  root-height threshold is absolute; up-vector cosine is g-free — verify
  standing/fall behaviour at 0.5-0.8 g), locomotion target semantics
  (v* stays m/s so sub-g walking is dynamically easier — expected and
  intended; effort in joules scales with weight, documented), and the
  standing-equilibrium arithmetic at reduced g.
- Evaluation AT 1g is the acceptance criterion (see 5): the ladder anneals
  to `gravity_scale = 1.0` before the trained checkpoint is judged, so any
  sub-g shortcut is exposed at the matched held-out eval.

## 5. Evaluation at 1g (acceptance plan)

Matched held-out harness, unchanged conditions: sustained 12 ep + standing
12 ep, seeds 97010000+, `--episodes 12`, run-dir outputs, on the 1.0-g phase;
per-checkpoint probes during training measure each g-step of the ladder;
acceptance compares against the b4_posture1 and b4_head1 rows (chest-up/head-
up %, completion, falls, displacement/speed, standing drift, components +
entropy, faults + rejections). A cross-g sweep (eval at 0.6/0.8/1.0 g) is a
cheap extra generalization readout worth adding — it directly measures
"controls a physical body" vs "specialised to one g".

## 6. Generalization (Jamie's framing)

The gravity seam is body-agnostic: any future agent (creature, robot,
vehicle) gets its own rig + curriculum, and the same phase-data mechanism
applies — no per-agent gravity machinery needed. What DOES generalise as
method: (1) audit the asset's mass/inertia against the real referent FIRST
(this document's table), (2) correct segment properties in the asset layer
(lineage-separated, measured), (3) use curriculum phase data (gravity,
disturbance, terrain) rather than reward surgery when the goal is
"learn the body", (4) evaluate at the target condition (1 g) with matched
held-out harnesses.

## Sequencing — Jamie's decisions (2026-11-18, mid-session)

**Milestone structure — one new experimental lineage set, phases independently
measurable, as a SEPARATE milestone after b4_posture1** (b4_posture1 finishes
unchanged; this could become a foundational Neural-Learning capability:
learning control of a body under different physical conditions, not one
walking animation):

- **Baseline**: existing B4 head-fixed rig at 1 g (the b4_posture1/b4_head1 rows).
- **Phase A**: anthropometric corrections trained AT 1 g — the direct control.
- **Phase B**: gravity curriculum on the corrected rig.
- **Final evaluation**: at 1 g, including **the corrected-rig-trained-directly-
  at-1g comparison** — required, else gravity progression's contribution is
  unmeasurable. Three-way readout: baseline (old rig), corrected @ 1 g
  (Phase A), corrected + gravity ladder (Phase B).

**Gravity ladder (Jamie's table)**:

| stage | gravity | objective |
|---|---|---|
| 1 | 0.6 g | learn basic posture and balance |
| 2 | 0.8 g | stability under greater loading |
| 3 | 1.0 g | normal-gravity control |
| 4 | 1.0 g | **disturbances** — only AFTER normal-gravity standing and locomotion are demonstrated |

The 0.6/0.8/1.0 g evaluation sweep is a **separate eval-time readout, never
part of the training curriculum**.

**Safeguard before the lineage exists** (the head-mechanics lesson, made a
gating requirement): neutral-gate test ALL anthropometric changes first —
segment masses and COMs, segment inertias, actuator capacity against the
revised loads, contact behaviour and physics warnings, standing equilibrium
AT EACH gravity level (the B4 neutral-gate + inertial-verify + hostile-replay
methodology exists and is the template; nothing ships from an analytic table).

## Naming proposal (for the stage contract when this starts)

Realistic-humanoid + gravity-curriculum milestone — candidate name
**M6-B7 "body under conditions"**, after b4_posture1's gate and before/along
whatever .aipg scheduling Jamie wants (B5/B6 remain planned; not re-litigated
here). Audit artifacts: `ai-playground-audit/b4_mass_audit.py` →
`b4_mass_audit.json` (per-body masses, inertias, COMs; scene vs humanoid
separation; Winter reference constants); the standing-invariance and
neutral-gate methodology: `b4_neutral_test.py`, `b4_head_inertial_verify.py`,
`b4_posture1_standing_check.py`.
