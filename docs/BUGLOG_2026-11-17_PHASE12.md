# BUGLOG — 2026-11-17 (Phase 12: Autonomous Physical Agent Integration)

Entries from the agent-facing physical integration work
(`docs/AGENT_PHYSICAL_INTEGRATION.md`). Every entry states the bug, the live
evidence, and the fix — and what the rig physically does now.

## 1. `ManipulationController.reach` drove joints in the wrong unit

**Bug.** `reach()` passed DEGREES into `set_joint_target` (radians): the shoulder
target 15.0 meant 15 radians and the PD slammed the arm toward its far limit.

**Evidence (live probe).** After `mc.reach()` the left shoulder read 1.759 rad
(~100 deg) with a requested "15 deg" — the old "reached" status was returned
unconditionally, which is exactly the "success because the function returned"
anti-pattern. The old `test_reach_and_grasp_continuous` passed only because the
old reach registered PD targets without stepping the sim, so `grasp` measured the
distance at the *unmoved* arm before the flail began.

**Fix.** 2-link sagittal-plane IK in RADIANS (law of cosines; straight-arm aim
when the target sits at/beyond full extension so the forearm never ploughs
through the target), gains kp 50/35, a 2.5 s settle, and measured verification:
`out_of_reach`/`reach_untracked` errors, `requested_*_rad` vs measured joint
angles in the payload. `reached` is only claimed when the arm tracked the request.

**Rig facts.** An outstretched arm is 0.54 m from the shoulder; a ground-level
object 0.75 m ahead leaves the hand ~0.4 m short — the measurement is the point,
not a promised grip. The strict `mc.grasp` 0.07 m hand gate is unsatisfiable for
0.5 m cubes near the body; the honest grip for agents is the contact-gated
`world.pick_up`, so that is what the `grasping`/`lifting` skills dispatch.

## 2. A leftover 'stand' gait controller steers the next walk off course

**Bug (introduced while fixing #1).** Bracing the figure with `set_gait("stand")`
during the reach settle and leaving it installed corrupted the next walk.

**Evidence (live probe, same rig).** stand-brace 4 s → walk east 1.2 m: the rig
strolled SOUTH-WEST (displacement +0.23/+0.43 for an EAST request) and honestly
verdicted "blocked by unknown obstacle" after 0.41 m (progress on the walk axis
was negative). With the controller restored to the state found before the brace,
the identical sequence walks 1.465 m.

**Fix.** `reach()` records whether a gait controller existed and nulls the one it
created after the measured settle — the world is left exactly as it was found.

## 3. Reach geometry vs a spawned block: straight-arm instead of elbow sweep

**Evidence.** With a block at the hand tip, a law-of-cosines elbow bend swept the
forearm through the block (ejecta: the block was displaced 174 m in one probe).
The rig now aims the straight arm at at-length/beyond-length targets and reports
the measured shortfall.

**Rig facts for fixtures.** 0.5 m cubes spawned with their centre at the arm tip
intersect the ground or the forearm capsule and get solver-ejected (z=82 m);
spawn them ground-floored and clear of the 0.44 m body and the arm sweep path.
A 12 kg rock rolls when walked into (300 N walk force vs rolling contact), so
walking-into-obstacle tests cannot rely on shovable objects stalling the walk.

## 4. No in-place turn exists (design fact, not a bug)

`set_gait("stand", heading=X)` holds zero waist forever (probe: torso yaw 0.0 deg,
waist 0.002 rad after 8 s). The heading servo lives only in the walk cycle's
`_gait_targets` — turning rides the contact-driven stride (waist twist against
planted-foot friction). `turn_to` therefore walks a short real arc at the
requested heading and reports measured facing, waist angle and displacement.
A stand-mode heading servo would have been a second locomotion implementation —
forbidden by the integration contract.

## 5. Fall detection is sustained-condition logic

`agent_fallen()` requires the low/up-vector condition to persist 0.3 s; the fold
thrash makes it flicker, so fall-invariant helpers must track it per step (like
`test_locomotion` does), never assert it at a single endpoint. A brief stand phase
BEFORE the leg fold registers the posture PD targets; without it the fold crumples
past the recoverable sit pose into a wedged floor collapse.
