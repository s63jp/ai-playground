# Action-Space Audit — AI Experimental Playground
_Audit date: 2026-11-13. Source of truth: live code + the real action log (4,443 agent/program actions across 106 sessions)._

## 1. What already exists (full schema: 40 tools, 7 categories)

**observation (7):** observe_world, inspect_object, inspect_position, get_inventory, get_world_boundaries, get_object_positions, look_around
**movement (5):** move (walk straight line), walk_to, jump, climb, wait
**object (6):** pick_up, put_down, throw, place, rotate_object, catch
**experiment (6):** measure_height, measure_distance, measure_velocity, measure_flight_time, inspect_collision, run_physics_test (private sandbox)
**world/programs (5):** inspect_world_history, inspect_recent_events, save_program, execute_program, list_programs
**memory (11):** read_memory, write_diary, add_note, record_experiment, record_discovery, update_discovery, record_mistake, set_goal, update_goal, assign_meaning, claim_achievement, end_session
**learning (5):** nn_list, nn_make_dataset, nn_train, nn_predict, nn_evaluate

## 2. Real usage frequencies (agent/program actions only)

| tool | calls | success | success-rate | note |
|---|---|---|---|---|
| move | 1078 | 816 | 76% | dominant |
| walk_to | 663 | 575 | 87% | |
| pick_up | 652 | 287 | **44%** | low success — agents struggle to grab |
| look_around | 440 | 439 | 100% | |
| throw | 196 | 146 | 74% | |
| place | 171 | 40 | **23%** | very low success |
| climb | 116 | 15 | **13%** | mostly fails |
| put_down | 100 | 90 | 90% | |
| jump | 78 | 73 | 94% | |
| rotate_object | 1 | 0 | 0% | essentially unused |
| catch | 6 | 0 | 0% | never succeeds |

**Key finding:** throwing is NOT the most-used verb — movement + pick_up dominate. The "throw-dominance" the user feels is in the **learning + benchmark layer**, not the raw action menu: the only NN target with a controller/task is `max_height`→`throw_height`. Construction verbs (stack/wall/platform/bridge) have **no first-class tools** — they're emergent from pick_up+place, which has a 23% success rate, so construction barely happens.

## 3. What's genuinely missing (no primitive at all)

- **Manipulation without lifting:** push, pull, slide, nudge, kick, roll. The engine has no "apply force to an object you are not holding" primitive. `throw` requires holding; `rotate_object` needs reach.
- **Construction intents:** stack_on, build_wall/row, build_bridge, dismantle. All must be synthesised from place, which is unreliable.
- **Stability testing:** no "poke it and see if it falls" / structural-stability probe (only passive observe + collision log).
- **Locomotion nuance:** run vs walk is just the `speed` arg (exists); step-back/side-step are just `move` with a direction (exist but unnamed → rarely chosen). No true "turn": the body is a box on **3 slide joints (x,y,z)** — `facing` is a cosmetic heading derived from motion, not a controllable DOF. So "turn" is not physical; don't fake it.

## 4. Object inventory (what physics can actually do)
- 12 blocks (+8 latent reserve), 1 **sphere** ("rock", rollable), 1 **torus** ("ring"), 4 fixed platforms, 1 spire. Island ±12 m.
- The sphere/torus make **roll** physically meaningful — currently no tool exercises rolling.

## 5. Architectural changes needed
1. **Physics primitives** in `world/physics.py`: `apply_push(oid, direction, force)` (impulse to an unheld object in reach), `nudge`, and reuse for `kick`/`roll`. Must emit events + records like throw does, so verifiers & NN datasets pick them up.
2. **Tools** in `interface/physical.py` + `interface/schema.py`: push, pull, slide, roll, kick, and construction helpers (stack_on, build_row) that are thin, discoverable wrappers — NOT hardcoded strategies.
3. **Feature/target expansion** (`ml/features.py`): extract rows for push/roll outcomes (distance moved, final rest position, toppled?) so networks can learn non-throw dynamics. Add controller tasks beyond throw_height: move_to_target, topple_tower, roll_to_target, build_to_height.
4. **Action-frequency recording** (requirement #7): per-session behaviour metrics already exist (self_directed_goals etc.); add a per-tool histogram to the behaviour record + a dashboard "action diversity" view so diversification is visible.
5. **Prompt** (`agents/prompts.py`): present the richer menu neutrally; discourage defaulting to throw; encourage action *combinations* (carry→stack→reposition→test).

## 6. Guardrails (per user requirements)
- No hardcoded behaviours — tools are primitives; agents discover technique.
- Every new action gets real physics, state transitions, events, and HVE tests (Happy/Validate/Error) before it counts as done.
- Verify diversification with the action histogram, not by assertion.
