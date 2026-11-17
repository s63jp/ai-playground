# Mission: Make AI Experimental Playground Deliver Everything Its README Claims

The README describes the intended product. Do not solve implementation gaps by watering down the vision, removing capabilities or replacing real behaviour with demonstrations.

Your task is to bring the existing repository up to the standard the README promises.

## 1. The required end state

Autonomous AI agents must inhabit a real MuJoCo-powered 3D island as articulated humanoids.

They must genuinely be able to:

### Physical body and movement
- Stand and maintain balance.
- Walk and run with stable, controlled locomotion.
- Accelerate, decelerate and change direction.
- Turn towards targets.
- Jump and land.
- Fall naturally.
- Detect when they have fallen.
- Recover and get back up through physical actuation.
- Navigate uneven terrain and obstacles.

### Object interaction
- Reach for objects.
- Grasp and release objects.
- Lift and carry objects.
- Push and pull objects.
- Throw objects with controlled force and direction.
- Catch moving objects.
- Interact with objects of different sizes, weights and physical properties.

### Combat and equipment
- Support multiple independently controlled humanoids in the same physical world.
- Provide equipment that agents can physically pick up, hold, carry and use.
- Support actual physical attacks and defensive actions.
- Implement equipment such as melee weapons, shields and throwable objects.
- Implement projectile-based equipment where appropriate.
- Detect real impacts through MuJoCo contacts.
- Calculate damage from physical interactions, equipment properties and impact conditions.
- Support health, incapacitation and match termination.
- Allow agents to make independent tactical decisions.
- Support duels and team-based encounters.
- Record and replay matches.

Equipment must not simply trigger an arbitrary damage event. It must interact with the physical world and produce verifiable physical consequences.

### Intelligence and learning
- LLM agents make independent high-level decisions.
- Physical controllers handle high-frequency movement and balance.
- Reinforcement learning can train and improve physical behaviours.
- Reusable physical skills can be developed and executed.
- Agents retain individual written memories across sessions.
- Previous experiences can influence later decisions.
- Experiments measure whether memory and learning actually improve performance.

### Simulation and evidence
- Preserve the existing 3D island and free-exploration environment.
- Support multiple agents sharing a world.
- Preserve individual agent identities and memories.
- Record actions and physical events.
- Maintain SHA-256-chained action logs.
- Provide replay and measurable experiment results.
- Preserve provider independence and the existing structured tool interface.

## 2. First task: establish the real implementation baseline

Before writing substantial code, inspect the repository.

Audit every README claim against:
- Actual implementation.
- Actual runtime behaviour.
- Existing tests.
- Real MuJoCo experiments.
- End-to-end integration.

Classify each capability:

1. Working and verified.
2. Implemented but unreliable.
3. Partially implemented.
4. Present as scaffolding only.
5. Missing entirely.

Identify dependencies between capabilities and produce an implementation sequence.

Do not mistake the presence of a module, class, CLI option or passing mock-based test for a working feature.

## 3. Implement in dependency order

Use independently verifiable milestones.

**Milestone 1 — Physics integrity and humanoid stability**

Resolve existing physics corruption, joint-control problems, fallen detection, locomotion and recovery.

Walking must be genuinely produced through the articulated body and its physical interaction with the environment. Do not fake movement using root velocity or position manipulation.

**Milestone 2 — Complete physical interaction**

Make reaching, grasping, lifting, carrying, pushing, pulling, throwing and catching work through real physics.

Verify object trajectories, contact events and physical outcomes.

**Milestone 3 — Multi-agent world**

Extend the existing world to support multiple independently controlled humanoids.

Preserve single-agent compatibility.

Each agent must have independent:
- Physical identity.
- Actuators and joint controls.
- Observations.
- Memory.
- Decision-making context.
- Action history.

**Milestone 4 — Equipment and combat**

Build actual physical equipment interactions.

Implement holding, attacking, defending, impacts, damage and incapacitation.

Support independently controlled opponents, duels and teams.

**Milestone 5 — Arena orchestration**

Connect the existing ArenaMatch architecture to real production world factories and session launchers.

Add the arena CLI and replayable match reports.

**Milestone 6 — Learning and integration**

Connect physical skills, reinforcement learning, persistent memory and measurable learning experiments.

Demonstrate that agents can improve through experience rather than simply claiming that learning is supported.

## 4. Architectural restrictions

- Extend existing systems rather than creating duplicate implementations.
- Preserve the provider-neutral architecture.
- Preserve the existing free-play experience.
- Respect existing session and ownership contracts.
- Keep high-frequency physical control separate from LLM reasoning.
- Do not use direct qpos/qvel writes to fake physical behaviour.
- Do not invent successful outcomes.
- Do not weaken tests to make implementations pass.
- Do not replace real dependencies with fake stubs.
- Do not introduce hardcoded tool counts.
- Preserve existing memory, persistence, event and replay systems.

If an existing implementation is defective, fix its underlying cause rather than building a second implementation around it.

## 5. Delegation and verification

Subagents may work on isolated components with explicit file ownership.

The coordinating agent must:
- Review integration boundaries.
- Resolve conflicting changes.
- Inspect actual diffs.
- Run real simulator experiments.
- Verify all claimed behaviours.
- Maintain the HVE evidence requirements.

Every milestone requires:
- Targeted tests.
- Real simulator verification.
- Relevant regression tests.
- Full-suite verification where appropriate.
- Recorded commands and exit codes.
- Documentation of remaining defects.

A feature is not complete because its implementation was written. It is complete when its intended behaviour has been demonstrated and verified.

## 6. Delivery requirements

Maintain a live implementation matrix matching every README claim to its implementation and verification evidence.

At the end of each milestone report:
- What now genuinely works.
- What changed.
- Which tests passed and failed.
- What was physically demonstrated.
- What remains incomplete.
- Any regressions or architectural risks.

Do not silently remove requirements or redefine success.

The ultimate acceptance test is the README vision working in the actual application: autonomous agents exploring a physical world, moving articulated bodies, manipulating objects, learning from experience, and competing against one another using physical equipment.

**Build the product described by the README. Do not rewrite the README to describe a smaller product.**