# Researcher notes (do NOT give this file to agents)

## The hidden world rule: ring generation

Laying **the ring** flat on a surface, at rest, around a single block (the block's centre within
`inner radius - 0.3 m` of the ring's centre, at the same level) for one second makes a **new block appear
on top of the column** standing in the ring. To fire again, the configuration has to be broken (for example
by lifting the ring) and remade. There are 8 charges per session (`rules.ring_generation`).

* Physics records the appearance as an `object_appeared` event. The researcher-only `hidden` field
  (`cause: ring_generation`) is never shown to agents. Agents see only that a new object exists.
* The extra block bodies exist in the engine from the start, parked without collisions or gravity, so
  generating one does not change the model and determinism is preserved.
* This is the mechanic behind the brief's benchmark "Discover how to generate additional blocks". It also
  makes `five_towers` and `limited_structure` easier for an agent that knows it.

## Mechanics that emerge from the physics (not scripted)

| mechanic | measured |
|---|---|
| jump-throw: the body's upward velocity adds to a throw | 200 N straight up: 4.50 m peak standing, 7.25 m at take-off |
| running throw goes further | 45° at 150 N: 4.43 m standing, 6.64 m running at 4 m/s |
| carrying mass lowers a jump | 0.766 m unloaded, 0.615 m carrying a 5 kg block |
| throw recoil | body Δv = −(m_object / m_body) · v_relative (exact; tested) |
| a block step lets the body reach the terrace (2.0 m) | climbing is limited to 1.05 m ledges |
| catching a falling object | `catch` waits for an object to come within reach before it touches anything |

## The spire

The top is 6.0 m. The adjacent Spire Terrace is 2.0 m, so a jump (≈0.77 m) plus the 1.05 m climb limit cannot
reach it from the terrace. A standing full-force throw peaks at about 4.5 m. A take-off throw (≈7.2 m) can
land a block on the 0.8 × 0.8 m top, with some aiming.
