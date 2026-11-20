"""Live Neural Training mode (M6-C foundations).

Watch a genuinely untrained policy learn to walk through the ONE authoritative
stack in front of the user:

  frames.py     read-only extraction of MuJoCo state into viewer packets
  tap.py        the read-only env tap (the trainer's env is LocoEnv wrapped in
                FrameTapEnv — physics and algorithm are untouched)
  telemetry.py  progress.jsonl writers/readers (own runs + read-only B3 tail)
  interop.py    the single seam to the B3 trainer modules (lazy imports)
  session.py    the live session state machine (start/pause/resume/stop/save)
  bus.py        SSE fanout (latest-wins frames, bounded event queues)
  rest.py       the FastAPI mount for live training on the dashboard server
  rest_playback.py  the FastAPI mount for checkpoint playback on that server
  playback.py   PlaybackSession: a saved brain drives the sim (no PPO updates)

The live viewer and checkpoint playback are served by the ONE dashboard server
(`playground serve`) via the mounts above — there is no separate sidecar
process. Nothing here trains anything by itself: a live training run only starts when
the user asks for it; it uses the real LocoEnv + PPOTrainer + checkpoint
format, in this process, writing only under its own run directory.
"""
