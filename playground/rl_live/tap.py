"""FrameTapEnv — a read-only tap between the trainer and the authoritative env.

PPOTrainer and collect_rollout interact with the env exclusively through
`reset(seed)`, `step(action)` and `episode_summary()`. The tap forwards every
one of those calls UNCHANGED to the real LocoEnv and, after each call,
packs a live frame / event through `playground.rl_live.frames` and hands it
to a sink (the server's broadcast bus).

Contract guarantees:
- the tap never mutates the env, the action, the seed or the 5-tuple;
- a failing sink or a failing frame-packer can NEVER break training: both
  are isolated (counter-guarded) and swallowed;
- episodes/decisions counted here are for display only — the trainer's own
  `total_valid_steps` stays the single source of truth.
"""

from __future__ import annotations

import time

from .frames import FrameIndex, pack_episode_event, pack_frame, pack_reset_event


class RecordingSink:
    """Collects messages in a list — test helper / offline capture."""

    def __init__(self):
        self.messages: list[dict] = []

    def __call__(self, msg: dict) -> None:
        self.messages.append(msg)


class FrameTapEnv:
    """Proxy: `reset` / `step` / `episode_summary` explicit, everything else
    delegated via __getattr__."""

    def __init__(self, env, sink=None):
        self._env = env
        self._sink = sink or (lambda msg: None)
        self._index = FrameIndex(env)
        self._episode = 0
        self._decision = 0
        self._live_frames = 0
        self._dropped = 0
        self._ended = False           # episode_end already emitted this episode
        self._last_error_t = 0.0

    # ------------------------------------------------------------- tapped
    def reset(self, seed=None):
        obs = self._env.reset(seed)
        self._episode += 1
        self._decision = 0
        self._ended = False
        try:
            self._sink(pack_reset_event(self._env, episode=self._episode, seed=seed))
        except Exception:
            self._drop()
        return obs

    def step(self, action):
        result = self._env.step(action)
        obs, reward, terminated, truncated, info = result
        self._decision += 1
        try:
            frame = pack_frame(self._env, action=action, reward=reward,
                               terminated=terminated, truncated=truncated,
                               info=info, episode=self._episode,
                               decision=self._decision, index=self._index)
            self._sink(frame)
            self._live_frames += 1
        except Exception:
            self._drop()
        if (terminated or truncated) and not self._ended:
            # once per episode: the env keeps reporting truncated after the
            # cap (it is the TRAINER that resets), but the episode ended
            self._ended = True
            try:
                self._sink(pack_episode_event(self._env, episode=self._episode,
                                              terminated=terminated,
                                              truncated=truncated))
            except Exception:
                self._drop()
        return result

    def episode_summary(self):
        return self._env.episode_summary()

    # ------------------------------------------------------------ passthrough
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def live_frames(self) -> int:
        return self._live_frames

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    @property
    def episode_count(self) -> int:
        return self._episode

    def _drop(self):
        self._dropped += 1
        now = time.time()
        if now - self._last_error_t > 60.0:
            self._last_error_t = now
            try:
                self._sink({"t": "tap_note", "dropped_frames": self._dropped})
            except Exception:
                pass
