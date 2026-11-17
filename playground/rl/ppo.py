"""Genuine PPO for the locomotion environment (M6-B stage B3).

THE one RL training algorithm for `LocoEnv` (contract `rl_env/3`).
Actor-critic Proximal Policy Optimisation on CPU, trained exclusively from
real MuJoCo transitions collected through the B2 environment: every action
passes `LocoEnv.step` -> `PhysicsWorld.set_joint_target` (B1 protections
intact); reward is the ONE pipeline in `rl/rewards.py`; the task is driven by
the ONE curriculum in `rl/curriculum.py`. Nothing here reimplements any of
those: this module consumes the env's 5-tuple protocol and its
`episode_summary()` measurements.

Algorithm (documented equations — docs/RL_ENVIRONMENT.md §7 and
reports/m6b/B3_PPO_LEARNING.md):

Actor. Tanh-squashed diagonal Gaussian policy pi(a|s):
    z = mu(s) + sigma * eps,  eps ~ N(0, I),  a = tanh(z)
    log pi(a|s) = sum_dim [ -((z_d - mu_d)^2)/(2 sigma_d^2) - log sigma_d
                            - 0.5*log(2*pi) ]  -  sum_dim log(1 - a_d^2)
The last sum is the EXACT change-of-variables correction of the tanh squash
(da/dz = 1 - a^2); the pre-squash sample z is stored and the PPO ratio is
always new-vs-old log-prob of the SAME z, so the ratio is internally
consistent. sigma is a state-independent learnable parameter (per-dim
log_std), initialised at sigma = 1, clamped to [e^-5, e^2]. Deterministic
evaluation uses a = tanh(mu(s)); actions are in (-1, 1) by construction.

Critic. An INDEPENDENT value network V(s) -> R regressed on actual rollout
returns (never on the actor output).

GAE (time-limit correct). Per decision row i (float64):
    delta_i = r_i + gamma * (1 - terminated_i) * V(next_obs_i) - V_i
    adv_i   = delta_i + gamma * lambda * (1 - done_i) * adv_{i+1}
done_i = terminated_i or truncated_i; `next_obs_i` is the STORED continuation
state of that exact window, so a TRUNCATED row bootstraps from its own true
next state (time-limit correction) while the advantage chain is cut at every
episode boundary; a TERMINATED row (fall or fault) gets no bootstrap.
Returns R_i = adv_i + V_i. Advantages are normalised (mean 0, std 1) once
per rollout.

PPO-clip update (K epochs over the rollout, minibatched):
    ratio = exp(logpi_new(z) - logpi_old(z))
    L_pol = -mean( min(ratio*A, clamp(ratio, 1-eps, 1+eps)*A) )
    L_val = mean( (V(obs) - R)^2 )
    H     = sum_dim (0.5*log(2*pi) + 1 + log sigma_d)   (Gaussian entropy)
    loss  = L_pol + value_coef*L_val - ent_coef*H   (minimised)
H is the entropy of the PRE-SQUASH Gaussian — the standard regulariser for
squashed policies; the exact differential entropy of the squashed
distribution has no closed form (documented honestly). Gradients are
clipped by global norm per network; `target_kl` optionally aborts an update
early when the mean policy KL (k3 estimator) exceeds the target.

Faulted transitions (B1) never enter the data: a window flagged
`info['faulted']` is dropped entirely, the PREVIOUS valid row is marked
terminated (the suppressed physics produced no trustworthy successor state,
so its GAE bootstrap is void) and the episode ends there. A fall terminates
normally WITH its negative reward — a fall is a real physical result, not a
fault.

Seeding: one run seed drives network init and action sampling (torch RNG)
and the episode start-pose draws (`env.reset(seed=seed + episode_index)`).
Checkpoints store the torch + numpy RNG states for exact resume
(`rl/checkpoint.py`). CPU only by design in B3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Algorithmic constants, mirrored by the `rl.ppo` config block (config value
# wins, constant is the fallback — the B2 convention for the rl: block).
PPO_HYPER_DEFAULTS = {
    "rollout_steps": 2048,
    "epochs": 4,
    "minibatch": 512,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.0,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "hidden": (64, 64),
    "log_std_init": 0.0,
    "log_std_min": -5.0,
    "log_std_max": 2.0,
    "target_kl": None,
}

ACTION_EPS = 1e-6        # boundary guard inside log(1 - a^2)
LOG2PI = 1.8378770664093453


class PPOError(RuntimeError):
    """Invalid training input/data. Never silently ignored."""


def _torch():
    import torch
    return torch


# ==================================================================== networks
class TanhGaussianActor:
    """MLP trunk -> tanh-squashed diagonal Gaussian head over `action_dim`.

    The policy NEVER touches qpos/qvel or any world structure: it maps
    observations to bounded actions (and log-probs) only.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden=(64, 64),
                 log_std_init: float = 0.0, seed: int | None = None,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        torch = _torch()
        self.torch = torch
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        if seed is not None:
            torch.manual_seed(int(seed))   # BEFORE init so the init is seeded
        nn = torch.nn
        layers = []
        width = self.obs_dim
        for h in hidden:
            lin = nn.Linear(width, h)
            nn.init.orthogonal_(lin.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            width = h
        mu_head = nn.Linear(width, self.action_dim)
        # small output init: the initial mean stays near 0 so actions sit near
        # the neutral posture with NO immediate output saturation
        nn.init.uniform_(mu_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(mu_head.bias)
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.mu_head = mu_head
        # state-independent per-dim log-std (learnable, clamped at use time)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))

    # -------------------------------------------------- forward / interfaces
    def parameters(self):
        return list(self.trunk.parameters()) + list(self.mu_head.parameters()) \
            + [self.log_std]

    def state_dict(self):
        """Flat, checkpoint-stable form: 'trunk.<k>', 'mu_head.<k>', 'log_std'."""
        flat = {}
        for k, v in self.trunk.state_dict().items():
            flat[f"trunk.{k}"] = v.detach().cpu().clone()
        for k, v in self.mu_head.state_dict().items():
            flat[f"mu_head.{k}"] = v.detach().cpu().clone()
        flat["log_std"] = self.log_std.detach().cpu().clone()
        return flat

    def load_state_dict(self, sd):
        trunk_sd = {k[len("trunk."):]: t for k, t in sd.items()
                    if k.startswith("trunk.")}
        mu_sd = {k[len("mu_head."):]: t for k, t in sd.items()
                 if k.startswith("mu_head.")}
        self.trunk.load_state_dict(trunk_sd)
        self.mu_head.load_state_dict(mu_sd)
        with self.torch.no_grad():
            self.log_std.copy_(self.torch.as_tensor(sd["log_std"])
                               if not isinstance(sd["log_std"], self.torch.Tensor)
                               else sd["log_std"])

    def _forward(self, obs):
        """obs: torch tensor [B, obs_dim] -> (mu [B, A], sigma [B, A])."""
        h = self.trunk(obs)
        mu = self.mu_head(h)
        logstd = self.log_std.clamp(self.log_std_min, self.log_std_max)
        sigma = self.torch.exp(logstd)
        return mu, sigma

    # -------------------------------------------------- action distribution
    def _gaussian_logp(self, z, mu, sigma):
        tz = self.torch
        return (-0.5 * ((z - mu) / sigma) ** 2 - tz.log(sigma)
                - 0.5 * LOG2PI)

    def _squash_logp_correction(self, a):
        tz = self.torch
        return tz.log(1.0 - a * a + ACTION_EPS).sum(-1)

    def sample(self, obs_np: np.ndarray, deterministic: bool = False):
        """One decision: returns (action [-1,1]^A, pre-squash z, logp)."""
        tz = self.torch
        obs = tz.as_tensor(np.asarray(obs_np, dtype=np.float32)).reshape(1, -1)
        mu, sigma = self._forward(obs)
        if deterministic:
            z = mu
        else:
            z = mu + sigma * tz.randn(mu.shape)
        a = tz.tanh(z)
        logp = (self._gaussian_logp(z, mu, sigma).sum(-1)
                - self._squash_logp_correction(a.reshape(1, -1)))
        return (a.detach().numpy().reshape(-1),
                z.detach().numpy().reshape(-1),
                float(logp.item()))

    def evaluate_z(self, obs_batch, z_batch):
        """Training pass: NEW log pi(z) for stored pre-squash samples.

        Returns (logp [B], entropy [scalar]) — the same event z under the
        updated policy, exactly what the PPO ratio needs.
        """
        tz = self.torch
        obs = tz.as_tensor(np.asarray(obs_batch, dtype=np.float32))
        z = tz.as_tensor(np.asarray(z_batch, dtype=np.float32))
        mu, sigma = self._forward(obs)
        logp = (self._gaussian_logp(z, mu, sigma).sum(-1)
                - self._squash_logp_correction(tz.tanh(z)))
        logstd = tz.log(sigma)
        entropy = (0.5 * LOG2PI + 1.0 + logstd).sum(-1).mean()
        return logp, entropy

    def deterministic_action(self, obs_np: np.ndarray) -> np.ndarray:
        tz = self.torch
        obs = tz.as_tensor(np.asarray(obs_np, dtype=np.float32)).reshape(1, -1)
        with tz.no_grad():
            mu, _ = self._forward(obs)
            return tz.tanh(mu).numpy().reshape(-1)

    def sigma_now(self) -> np.ndarray:
        logstd = float(self.log_std.detach().numpy().mean())
        return float(math.exp(max(self.log_std_min, min(self.log_std_max, logstd))))


class Critic:
    """Independent value network V(s) -> R."""

    def __init__(self, obs_dim: int, hidden=(64, 64), seed: int | None = None):
        torch = _torch()
        self.torch = torch
        self.obs_dim = int(obs_dim)
        if seed is not None:
            torch.manual_seed(int(seed))   # BEFORE init so the init is seeded
        nn = torch.nn
        layers = []
        width = self.obs_dim
        for h in hidden:
            lin = nn.Linear(width, h)
            nn.init.orthogonal_(lin.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            width = h
        out = nn.Linear(width, 1)
        nn.init.orthogonal_(out.weight, gain=1.0)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def parameters(self):
        return list(self.net.parameters())

    def forward(self, obs):
        tz = self.torch
        if not tz.is_tensor(obs):
            obs = tz.as_tensor(np.asarray(obs, dtype=np.float32))
        return self.net(obs)

    def value(self, obs_np: np.ndarray) -> float:
        tz = self.torch
        obs = tz.as_tensor(np.asarray(obs_np, dtype=np.float32)).reshape(1, -1)
        with tz.no_grad():
            return float(self.net(obs).item())

    def values_batch(self, obs_batch) -> np.ndarray:
        tz = self.torch
        x = tz.as_tensor(np.asarray(obs_batch, dtype=np.float32))
        with tz.no_grad():
            return self.net(x).squeeze(-1).numpy()

    def state_dict(self):
        return dict(self.net.state_dict())

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)
        self.net.eval()


# ==================================================================== config
@dataclass
class PPOConfig:
    rollout_steps: int = 2048
    epochs: int = 4
    minibatch: int = 512
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: tuple = (64, 64)
    log_std_init: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    target_kl: float | None = None

    @classmethod
    def from_rl_cfg(cls, rl_cfg: dict | None) -> "PPOConfig":
        """Merge the `rl.ppo:` config block over PPO_HYPER_DEFAULTS."""
        cfg = dict(PPO_HYPER_DEFAULTS)
        cfg.update(dict((rl_cfg or {}).get("ppo") or {}))
        hidden = cfg["hidden"]
        if isinstance(hidden, (list, tuple)):
            hidden = tuple(int(h) for h in hidden)
        else:
            raise PPOError("rl.ppo.hidden must be a list of ints")
        if hidden and (any(h < 1 for h in hidden) or len(hidden) > 8):
            raise PPOError("rl.ppo.hidden must be 1..8 positive layer widths")
        for k in ("rollout_steps", "epochs", "minibatch"):
            if int(cfg[k]) < 1:
                raise PPOError(f"rl.ppo.{k} must be >= 1")
        if not (0.0 < float(cfg["gamma"]) < 1.0):
            raise PPOError("rl.ppo.gamma must be in (0, 1)")
        if not (0.0 < float(cfg["gae_lambda"]) <= 1.0):
            raise PPOError("rl.ppo.gae_lambda must be in (0, 1]")
        if int(cfg["minibatch"]) > int(cfg["rollout_steps"]):
            raise PPOError("rl.ppo.minibatch must be <= rollout_steps")
        return cls(rollout_steps=int(cfg["rollout_steps"]),
                   epochs=int(cfg["epochs"]),
                   minibatch=int(cfg["minibatch"]),
                   lr=float(cfg["lr"]),
                   gamma=float(cfg["gamma"]),
                   gae_lambda=float(cfg["gae_lambda"]),
                   clip_eps=float(cfg["clip_eps"]),
                   ent_coef=float(cfg["ent_coef"]),
                   value_coef=float(cfg["value_coef"]),
                   max_grad_norm=float(cfg["max_grad_norm"]),
                   hidden=hidden,
                   log_std_init=float(cfg["log_std_init"]),
                   log_std_min=float(cfg["log_std_min"]),
                   log_std_max=float(cfg["log_std_max"]),
                   target_kl=(None if cfg["target_kl"] is None
                              else float(cfg["target_kl"])))

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["hidden"] = list(d["hidden"])
        return d


# ==================================================================== GAE
def compute_gae(rewards: np.ndarray, values: np.ndarray, values_next: np.ndarray,
                terminated: np.ndarray, done: np.ndarray,
                gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation over a contiguous multi-episode buffer.

    delta_i = r_i + gamma*(1 - terminated_i)*V(next_obs_i) - V_i
    adv_i   = delta_i + gamma*lam*(1 - done_i)*adv_{i+1}

    `terminated` masks the bootstrap (a FALL or a FAULT: no trustworthy
    successor), `done` (terminated OR truncated) cuts the advantage CHAIN so
    one episode can never leak advantage into another; a truncated row keeps
    its bootstrap via delta_i (the time-limit correction). The LAST row may be
    mid-episode: its delta_i already carries the bootstrap from its stored
    next_obs, which is exactly how rollouts that end mid-episode are handled.
    Returns (advantages, returns); returns_i = adv_i + values_i.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    values_next = np.asarray(values_next, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=np.float64)
    done = np.asarray(done, dtype=np.float64)
    n = rewards.shape[0]
    if not (values.shape == values_next.shape == terminated.shape
            == done.shape == (n,)):
        raise PPOError("compute_gae: array length mismatch")
    adv = np.zeros(n, dtype=np.float64)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        delta = rewards[t] + gamma * (1.0 - terminated[t]) * values_next[t] - values[t]
        acc = delta + gamma * gae_lambda * (1.0 - done[t]) * acc
        adv[t] = acc
    returns = adv + values
    return adv, returns


def normalize_advantage(adv: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-std over the whole rollout (documented choice;
    fixed between epochs so the objective does not drift under the network)."""
    a = np.asarray(adv, dtype=np.float64)
    sd = a.std()
    if sd < 1e-8:
        return np.zeros_like(a)
    return (a - a.mean()) / sd


# ==================================================================== buffer
@dataclass
class RolloutBuffer:
    """Per-decision rollout data, sufficient to reproduce the PPO update.

    Rows are VALID decision windows only (faulted windows are dropped at
    collection; see module docstring). obs/next_obs float32[obs_dim],
    z float32[action_dim] (pre-squash sample), actions float32[action_dim]
    (post-squash, for diagnostics), logp/value float (old policy/critic).
    """
    obs_dim: int
    action_dim: int
    obs: list = field(default_factory=list)
    next_obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    z: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    terminated: list = field(default_factory=list)   # bootstrap cut
    done: list = field(default_factory=list)         # chain cut (term||trunc)
    prev_row: int | None = None                      # index of the live row
    # per-episode diagnostics collected alongside the transitions
    episode_summaries: list = field(default_factory=list)

    def begin(self):
        self.prev_row = None

    def add(self, obs, next_obs, action, z, logp, value, reward,
            terminated, done):
        idx = len(self.obs)
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.next_obs.append(np.asarray(next_obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.z.append(np.asarray(z, dtype=np.float32))
        self.logp.append(float(logp))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.done.append(bool(done))
        self.prev_row = idx

    def close_episode(self, mark_prev_terminated: bool, summary: dict | None):
        """End the current trajectory. Faulted windows: mark_prev_terminated
        True voids the previous row's bootstrap (physics suppressed)."""
        if mark_prev_terminated and self.prev_row is not None:
            self.terminated[self.prev_row] = True
            self.done[self.prev_row] = True
        if summary is not None:
            self.episode_summaries.append(summary)
        self.prev_row = None

    def finalize(self) -> "RolloutBuffer":
        n = len(self.obs)
        if n == 0:
            raise PPOError("empty rollout (all windows faulted?) — refusing to update")
        for name, arr in (("obs", self.obs), ("next_obs", self.next_obs),
                          ("z", self.z), ("actions", self.actions)):
            for row in arr:
                if row.shape != (self.obs_dim if name != "z" and name != "actions"
                                 else self.action_dim,):
                    raise PPOError(f"rollout row {name} wrong shape: {row.shape}")
            if name in ("obs", "next_obs", "z", "actions"):
                stacked = np.stack(arr).astype(np.float32, copy=False)
                if not np.all(np.isfinite(stacked)):
                    raise PPOError(f"non-finite rollout data in {name}; "
                                   "refusing to update (invalid physics would corrupt learning)")
                setattr(self, name, stacked)
        for name in ("logp", "values", "rewards"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise PPOError(f"non-finite rollout data in {name}; refusing to update")
            setattr(self, name, arr)
        t_arr = np.asarray(self.terminated, dtype=bool)
        d_arr = np.asarray(self.done, dtype=bool)
        # contract consistency: terminated implies done
        if np.any(t_arr & ~d_arr):
            raise PPOError("row marked terminated but not done")
        self.terminated = t_arr
        self.done = d_arr
        return self

    def __len__(self):
        return len(self.obs)


# ==================================================================== update
def ppo_update(actor: TanhGaussianActor, critic: Critic, buffer: RolloutBuffer,
               cfg: PPOConfig, iteration_seed: int = 0,
               opt_actor=None, opt_critic=None) -> dict:
    """One PPO optimisation pass over a finalised rollout buffer.

    The optimisers are caller-owned (the trainer keeps persistent Adam state
    across iterations — the standard PPO practice — so checkpoints can carry
    it for exact resume). When omitted (unit-test/stateless use), fresh Adam
    optimisers are constructed for this pass only. Raises PPOError on invalid
    data before any parameter step, leaving weights untouched.
    """
    torch = actor.torch
    if len(buffer) < 2:
        raise PPOError("rollout too small to update")
    values_next = critic.values_batch(buffer.next_obs)  # stored continuations
    adv, returns = compute_gae(buffer.rewards, buffer.values, values_next,
                               buffer.terminated, buffer.done,
                               cfg.gamma, cfg.gae_lambda)
    adv_norm = normalize_advantage(adv)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr) if opt_actor is None \
        else opt_actor
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr) if opt_critic is None \
        else opt_critic
    g = torch.Generator().manual_seed(int(iteration_seed))
    n = len(buffer)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0, "batches": 0}
    # optimisation epochs: at least one batch must run
    if cfg.epochs < 1:
        raise PPOError("ppo_update needs epochs >= 1")
    for _ in range(cfg.epochs):
        if cfg.target_kl is not None and stats["batches"] and \
                stats["approx_kl"] / stats["batches"] > cfg.target_kl:
            break
        perm = torch.randperm(n, generator=g).numpy()
        for start in range(0, n, cfg.minibatch):
            mb = perm[start:start + cfg.minibatch]
            logp_new, entropy = actor.evaluate_z(buffer.obs[mb], buffer.z[mb])
            logp_old = torch.as_tensor(buffer.logp[mb], dtype=torch.float32)
            adv_t = torch.as_tensor(adv_norm[mb], dtype=torch.float32)
            ret_t = torch.as_tensor(returns[mb], dtype=torch.float32)
            ratio = torch.exp(logp_new - logp_old)
            pol_unclipped = ratio * adv_t
            pol_clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps,
                                      1.0 + cfg.clip_eps) * adv_t
            policy_loss = -torch.min(pol_unclipped, pol_clipped).mean()
            value_loss = ((critic.forward(buffer.obs[mb]).squeeze(-1)
                           - ret_t) ** 2).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.ent_coef * entropy
            opt_a.zero_grad(set_to_none=True)
            opt_c.zero_grad(set_to_none=True)
            loss.backward()
            gn_a = torch.nn.utils.clip_grad_norm_(actor.parameters(),
                                                  cfg.max_grad_norm)
            gn_c = torch.nn.utils.clip_grad_norm_(critic.parameters(),
                                                  cfg.max_grad_norm)
            if not (torch.isfinite(loss) and torch.isfinite(policy_loss)
                    and torch.isfinite(value_loss)):
                raise PPOError("non-finite loss in update — weights untouched "
                               "(zero_grad already applied; step() never reached)")
            gn = float(max(gn_a.item() if torch.is_tensor(gn_a) else gn_a,
                           gn_c.item() if torch.is_tensor(gn_c) else gn_c))
            opt_a.step()
            opt_c.step()
            with torch.no_grad():
                kl = 0.5 * ((logp_new - logp_old) ** 2).mean()
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["approx_kl"] += float(kl.item())
            stats["clip_frac"] += float(((ratio - 1.0).abs()
                                         > cfg.clip_eps).float().mean().item())
            stats["grad_norm"] += gn
            stats["batches"] += 1
    if stats["batches"] == 0:
        raise PPOError("no optimisation batches ran (target_kl too tight?)")
    return {"policy_loss": stats["policy_loss"] / stats["batches"],
            "value_loss": stats["value_loss"] / stats["batches"],
            "entropy": stats["entropy"] / stats["batches"],
            "approx_kl": stats["approx_kl"] / stats["batches"],
            "clip_frac": stats["clip_frac"] / stats["batches"],
            "grad_norm": stats["grad_norm"] / stats["batches"],
            "batches": stats["batches"],
            "mean_adv": float(adv_norm.mean()),
            "std_adv": float(adv_norm.std()),
            "mean_return": float(returns.mean()),
            "mean_reward": float(buffer.rewards.mean()),
            "mean_value": float(np.asarray(buffer.values).mean())}


# ==================================================================== collection
def collect_rollout(env, actor: TanhGaussianActor, critic: Critic,
                    n_steps: int, seed: int, episode_counter: list,
                    summaries_cap: int = 512) -> RolloutBuffer:
    """Collect `n_steps` VALID decision windows from the real B2 env.

    Every physical effect happens through env.step (B1-protected PD path);
    episodes reset through env.reset(seed) (seeded pose jitter, B2 semantics).
    Faulted windows are dropped per the module docstring. Curriculum ticking
    is NOT done here (deliberate: the trainer advances stages explicitly
    through Curriculum.advance() at configured budgets — see PPOTrainer).
    `episode_counter` is a one-element list mutated across calls.
    """
    if len(episode_counter) != 1:
        raise PPOError("episode_counter must be a one-element list")
    buf = RolloutBuffer(obs_dim=actor.obs_dim, action_dim=actor.action_dim)
    buf.begin()
    obs = env.reset(seed=int(seed) + episode_counter[0])
    episode_counter[0] += 1
    while len(buf) < n_steps:
        value = critic.value(obs)
        action, z, logp = actor.sample(obs)
        obs_next, reward, terminated, truncated, info = env.step(action)
        faulted = bool(info.get("faulted"))
        if faulted:
            # B1: the window's physics was (partially) suppressed. The row is
            # excluded (never training data); the previous row loses its GAE
            # bootstrap (no trustworthy successor state) and the episode ends.
            buf.close_episode(mark_prev_terminated=True,
                              summary=dict(env.episode_summary()))
            obs = env.reset(seed=int(seed) + episode_counter[0])
            episode_counter[0] += 1
            continue
        done = bool(terminated) or bool(truncated)
        buf.add(obs, obs_next, action, z, logp, value, float(reward),
                bool(terminated), done)
        if done:
            buf.close_episode(mark_prev_terminated=False,
                              summary=dict(env.episode_summary()))
            obs = env.reset(seed=int(seed) + episode_counter[0])
            episode_counter[0] += 1
        else:
            obs = obs_next
    if len(buf.episode_summaries) > summaries_cap:
        buf.episode_summaries = buf.episode_summaries[-summaries_cap:]
    return buf.finalize()


def aggregate_summaries(summaries) -> dict:
    """Aggregate per-episode summaries into rollout-level statistics (the
    training-time view of the engine's own measurements)."""
    if not summaries:
        return {"episodes": 0}
    def col(key):
        return [s[key] for s in summaries if s.get(key) is not None]

    def signed_axis_disp(s):
        try:
            if s.get("start_position") is None or s.get("final_position") is None:
                return None
            tv = np.asarray(s.get("target_velocity") or (0.0, 0.0, 0.0), dtype=float)[:3]
            sp = float(np.linalg.norm(tv))
            if sp <= 1e-9:
                return None
            h = tv / sp
            d = np.asarray(s["final_position"], dtype=float) - \
                np.asarray(s["start_position"], dtype=float)
            return float(d[:3] @ h)
        except Exception:
            return None

    disp_axis = [x for x in (signed_axis_disp(s) for s in summaries) if x is not None]
    disp = col("net_displacement_m")
    prog = col("heading_progress_rewarded_m")
    upright = col("upright_fraction")
    effort = col("effort_mean_w")
    rew = col("reward_total")
    steps = col("steps")
    out = {
        "episodes": len(summaries),
        "steps": float(np.sum(steps)) if steps else 0.0,
        "falls": int(sum(col("falls"))),
        "faulted_windows": int(sum(col("faulted_windows"))),
        "reward_total_mean": float(np.mean(rew)) if rew else None,
        "net_displacement_mean": float(np.mean(disp)) if disp else None,
        "net_displacement_std": float(np.std(disp)) if disp else None,
        "axis_displacement_mean": float(np.mean(disp_axis)) if disp_axis else None,
        "axis_displacement_std": float(np.std(disp_axis)) if disp_axis else None,
        "progress_rewarded_mean_m": float(np.mean(prog)) if prog else None,
        "upright_fraction_mean": float(np.mean(upright)) if upright else None,
        "effort_mean_w_mean": float(np.mean(effort)) if effort else None,
        "truncation_rate": (float(np.mean([1.0 if s.get("truncated_at") else 0.0
                                           for s in summaries]))
                            if summaries else None),
    }
    # per-component reward sums (per-episode mean of the episode component
    # sums) — the reward-exploitation monitors (progress vs tracking balance,
    # fall penalty weight, effort share)
    comp_sums = {}
    for s in summaries:
        for k, v in (s.get("components") or {}).items():
            comp_sums.setdefault(k, []).append(float(v))
    for k, vals in comp_sums.items():
        out[f"comp_{k}_mean"] = float(np.mean(vals))
        out[f"comp_{k}_std"] = float(np.std(vals))
    return out


# ==================================================================== trainer
class PPOTrainer:
    """The training lifecycle: start fresh / resume, run, stop safely,
    report per iteration, checkpoint via the caller's hook.

    Stage progression: explicit. The trainer advances the B2 curriculum
    through `Curriculum.advance()` when a stage's configured decision budget
    is exhausted (stage_budgets maps stage-id -> decisions; budgets for the
    final ladder stage and beyond run forever at the final stage). This uses
    the curriculum's documented explicit-transition API; the stage ladder
    data itself is untouched. The per-decision auto-advance (`Curriculum.step`)
    is NOT ticked during training: its 400/1200/1600-decision demo timing is
    not a training schedule.
    """

    def __init__(self, env, actor: TanhGaussianActor, critic: Critic,
                 cfg: PPOConfig, *, seed: int = 0, total_steps: int = 1_000_000,
                 stage_budgets: dict | None = None, curriculum=None,
                 on_iteration=None, should_stop=None):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        torch = actor.torch
        # persistent optimizers (state lives in checkpoints for exact resume)
        self.opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
        self.opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
        self.seed = int(seed)
        self.total_steps = int(total_steps)
        self.stage_budgets = dict(stage_budgets or {})
        self.curriculum = curriculum
        self.on_iteration = on_iteration
        self.should_stop = should_stop
        self.total_valid_steps = 0
        self.iteration = 0
        self.episode_counter = [0]
        self._stage_steps = 0
        self.faulted_windows_total = 0
        self.rollout_rejections = 0

    # ------------------------------------------------------------ stage flow
    def _maybe_advance_stage(self):
        if not self.stage_budgets or self.curriculum is None:
            return None
        sid = self.curriculum.stage.id
        budget = self.stage_budgets.get(sid)
        if budget is None or self._stage_steps < int(budget):
            return None
        target_idx = self.curriculum.index + 1
        stages = self.curriculum.stages
        if target_idx >= len(stages) or not stages[target_idx].env_ready:
            return None  # final stage: run there forever
        self.curriculum.advance()          # documented explicit transition
        self._stage_steps = 0
        return stages[target_idx].id

    # ------------------------------------------------------------ main loop
    def train(self) -> dict:
        import time
        t_start = time.perf_counter()
        while self.total_valid_steps < self.total_steps:
            if self.should_stop is not None and self.should_stop():
                return {"stopped": True, "reason": "should_stop",
                        "total_valid_steps": self.total_valid_steps,
                        "iteration": self.iteration}
            t0 = time.perf_counter()
            buf = collect_rollout(self.env, self.actor, self.critic,
                                  self.cfg.rollout_steps, self.seed,
                                  self.episode_counter)
            steps = len(buf)
            roll = aggregate_summaries(buf.episode_summaries)
            self.total_valid_steps += steps
            self._stage_steps += steps
            self.iteration += 1
            try:
                metrics = ppo_update(self.actor, self.critic, buf, self.cfg,
                                     iteration_seed=self.seed + self.iteration,
                                     opt_actor=self.opt_a, opt_critic=self.opt_c)
            except PPOError:
                # invalid rollout data: refuse the update, keep the policy,
                # record it, carry on (never corrupt the parameters)
                self.rollout_rejections += 1
                metrics = {"rejected": True}
            else:
                self.rollout_rejections = 0
            if self.rollout_rejections > 2:
                raise PPOError(
                    f"{self.rollout_rejections} consecutive invalid rollouts — "
                    "stopping training (data pipeline is broken, not the policy)")
            wall = time.perf_counter() - t0
            self.faulted_windows_total += int(roll.get("faulted_windows") or 0)
            new_stage = self._maybe_advance_stage()
            rec = {
                "iteration": self.iteration,
                "total_valid_steps": self.total_valid_steps,
                "stage": (self.curriculum.stage.id if self.curriculum else None),
                "stage_transition_to": new_stage,
                "rollout_steps": steps,
                "episodes": roll.get("episodes"),
                "wall_s": round(wall, 3),
                "steps_per_s": round(steps / wall, 1) if wall > 0 else None,
                "sigma": self.actor.sigma_now(),
                "faulted_windows": roll.get("faulted_windows"),
                "update_rejected": bool(metrics.get("rejected")),
            }
            rec.update({k: v for k, v in metrics.items() if k != "rejected"})
            rec.update({f"roll_{k}": v for k, v in roll.items()})
            if self.on_iteration is not None:
                self.on_iteration(rec)
        return {"stopped": False, "total_valid_steps": self.total_valid_steps,
                "iteration": self.iteration}

    # ------------------------------------------------------------ rng/serial
    def rng_state(self) -> dict:
        torch = self.actor.torch
        return {"torch_rng": torch.get_rng_state().numpy().copy(),
                "episode_counter": int(self.episode_counter[0]),
                "total_valid_steps": self.total_valid_steps,
                "iteration": self.iteration,
                "_stage_steps": self._stage_steps,
                "faulted_windows_total": self.faulted_windows_total}

    def load_rng_state(self, st: dict) -> None:
        """Restore the trainer's counters. NOTE: the torch RNG itself is
        restored by checkpoint.apply_rng (the checkpoint stores it base64-
        encoded; the in-memory form only exists inside PPOTrainer.rng_state)."""
        self.episode_counter = [int(st.get("episode_counter", 0))]
        self.total_valid_steps = int(st.get("total_valid_steps", 0))
        self.iteration = int(st.get("iteration", 0))
        self._stage_steps = int(st.get("_stage_steps", 0))
        self.faulted_windows_total = int(st.get("faulted_windows_total", 0))

