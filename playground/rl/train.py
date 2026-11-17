
import os, json, math, random, hashlib
import numpy as np
import torch
from torch import nn
from .env import LocoEnv
from .rewards import DEFAULTS, velocity_tracking, upright, alive, energy_penalty, smoothness, joint_limit

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_dim), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

def compute_reward(obs, action, prev_action, joint_range, coeffs):
    r = 0.0
    r += coeffs.get('velocity_tracking',0)*velocity_tracking(obs)
    r += coeffs.get('upright',0)*upright(obs)
    r += coeffs.get('alive',0)*alive(obs)
    r += coeffs.get('energy_penalty',0)*energy_penalty(action)
    r += coeffs.get('smoothness',0)*smoothness(action, prev_action)
    r += coeffs.get('joint_limit',0)*joint_limit(action, joint_range)

def train(cfg):
    torch.manual_seed(cfg.get('seed', 0))
    np.random.seed(cfg.get('seed',0))
    env = LocoEnv(cfg.get('env_cfg'))
    obs_dim = env.obs_dim
    act_dim = env.action_dim
    policy = Policy(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.get('lr',1e-3))
    max_steps = cfg.get('max_steps', 2000)
    batch = cfg.get('batch', 64)
    gamma = 0.99
    rewards_coeff = cfg.get('reward_coeffs', DEFAULTS)
    joint_range = [env.joint_range[name] for name in env.joint_names]
    # simple on‑policy loop
    obs = env.reset()
    prev_action = np.zeros(act_dim, dtype=np.float32)
    ep_rewards = []
    for step in range(max_steps):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            action = policy(obs_tensor).squeeze(0).numpy()
        next_obs, _, done, _ = env.step(action)
        reward = compute_reward(next_obs, action, prev_action, joint_range, rewards_coeff)
        ep_rewards.append(reward)
        # train on single step for simplicity
        target = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        pred = policy(obs_tensor).mean(dim=1, keepdim=True)
        loss = nn.MSELoss()(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prev_action = action.copy()
        obs = next_obs
        if done:
            obs = env.reset()
        if (step+1) % cfg.get('checkpoint_every', 500) == 0:
            run_id = cfg.get('run_id','run')
            out_dir = os.path.join('experiments','rl',run_id)
            os.makedirs(out_dir, exist_ok=True)
            checkpoints_dir = os.path.join(out_dir,'checkpoints')
            os.makedirs(checkpoints_dir, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(checkpoints_dir,f'step_{step+1:06d}.pt'))
            with open(os.path.join(out_dir,'metrics.jsonl'),'a') as f:
                f.write(json.dumps({'step':step+1,'reward':float(np.mean(ep_rewards))}) + '\n')
    # final checkpoint
    run_id = cfg.get('run_id','run')
    out_dir = os.path.join('experiments','rl',run_id)
    os.makedirs(out_dir, exist_ok=True)
    checkpoints_dir = os.path.join(out_dir,'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(checkpoints_dir,'final.pt'))
    with open(os.path.join(out_dir,'config.json'),'w') as f:
        json.dump(cfg, f, indent=2)
    return out_dir
