
import numpy as np

def velocity_tracking(obs, target_vel=np.array([0.5,0.0,0.0]), weight=1.0, sigma=0.1):
    """Tracking-error reward toward a target world-frame torso velocity.

    reward = weight * exp(-||v_actual - v_target||^2 / sigma)

    Standing still with a nonzero target scores ~exp(-||target||^2/sigma) (small);
    matching the target scores 1. The FULL velocity error vector is used (not a signed
    projection), so circle-walkers and backwards motion cannot out-score honest progress.
    """
    vel = obs[31:34]
    t = np.asarray(target_vel, dtype=np.float64)[:3]
    err = vel.astype(np.float64) - t
    sq = float(err @ err)
    return weight * float(np.exp(-sq / float(sigma)))


def upright(obs, weight=1.0):
    # Layout: pos 0:14, vel 14:28, torso_up 28:31, torso_vel 31:34, margin 34.
    up = obs[28:31]
    return weight * up[2]

def alive(obs, threshold=0.4, reward=1.0):
    # simple check on upright margin (last element)
    margin = obs[-1]
    return reward if margin > threshold else -reward

def energy_penalty(action, coeff=0.01):
    return -coeff * np.sum(np.square(action))

def smoothness(action, prev_action, coeff=0.1):
    diff = action - prev_action
    return -coeff * np.sum(np.square(diff))

def joint_limit(action, joint_range, coeff=0.5):
    penalty = 0.0
    for i, a in enumerate(action):
        low, high = joint_range[i]
        if a < low or a > high:
            penalty -= coeff * (min(a - low, 0) ** 2 + max(a - high, 0) ** 2)
    return penalty

DEFAULTS = {
    'velocity_tracking': 1.0,
    'upright': 0.5,
    'alive': 1.0,
    'energy_penalty': 0.01,
    'smoothness': 0.1,
    'joint_limit': 0.5,
}

