'''Shared quad2d setup for the altitude-corridor stochastic collection.

Everything about the plant is transcribed from q2_common.py, which in turn came
from deterministic/quadrotor2D_rl. The only difference is the disturbance: a
one-sided +x force whose bound is gated by altitude, forming a corridor of
disturbed air below the goal.

Spec: docs/superpowers/specs/2026-08-17-quad2d-altitude-corridor-design.md
'''
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from functools import partial  # noqa: E402

from generate_quadrotor_2d_trajectories_rl import (ALGO_CONFIGS, SAFE_EXPLORER_CONSTRAINTS,  # noqa: E402
                                                   normalize_angle)
from safe_control_gym.utils.registration import make  # noqa: E402

DET = os.environ.get(
    'Q2_DET_DIR',
    '/common/users/shared/pracsys/genMoPlan/data_trajectories/'
    'deterministic/quadrotor2D_rl')
MODEL = os.environ.get(
    'Q2_MODEL',
    'examples/rl/models/safe_explorer_ppo/safe_explorer_ppo_model_quadrotor_2D_stab.pt')

HORIZON = 1200
TOL = 0.2
GOAL = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])   # file order [x,z,theta,xd,zd,td]

PROFILE = 'gaussian'               # ALTITUDE_PROFILES entry -- profile D
CENTRE = 0.55                      # corridor centre, m
WIDTH = 0.12                       # gaussian width parameter, m
# Altitudes where sigma is above 1% of peak. Everything outside is calm enough
# that a trajectory crossing it is effectively undisturbed.
_K = WIDTH * math.sqrt(2 * math.log(100))
BAND = (CENTRE - _K, CENTRE + _K)   # (0.186, 0.914)

# env state order is [x, x_dot, z, z_dot, theta, theta_dot]
STATE_BOUNDS = {0: (-1.0, 1.0), 1: (-1.0, 1.0), 2: (0.1, 1.5),
                3: (-1.0, 1.0), 5: (-8.0, 8.0)}
TASK_INFO = {'stabilization_goal': [0, 1], 'stabilization_goal_tolerance': TOL}


def sigma(z, f_max):
    '''The draw bound at altitude z, in newtons.'''
    return f_max * math.exp(-0.5 * ((z - CENTRE) / WIDTH) ** 2)


def rollout_seed(base, split_id, index, trial):
    '''Pure function of the coordinates -- deliberately excludes the disturbance
    strength.

    A resumed shard draws exactly what an uninterrupted run would have drawn,
    and every disturbance strength sees the same stream per (start, trial), so
    strengths are paired and strength-to-strength differences carry far less
    variance than the individual estimates.
    '''
    return int((base + split_id * 1_000_003 + index * 7919 + trial * 104_729)
               % (2 ** 31 - 1))


DRAW_FUNCS = {'uniform': 'altitude_gated', 'sine': 'altitude_gated_sine'}
SINE_PERIOD = 2.0   # seconds; see AltitudeGatedSineNoise and the spec's re-sweep note


def build(f_max, draw='uniform'):
    '''draw selects the _draw law: 'uniform' (per-step, the original family) or
    'sine' (AltitudeGatedSineNoise, the per-rollout coherent draw adopted after
    the sweep found the per-step law's fraction_interior peaking at 0.025 --
    see "The sweep falsified the per-step draw" in the corridor design spec).
    'uniform' stays the default so existing callers are unchanged.
    '''
    if draw not in DRAW_FUNCS:
        raise ValueError(f'[ERROR] q2_corridor_common.build(): unknown draw '
                         f'{draw!r}; choose from {sorted(DRAW_FUNCS)}.')
    kw = dict(quad_type=2, task='stabilization', task_info=TASK_INFO,
              ctrl_freq=100, pyb_freq=5000, gui=False, randomized_init=False,
              episode_len_sec=1000, cost='quadratic', done_on_out_of_bound=True,
              normalized_rl_action_space=True,
              constraints=SAFE_EXPLORER_CONSTRAINTS, done_on_violation=False)
    if f_max > 0:
        disturb_cfg = {'disturbance_func': DRAW_FUNCS[draw],
                       'f_max': f_max, 'profile': PROFILE,
                       'centre': CENTRE, 'width': WIDTH, 'mask': [1, 0]}
        if draw == 'sine':
            disturb_cfg['period'] = SINE_PERIOD
        kw['disturbances'] = {'dynamics': [disturb_cfg]}
    env_func = partial(make, 'quadrotor', **kw)
    cfg = ALGO_CONFIGS['safe_explorer_ppo'].copy()
    tmp = tempfile.mkdtemp(prefix='q2corr-')
    ctrl = make('safe_explorer_ppo', env_func, **cfg, output_dir=tmp)
    ctrl.load(MODEL)
    ctrl.obs_normalizer.set_read_only()
    env = env_func()
    for i, (lo, hi) in STATE_BOUNDS.items():
        env.state_space.low[i], env.state_space.high[i] = lo, hi
    return env, ctrl


def roll(env, ctrl, state6, seed, keep=False, track_band=False):
    '''state6 is FILE order [x, z, theta, x_dot, z_dot, theta_dot].

    `track_band` additionally reports whether the trajectory ever entered the
    corridor, which the eval split uses to skip states the disturbance cannot
    reach.
    '''
    import pybullet as pb
    x, z, theta, x_dot, z_dot, theta_dot = state6
    obs, info = env.reset(seed=int(seed))
    pb.resetBasePositionAndOrientation(
        env.DRONE_ID, [x, 0, z], pb.getQuaternionFromEuler([0, theta, 0]),
        physicsClientId=env.PYB_CLIENT)
    pb.resetBaseVelocity(env.DRONE_ID, [x_dot, 0, z_dot], [0, theta_dot, 0],
                         physicsClientId=env.PYB_CLIENT)
    env._update_and_store_kinematic_information()
    obs = env._get_observation()
    if getattr(env, 'constraints', None) is not None:
        info['constraint_values'] = env.constraints.get_values(env, only_state=True)
    traj = [[x, z, normalize_angle(theta), x_dot, z_dot, theta_dot]] if keep else None
    entered = BAND[0] <= z <= BAND[1]
    success = False
    steps = 0
    for steps in range(1, HORIZON + 1):
        action = ctrl.select_action(ctrl.obs_normalizer(obs), info)
        obs, _, terminated, truncated, info = env.step(action)
        if keep:
            xx, xd, zz, zd, th, td = obs[:6]
            traj.append([xx, zz, normalize_angle(th), xd, zd, td])
        if track_band and not entered:
            entered = BAND[0] <= float(env.state[2]) <= BAND[1]
        if terminated or truncated:
            success = bool(info.get('goal_reached', False))
            break
    if track_band:
        return success, steps, traj, entered
    return success, steps, traj


def grid_states(lo, hi):
    '''Rows [lo:hi) of roa_labels.txt: 6 state columns plus the shipped label.'''
    rows = np.loadtxt(DET + '/roa_labels.txt', delimiter=',',
                      skiprows=lo, max_rows=hi - lo, ndmin=2)
    return rows[:, 0:6], rows[:, 6].astype(int)
