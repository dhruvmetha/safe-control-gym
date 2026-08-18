'''Aggregate the quad2d altitude-corridor shards into one dataset directory
per noise level.

Mirrors q2_reduce.py, which is the precedent for the schema: same per-level
directory contents, same key names, so the corridor family is readable by
whatever already reads the shipped quad2d set. Two things differ because the
corridor collector's shards differ from the planar-force collector's:

  train shards carry no det_labels (q2_corridor_collect.py's shard_train
  never rolls the deterministic grid), so train.npz and train_statistics
  drop the deterministic-agreement fields q2_reduce.py reports.

  eval shards carry trials_used PER STATE, not a single scalar trials --
  the reachability shortcut in q2_corridor_collect.py's shard_eval rolls
  a state once instead of `trials` times when its undisturbed trajectory
  never comes within MARGIN of the corridor. p_success is therefore
  hits / trials_used elementwise, and eval_success_prob.npz's `trials` key
  holds that per-state array (constant in q2_reduce.py, variable here --
  same key name so downstream readers are unchanged).

The description also gains a `generation_parameters.corridor` block record-
ing the band geometry, and eval_statistics gains trials_shortcut /
trials_full so the shortcut's effect on the estimate is visible next to the
numbers it produced.
'''
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from q2_common import HORIZON, STATE_BOUNDS, TOL  # noqa: E402
from q2_corridor_common import BAND, CENTRE, PROFILE, SINE_PERIOD, WIDTH, sigma  # noqa: E402

ROOT = os.environ.get('Q2CORR_OUT', os.path.expanduser('~/scg-repo/q2corrout'))
DEST = os.environ.get('Q2CORR_DATASET', os.path.expanduser('~/scg-repo/q2corrdataset'))
CAL_N = 10_000
SPLIT_SEED = 20260813
WEIGHT_N = 0.027 * 9.81
SKIP_MARGIN_M = 0.10  # q2_corridor_collect.py's MARGIN; kept as a literal here
# since the collector does not export the name.


def reduce_train(level, out_dir):
    files = sorted(glob.glob(os.path.join(ROOT, 'train', f'L{level}_s*.npz')))
    if not files:
        return None
    states, offsets, starts, labels, seeds = [], [0], [], [], []
    draw = None
    for f in files:
        d = np.load(f)
        states.append(d['states'])
        offsets.extend((d['offsets'][1:] + offsets[-1]).tolist())
        starts.append(d['starts'])
        labels.append(d['labels'])
        seeds.append(d['seeds'])
        draw = str(d['draw'])
    states = np.concatenate(states)
    labels = np.concatenate(labels)
    offsets = np.asarray(offsets, dtype=np.int64)
    assert offsets[-1] == len(states), 'offsets do not span the state array'
    assert len(offsets) - 1 == len(labels), 'offset/label count mismatch'
    np.savez(os.path.join(out_dir, 'train.npz'),
             states=states, offsets=offsets, starts=np.concatenate(starts),
             labels=labels, seeds=np.concatenate(seeds))
    os.makedirs(os.path.join(out_dir, 'train_test_splits'), exist_ok=True)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(labels))
    with open(os.path.join(out_dir, 'train_test_splits', 'shuffled_indices_0.txt'), 'w') as fh:
        fh.write('\n'.join(f'sequence_{i}.txt' for i in order) + '\n')
    with open(os.path.join(out_dir, 'train_test_splits', 'shuffled_labels_0.txt'), 'w') as fh:
        fh.write('\n'.join(str(int(labels[i])) for i in order) + '\n')
    lengths = np.diff(offsets)
    return dict(num_trajectories=int(len(labels)),
                success_count=int(labels.sum()),
                success_rate=float(labels.mean()),
                mean_length=float(lengths.mean()), max_length=int(lengths.max()),
                hit_horizon=int((lengths - 1 >= HORIZON).sum()),
                total_states=int(len(states)), shards=len(files), draw=draw)


def reduce_eval(level, out_dir):
    files = sorted(glob.glob(os.path.join(ROOT, 'eval', f'L{level}_s*.npz')))
    if not files:
        return None
    starts, hits, det, used = [], [], [], []
    draw = None
    for f in files:
        d = np.load(f)
        starts.append(d['starts'])
        hits.append(d['hits'])
        det.append(d['det_labels'])
        used.append(d['trials_used'])
        draw = str(d['draw'])
    starts = np.concatenate(starts)
    hits = np.concatenate(hits)
    det = np.concatenate(det)
    used = np.concatenate(used)
    p = hits / used

    body = '\n'.join(','.join(f'{v:.6f}' for v in r) + f',{q:.4f}'
                     for r, q in zip(starts, p))
    for name in ('roa_labels.txt', 'eval_states.txt'):
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write(body + '\n')
    lines = body.split('\n')
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(lines))
    cal_n = min(CAL_N, len(lines) // 10)
    for name, idx in (('cal_set.txt', perm[:cal_n]), ('test_set.txt', perm[cal_n:])):
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write('\n'.join(lines[i] for i in idx) + '\n')

    np.savez(os.path.join(out_dir, 'eval_success_prob.npz'),
             starts=starts, successes=hits, trials=used,
             p_success=p, det_labels=det)
    return dict(num_states=int(len(p)), mean_trials=float(used.mean()),
                mean_p_success=float(p.mean()),
                fraction_interior=float(((p > 0) & (p < 1)).mean()),
                agreement_with_deterministic=float(((hits > 0).astype(int) == det).mean()),
                deterministic_rate=float((det == 1).mean()), shards=len(files),
                draw=draw, trials_shortcut=int((used == 1).sum()),
                trials_full=int((used > 1).sum()))


def describe(level, f_max, tr, ev):
    desc = {
        'dataset_name': ('2D Quadrotor RL (safe_explorer_ppo) under an '
                         f'altitude-gated corridor disturbance, f_max={level}'),
        'mechanism': {
            'kind': 'dynamics', 'dim': 1, 'frame': 'world',
            'applied_as': '[Fx, 0, 0] at the COM link -- one-sided, +x only, NO torque',
            'distribution': 'uniform', 'low': 0.0, 'high': 'sigma(z) (altitude-gated)',
            'hold': 'coherent per-rollout sinusoid (draw_law sine), NOT redrawn per step',
            'matched': False,
            'reference_scale': {'body_weight_N': WEIGHT_N,
                                'level_as_fraction_of_weight': float(f_max) / WEIGHT_N},
            'note': ('A corridor of disturbed air below the goal, gated by '
                     'altitude rather than uniform over the state space -- see '
                     "the 'corridor' block below for the geometry."),
        },
        'controller': {'type': 'safe_explorer_ppo',
                       'model': 'safe_explorer_ppo_model_quadrotor_2D_stab.pt',
                       'obs_normalizer': 'frozen (set_read_only) and applied every step',
                       'note': "info['constraint_values'] is seeded before the first step; "
                               'the policy reads it internally'},
        'success_criteria': {
            'type': 'radius', 'threshold': TOL,
            'goal_state': [0, 1, 0, 0, 0, 0],
            'entry_cut': True,
            'note': ('Stops at first entry, so the label is a function of the '
                     'terminal state.'),
        },
        'horizon': {
            'steps': HORIZON, 'seconds': HORIZON / 100.0,
            'inherited': True,
            'note': 'INHERITED from the deterministic set and the planar-force family.',
        },
        'termination_thresholds': {
            'x': 1.0, 'z_min': 0.1, 'z_max': 1.5, 'theta': 'inf',
            'x_dot': 1.0, 'z_dot': 1.0, 'theta_dot': 8.0,
            'source': ('deterministic dataset_description.json; the quad2d RL '
                       'generator sets state_space explicitly'),
            'env_state_indices': {str(k): list(v) for k, v in STATE_BOUNDS.items()},
        },
        'plant': {'quad_type': 2, 'ctrl_freq': 100, 'pyb_freq': 5000,
                  'cost': 'quadratic', 'randomized_init': False,
                  'normalized_rl_action_space': True,
                  'constraints': 'SAFE_EXPLORER_CONSTRAINTS, done_on_violation=False'},
        'data_format': {
            'state_order': ['x', 'z', 'theta', 'x_dot', 'z_dot', 'theta_dot'],
            'angular_velocity_frame': ('world -- for TWO_D the env stores ang_v[1] '
                                       'directly with no body conversion'),
            'theta': 'wrapped to [-pi, pi] when stored',
            'train': 'train.npz -- states(float32,6) offsets starts labels seeds',
            'eval': 'roa_labels.txt / eval_states.txt -- 6 state cols + p_success',
            'precision': {'state': 6, 'p_success': 4},
        },
        'sampling': {'type': 'stratified_grid',
                     'note': ('eval takes the same 489,789 deterministic grid '
                              'states as the planar-force family; train uses '
                              'off-lattice random starts within the same bounds.')},
        'reproducibility': {
            'seed_fn': 'rollout_seed(base, split_id, index, trial), base 20260817',
            'note': ('The disturbance strength is excluded from the seed, so '
                     'levels are paired under common random numbers.'),
        },
        'train_statistics': tr,
        'eval_statistics': ev,
    }
    desc['generation_parameters'] = {
        'corridor': {
            'mechanism': 'altitude_gated',
            'channel': 'dynamics',
            'one_sided': True,
            'direction': '+x',
            'profile': PROFILE,
            'formula': 'F_x ~ U(0, f_max * exp(-0.5*((z-centre)/width)**2))',
            'f_max': float(f_max),
            'centre': CENTRE,
            'width': WIDTH,
            'sigma_at_goal': float(sigma(1.0, f_max)),
            # 3dp, not 4: BAND is (0.1858175, 0.9141825), and the canonical
            # label used everywhere else (q2_corridor_common.py's own BAND
            # comment, the design spec, and this task's own verification
            # step) is the rounder [0.186, 0.914]. round(BAND, 4) would give
            # [0.1858, 0.9142], which is off by a digit from that label.
            'band_1pct': [round(BAND[0], 3), round(BAND[1], 3)],
            'draw_law': 'sine',
            'period_s': SINE_PERIOD,
            'redraw': 'per-rollout phase and amplitude; deterministic within a rollout',
            'skip_margin_m': SKIP_MARGIN_M,
        },
    }
    if ev is not None:
        desc['eval_statistics']['trials_shortcut'] = ev['trials_shortcut']
        desc['eval_statistics']['trials_full'] = ev['trials_full']
    return desc


def main():
    levels = sys.argv[1:] or ['0', '0.050', '0.080', '0.130', '0.300']
    for lv in levels:
        fl = float(lv)
        out_dir = os.path.join(DEST, f'f_{fl:.3f}')
        os.makedirs(out_dir, exist_ok=True)
        tr = reduce_train(lv, out_dir)
        ev = reduce_eval(lv, out_dir)
        desc = describe(lv, fl, tr, ev)
        for name, payload in (('dataset_description.json', desc),
                              ('train_description.json', {**desc, 'split': 'train'}),
                              ('eval_description.json', {**desc, 'split': 'eval'})):
            with open(os.path.join(out_dir, name), 'w') as fh:
                json.dump(payload, fh, indent=2)
        print(f'f_{fl:.3f}: train={tr} eval={ev}', flush=True)


if __name__ == '__main__':
    main()
