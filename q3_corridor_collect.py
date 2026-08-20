'''quad3d stochastic collection under the twin-curtain corridor.

The quad2d collector's sibling. It exists separately from
generate_quadrotor_3d_noisy.py because that one's build() takes a scalar level
plus a mechanism string and knows nothing about curtains; the corridor stack
lives in q3_corridor_common.build(f_max, model, ambient). Everything else is
shared: eval_starts, inject_stored and rollout_seed all come from the same
place, so a shard here covers the same rows of the shipped eval_states.txt that
the deterministic set does.

NO REACHABILITY SHORTCUT WHEN AN AMBIENT TERM IS PRESENT. The quad2d collector
carried one until 2026-08-19: it settled a start in a single flight whenever the
undisturbed path never came within MARGIN of the band, reasoning that an
altitude-gated disturbance cannot reach it. That is true of the curtain, which
sigma(x) gates, and false of the ambient term, which build() adds as ungated
white_noise. Ten skipped quad2d starts re-flown 20 times had 7 vary. The same
trap is available here, since the curtain is likewise gated (on x, not z), so
the gate is written in from the start rather than retrofitted. See the FALSIFIED
block in the quad2d corridor design spec.

Usage:
  python q3_corridor_collect.py --split eval --level 0.20 --model sine+ambient \\
      --ambient 0.008 --trials 20 --shard 0 --nshards 2000 --out eval_0.npz
'''
import argparse
import os

import numpy as np

from generate_quadrotor_3d_noisy import N_TRAIN, eval_starts, rollout_seed, sampler_starts, to_row13
from q3_corridor_common import BAND, NOISE_MODELS, build, roll

TRAIN_SPLIT_ID, EVAL_SPLIT_ID = 0, 1
N_EVAL = 1_000_000

# Slack in x around each curtain, the twin of quad2d's MARGIN. Only consulted
# for models with no ambient term; see the module docstring.
MARGIN = 0.10


def _within_margin(x):
    '''True if x sits inside either curtain's 1%-of-peak band, plus MARGIN.'''
    return any(lo - MARGIN <= x <= hi + MARGIN for lo, hi in BAND)


def shard_eval(args, lo, hi):
    starts, det_labels = eval_starts(lo, hi)
    env, ctrl = build(args.level, model=args.model, ambient=args.ambient)
    hits = np.zeros(len(starts), dtype=np.int32)
    used = np.zeros(len(starts), dtype=np.int32)
    ambient_on = args.ambient is not None and args.ambient > 0
    try:
        for i in range(len(starts)):
            # Trial 0 doubles as the reachability probe. In a top-up window
            # (trial_lo > 0) it replays identically by seeding and is NOT
            # counted; its hits/used belong to the first window's file.
            ok, _, _, entered = roll(env, ctrl, starts[i],
                                     rollout_seed(args.base_seed, EVAL_SPLIT_ID, lo + i, 0),
                                     track_band=True)
            if args.trial_lo == 0:
                hits[i] = int(ok)
                used[i] = 1
            if not ambient_on:
                reachable = entered or _within_margin(float(starts[i][0]))
                if args.level == 0 or not reachable:
                    continue
            for k in range(max(1, args.trial_lo), args.trials):
                ok, _, _ = roll(env, ctrl, starts[i],
                                rollout_seed(args.base_seed, EVAL_SPLIT_ID, lo + i, k))
                hits[i] += int(ok)
                used[i] += 1
    finally:
        env.close()
    np.savez(args.out, starts=starts, hits=hits, trials_used=used,
             det_labels=det_labels, lo=lo, hi=hi, f_max=args.level,
             model=args.model, trial_lo=args.trial_lo, trials=args.trials,
             ambient=-1.0 if args.ambient is None else args.ambient)
    return int(hits.sum()), int(used.sum())


def shard_train(args, lo, hi):
    '''One rollout per start, states kept. Never shortcuts: train wants the
    trajectory, not a probability, so there is nothing to skip.'''
    starts = sampler_starts()[lo:hi]
    env, ctrl = build(args.level, model=args.model, ambient=args.ambient)
    states, offsets, labels, seeds, kept_starts = [], [0], [], [], []
    try:
        for i in range(len(starts)):
            seed = rollout_seed(args.base_seed, TRAIN_SPLIT_ID, lo + i, 0)
            ok, _, traj = roll(env, ctrl, to_row13(starts[i]), seed, keep=True)
            states.append(np.asarray(traj, dtype=np.float32))
            offsets.append(offsets[-1] + len(traj))
            labels.append(int(ok))
            seeds.append(seed)
            kept_starts.append(to_row13(starts[i]))
    finally:
        env.close()
    np.savez(args.out, states=np.concatenate(states),
             offsets=np.asarray(offsets, dtype=np.int64),
             starts=np.asarray(kept_starts), labels=np.asarray(labels, dtype=np.int8),
             seeds=np.asarray(seeds, dtype=np.int64), lo=lo, hi=hi,
             f_max=args.level, model=args.model, trial_lo=0, trials=1,
             ambient=-1.0 if args.ambient is None else args.ambient)
    return int(np.sum(labels)), len(labels)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', choices=['train', 'eval'], required=True)
    ap.add_argument('--level', type=float, required=True, help='curtain f_max, newtons')
    ap.add_argument('--model', choices=sorted(NOISE_MODELS), default='sine+ambient')
    ap.add_argument('--ambient', type=float, default=None)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--trial_lo', type=int, default=0,
                    help='first trial index to roll; >0 writes a top-up window')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--base_seed', type=int, default=20260817)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    # Same guardrails as the sweep CLI: never silently drop or invent ambient.
    entry_ambient = NOISE_MODELS[args.model]['ambient']
    if entry_ambient is None and args.ambient is None:
        ap.error(f'model {args.model!r} requires --ambient (a std in newtons)')
    if entry_ambient is not None and args.ambient is not None:
        ap.error(f'model {args.model!r} has a fixed ambient ({entry_ambient}); '
                 f"drop --ambient or use 'sine+ambient'")
    if args.trial_lo >= args.trials:
        ap.error(f'--trial_lo {args.trial_lo} must be below --trials {args.trials}')
    return args


def main(argv=None):
    args = parse_args(argv)
    # Idempotent: a completed shard is never redone, so resubmitting a partly
    # failed array costs only the missing work.
    if os.path.exists(args.out):
        print(f'{args.out} exists, skipping', flush=True)
        return

    total = N_TRAIN if args.split == 'train' else N_EVAL
    edges = np.linspace(0, total, args.nshards + 1).astype(int)
    lo, hi = int(edges[args.shard]), int(edges[args.shard + 1])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    fn = shard_train if args.split == 'train' else shard_eval
    got, n = fn(args, lo, hi)
    print(f'{args.split} level={args.level} ambient={args.ambient} '
          f'shard {args.shard}/{args.nshards} [{lo}:{hi}] -> {got}/{n}', flush=True)


if __name__ == '__main__':
    main()
