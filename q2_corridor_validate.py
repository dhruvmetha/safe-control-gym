'''Level-0 gate: at f_max = 0 the corridor collector must reproduce the shipped
deterministic quad2d labels. Anything less means the plant transcription is
wrong and every noisy level built on it would be wrong the same way.

Usage: python q2_corridor_validate.py [--n 300] [--min_agreement 0.98]
'''
import argparse

import numpy as np

from q2_corridor_common import build, grid_states, roll, rollout_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--min_agreement', type=float, default=0.98)
    ap.add_argument('--base_seed', type=int, default=20260817)
    args = ap.parse_args()

    starts, det_labels = grid_states(0, args.n)
    env, ctrl = build(0.0)
    try:
        got = np.zeros(len(starts), dtype=int)
        for i, s in enumerate(starts):
            ok, _, _ = roll(env, ctrl, s, rollout_seed(args.base_seed, 1, i, 0))
            got[i] = int(ok)
    finally:
        env.close()

    agree = int((got == det_labels).sum())
    frac = agree / len(starts)
    print(f'agreement {agree}/{len(starts)} = {frac:.4f}')
    if frac < args.min_agreement:
        raise SystemExit(f'FAIL: below {args.min_agreement}')
    print('PASS')


if __name__ == '__main__':
    main()
