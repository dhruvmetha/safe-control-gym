# quad2d stochastic collection under an altitude-gated one-sided sidewind

Date: 2026-08-17
System: quadrotor 2D, `safe_explorer_ppo`, stabilization
Status: draw law revised 2026-08-18 -- per-step uniform measured degenerate (see The sweep falsified the per-step draw), per-rollout sinusoid adopted; re-sweep pending

## What this is

A disturbance **corridor**: a band of disturbed air at a fixed altitude that a
trajectory has to cross to reach the goal. The force is one-sided, always +x,
and its magnitude is gated by altitude so that the band is localised and the
goal sits in calm air above it.

Two things here are new to this repo. It is the first **biased** disturbance --
every existing family is zero-mean and symmetric, `uniform(-f, +f)` or
`Normal(0, sigma)`. And it is the first whose scale is a function of **env
state** rather than of the commanded signal; `SignalDependentNoise` keys off
`|target|`, which is a different mechanism (see Rejected alternatives).

The object being measured is unchanged: a stochastic ROA, published as
`p_success` over K trials at fixed eval start states, plus a train split of
labelled trajectories.

## The noise model

```
sigma(z) = F_max * exp(-0.5 * ((z - 0.55) / 0.12)**2)     # newtons, the draw bound
F_x      ~ U(0, sigma(z_t))                                # fresh draw, every control step
F_z      = 0
```

`sigma(z)` is the **upper bound of the draw**, not a scale on a fixed
distribution -- the support itself moves with altitude. Mean load is
`sigma(z)/2`, standard deviation `sigma(z)/(2*sqrt(3))`. The two are locked at
a ratio of `1/sqrt(3)`, so a single-parameter `U(0, sigma)` cannot set the push
and the variability independently. Accepted: the corridor's job is to be a
hazard of a given strength, not to disentangle bias from spread.

### The envelope is a strategy

`sigma(z)` is not hard-coded. The disturbance class resolves it from a named
profile registry, `ALTITUDE_PROFILES`, with the Gaussian bump above registered
as `'gaussian'` -- the only entry for now. A future profile (a different shape,
a different placement) is a new registry entry plus its parameters in the
config; nothing else changes. Likewise the draw law is isolated in a single
`_draw(bound)` method so an alternative law (see the sine entry under Rejected
alternatives) is a subclass override, not a rewrite. Every
`dataset_description.json` records the profile name beside its parameters,
because two datasets with the same `f_max` and different profiles are not
comparable.

Injected through the existing `dynamics` channel, so: a world-frame force at
the COM, producing **no torque**, re-applied on every one of the 50 PyBullet
substeps because PyBullet clears external forces per `stepSimulation`
(`base_aviary.py:280-288`). Unmatched -- `range(B_d)` is not contained in
`range(G(x))`, and the vehicle can only oppose a lateral force by tilting first.

For `TWO_D` the disturbance vector is 2-D and maps to `[Fx, 0, Fz]`
(`quadrotor.py:465`), so a `[1, 0]` mask restricts it to +x.

### Band geometry

| quantity | value |
| --- | --- |
| centre | z = 0.55 |
| width parameter | 0.12 m |
| core, >= 60% of peak | z in [0.43, 0.67], 0.24 m |
| extent, > 1% of peak | z in [0.186, 0.914], 0.728 m |
| sigma at the goal (z = 1.0) | 0.09% of peak -- effectively zero |
| goal ball altitude span | z in [0.8, 1.2] |

"Narrow" is relative. At the 1% threshold the band spans more than half the
usable altitude range; it is a broad gradient with a concentrated core.

## Why the goal must be calm

This is the constraint that drove the whole design, and it is not obvious.

`goal_reached` is `||state - X_GOAL|| < 0.2` over the **full 6-D state**
(`quadrotor.py:920`), and for `TWO_D` stabilization `X_GOAL = [0, 0, 1, 0, 0, 0]`
(`quadrotor.py:294-297`). **Theta = 0 is inside the success test.** The
out-of-bounds mask exempts theta as periodic; the goal test does not.

Holding station against a mean lateral load requires a standing tilt
`theta_ss = atan((sigma/2) / mg)`, with `mg = 0.027 * 9.8 = 0.2646 N`. That tilt
is charged against the same 0.2 ball that defines success:

| mean load | theta_ss | share of the 0.2 ball |
| --- | --- | --- |
| 0.005 N | 0.019 rad | 9% |
| 0.010 N | 0.038 rad | 19% |
| 0.025 N | 0.094 rad | 47% |

So a profile with appreciable `sigma` at the goal cannot be satisfied: the tilt
that cancels the force is scored against the vehicle. Placing the corridor below
the goal removes the conflict -- tilt during transit is free, because only the
state at the moment of the goal test is measured.

## Rejected alternatives

**Profile A, a bump spanning the full altitude range** (zeros at z = 0.1 and
1.5, peak at 0.8). Rejected on measurement: the goal ball spans z in [0.8, 1.2]
and A peaks at 0.8, so the peak sits *inside* the success region -- 81% of peak
force at z = 1.0. Keeping the standing tilt under a quarter of the ball caps
`F_max` at 0.033 N, an order of magnitude below the 0.070-0.200 N levels the
existing zero-mean quad2d family already uses. Most of any sweep would read
`p_success ~ 0` for reasons that have nothing to do with the ROA.

**Profiles B and C** (peak below the goal; twin bands either side). Both satisfy
the tilt constraint. Rejected in favour of D because neither is localised --
they span the whole approach, so there is no "outside the corridor" and no
control group of trajectories that never meet the disturbance.

**A profile carrying a small non-zero sigma at the goal.** Proposed as a fifth
candidate to preserve the terminal asymmetry a biased force produces. Not
adopted: it optimises for asymmetry in the terminal-state distribution, which is
a different experiment from a crossing hazard. Recorded because the trade is
real -- a corridor deliberately gives the controller calm air in which to undo
the asymmetry imprinted during the crossing.

**Deterministic field** (`F_x = sigma(z)`, no draw). Rejected: given a start
state the outcome is fixed, `p_success` collapses to {0, 1}, and K trials buy
nothing.

**Frozen random field indexed by z**, drawn once per rollout. Attractive because
it makes the disturbance a property of the airspace rather than of the clock,
and because it defeats the averaging described below. Rejected for cost: it
needs a correlation length as a second parameter and a field-generation path
with its own reproducibility story, and the existing per-step mechanism is
already demonstrated to produce usable spread in this exact setup
(`fraction_interior` 0.122 at `f = 0.150` in the shipped quad2d family).

**Per-rollout scalar scale** (`F_x = sigma(z) * u`, `u ~ U(0,1)` at reset).
Rejected as the coarsest of the three; it makes every crossing a scaled copy of
the same gust.

**Per-rollout sinusoid** (`F_x = sigma(z) * [0.5 + 0.5 * A * sin(omega*t + phi)]`,
`A ~ U(0,1)` and `phi ~ U(-pi, pi)` drawn at reset, `omega` fixed). The measured
fallback if the per-step draw's transition shell proves too thin. On the
standard crossing (z_dot = 0.5, 1.46 s in band) the delivered-impulse spread is
53.5% with phase alone and 30.9% with phase+amplitude, against 6.3% for per-step
i.i.d. -- the coherence defeats the averaging, and the amplitude decouples mean
(set by the 0.5 bias term) from spread (set by A). Rejected as the default for
five reasons, in order: omega is a timescale whose effect depends on crossing
time and is therefore entangled with the start-state distribution; the
randomness collapses to two per-rollout scalars, making p_success a 2-D integral
over a fixed gust family rather than robustness to noise; a single tone
concentrates all disturbance energy at one frequency, so any closed-loop
resonance there becomes a dataset-wide artifact; it changes two things at once
relative to the flat `f_*` family (gating and temporal structure), breaking
attribution; and its mean is sigma/2 only per full period, so effective
difficulty varies with entry timing. Revive it -- as a `_draw` override, see
"The envelope is a strategy" -- only if the sweep's fraction_interior stop
condition triggers.

## The sweep falsified the per-step draw

Ran 2026-08-18: 400 random grid states, 30 paired trials, fourteen levels
0.002-0.80 N. Retention spans 1.00 down to 0.06, but `fraction_interior`
peaks at **0.025** (10/400 states, at 0.13 N) and never approaches the 0.122
of the shipped zero-mean family. Raising the force does not help: at 0.50 N
retention hits 0.09 -- the structural floor set by the 9.25% of successes that
never enter the band -- with 283 horizon-hits (the quad3d deadline artifact),
and at 0.80 N interior is exactly 0. The averaging mechanism below is real and
no F_max escapes it. Decision [user, 2026-08-18]: switch the draw law to the
banked per-rollout sinusoid; the ladder is re-swept under it.

The switch is `AltitudeGatedSineNoise(AltitudeGatedNoise)`, registered as
`'altitude_gated_sine'`: `reset()` draws `phi ~ U(-pi, pi)` and `A ~ U(0, 1)`
once per episode from the seeded stream; `_draw` returns
`bound * (0.5 + 0.5 * A * sin(2*pi/PERIOD * t + phi))` with `t` the control
step time and `PERIOD = 2.0 s` (about 1.4x the median band-crossing time --
inside the control bandwidth, long enough not to average out, short enough
that a crossing sees a partial cycle rather than a frozen offset). Mean force
is unchanged (`E[0.5 + 0.5*A*sin] = 0.5`), so the tilt-budget analysis
carries over. The five objections recorded against the sinusoid below were
weighed against the measured degeneracy and accepted as the lesser cost.

## Known cost of the per-step draw

Per-step i.i.d. draws average along a crossing. Measured analytically for D at
`F_max = 0.05 N`:

| z_dot | steps in band | mean delta-v_x | std of delta-v_x | spread |
| --- | --- | --- | --- | --- |
| 0.25 | 291 | 1.114 m/s | 0.049 m/s | 4.4% |
| 0.50 | 146 | 0.557 m/s | 0.035 m/s | 6.3% |
| 1.00 | 73 | 0.279 m/s | 0.025 m/s | 8.9% |

The consequence is a **thin transition shell** rather than a degenerate
`p_success`: states far from the success boundary read 0 or 1, and only a narrow
band of marginal states reads intermediate. This is accepted, but the sweep must
measure the realised spread rather than trust this estimate.

Note also that the mean impulse is large. At `F_max = 0.05 N` a crossing at
`z_dot = 0.5` delivers 0.557 m/s of lateral velocity against an `x_dot` bound of
+/-1.0, which is why the level bracket below sits well under the existing
family's.

## Corridor coverage

Over all 489,789 shipped eval start states (`roa_labels.txt`, column 1):

| region | share | count |
| --- | --- | --- |
| below the band (z < 0.186) | 14.29% | 69,968 |
| inside the band | 42.86% | 209,912 |
| above the band (z > 0.914) | 42.86% | 209,909 |

The near-exact fractions come from the stratified grid the shipped set was
sampled on.

Measured on the undisturbed (`f_max = 0`) trajectory from 20,000 states sampled
uniformly at random from the shipped set (`q2_corridor_entry.py`):

| region | n | entered |
| --- | --- | --- |
| below the band | 2,791 | 5.63% |
| inside the band | 8,616 | 100.00% |
| above the band | 8,593 | 6.85% |
| overall | 20,000 | 46.81% |

Starts outside the band almost never enter it: below the band the trajectory
either terminates near the floor or never climbs into the corridor; above the
band it flies directly to the goal, which sits at z=1.0 above the band's top
edge (0.914), without dropping through. So the disturbance can only reach
about 47% of eval states, not the 100% a start-state count would suggest.
Under the corridor framing the states that do cross are not wasted -- they are
the control group, and the contrast between "had to cross" and "did not" is
the signal. But a state whose trajectory never enters the band gives the same
answer on all K trials, so skipping it is worth roughly half the eval compute.

**Mitigation to implement:** roll each eval state once deterministically, and
spend the K trials only on states whose trajectory enters the band, with a
margin in z to cover trajectories that noise could push in. States outside the
margin get `p_success` from the single roll.

## Implementation

One new `Disturbance` subclass. `SignalDependentNoise` cannot serve: its scale
is `alpha + beta * |target|`, and on the `dynamics` channel `target` is the
zero vector `disturb_force` (`quadrotor.py:450`), so it would collapse to a
constant `alpha`. The new class reads altitude off the env instead --
`apply(target, env)` already receives `env`, and the TWO_D state order is
`[x, x_dot, z, z_dot, theta, theta_dot]`, so altitude is `env.state[2]`.

Registered in `DISTURBANCE_TYPES` alongside the existing six, and configured the
same way as every other family:

```python
disturbances = {'dynamics': [{'disturbance_func': 'altitude_gated',
                              'f_max': F_MAX, 'centre': 0.55, 'width': 0.12,
                              'mask': [1, 0]}]}
```

Everything else is transcribed from `q2_common.py` unchanged: horizon 1200,
tolerance 0.2 entry-cut, `cost='quadratic'`, bounds x +/-1.0, z [0.1, 1.5],
velocities +/-1.0, theta_dot +/-8.0, `safe_explorer_ppo` with the shipped model
and a frozen `obs_normalizer`.

Collection follows the existing q2 pipeline shape: a collector, a level-0 gate
that must reproduce the deterministic labels before any noisy level runs, a
sweep that chooses the levels, and a reducer that writes the dataset layout.

## To be measured before collection

1. **Level ladder.** Starting bracket `F_max` in roughly 0.002-0.020 N, derived
   from the impulse numbers above, not from the existing family's 0.070-0.200 N
   -- those are zero-mean and un-gated and do not transfer. `q2_sweep.py` is the
   pattern.
2. **Realised trial-to-trial spread**, and the `fraction_interior` it produces.
   If the transition shell is too thin to be useful, the frozen-field
   alternative above is the fallback and this spec should be revised rather than
   the levels pushed higher.
3. **Entry rate.** What fraction of trajectories from each start region actually
   enter the band, which sets both the mitigation margin and the interpretation
   of any level-averaged `p_success`.

## Level naming

`f_*` in the existing families means peak newtons of a flat uniform bound. Here
`F_max` is the peak of an altitude-varying bound, reached only at z = 0.55.
A single scalar therefore does not describe the disturbance. Levels are named by
`F_max`, and each `dataset_description.json` must additionally record the centre,
the width, `sigma` at the goal, and the band extent -- otherwise the level names
are not comparable with any other family, or with each other under a later change
of geometry.
