'''The corridor collector's shared setup.

Pinned here because these constants are transcribed from the shipped
deterministic set and a silent drift in any of them invalidates the whole
comparison.
'''
import importlib.util
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    'q2_corridor_common', os.path.join(ROOT, 'q2_corridor_common.py'))
q2c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q2c)


def test_plant_constants_match_the_shipped_set():
    assert q2c.HORIZON == 1200
    assert q2c.TOL == 0.2
    assert q2c.CENTRE == 0.55
    assert q2c.WIDTH == 0.12


def test_sigma_is_zero_at_the_goal_to_four_decimals():
    assert round(q2c.sigma(1.0, 0.05), 4) == 0.0


def test_band_bounds_are_the_one_percent_contour():
    lo, hi = q2c.BAND
    assert abs(q2c.sigma(lo, 1.0) - 0.01) < 1e-9
    assert abs(q2c.sigma(hi, 1.0) - 0.01) < 1e-9


def test_rollout_seed_excludes_the_level():
    '''Levels must be paired: same coordinates, same stream, any f_max.'''
    import inspect
    src = inspect.getsource(q2c.rollout_seed)
    assert 'f_max' not in src and 'level' not in src


def test_build_wires_the_corridor_to_the_dynamics_channel():
    from safe_control_gym.envs.disturbances import AltitudeGatedNoise
    env, ctrl = q2c.build(0.01)
    try:
        assert 'dynamics' in env.disturbances
        assert 'action' not in env.disturbances
        dist = env.disturbances['dynamics'].disturbances[0]
        assert isinstance(dist, AltitudeGatedNoise)
        assert dist.profile == 'gaussian'
        assert list(dist.mask) == [1.0, 0.0]
    finally:
        env.close()


def test_build_at_zero_installs_no_disturbance():
    env, ctrl = q2c.build(0.0)
    try:
        assert env.disturbances == {}
    finally:
        env.close()


def test_state_bounds_are_applied_after_construction():
    env, _ = q2c.build(0.0)
    try:
        assert env.state_space.low[2] == 0.1
        assert env.state_space.high[2] == 1.5
        assert env.state_space.high[5] == 8.0
        # theta (index 4) is absent from STATE_BOUNDS, so it stays at the
        # env's own default -- not re-bounded by the corridor's overrides.
        assert env.state_space.high[4] == np.float32(env.theta_threshold_radians)
    finally:
        env.close()
