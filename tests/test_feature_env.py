# tests/test_feature_env.py
"""
Unit tests for the ParkingFeature RL environment.
Run with: pytest tests/test_feature_env.py
"""
import pytest
from envs.parking_feature_env import ParkingFeature

def test_env_reset_and_step():
    env = ParkingFeature()
    obs, info = env.reset()
    assert obs is not None
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        if terminated or truncated:
            obs, info = env.reset()

def test_env_boundaries():
    env = ParkingFeature()
    obs, info = env.reset()
    # Repeatedly move up beyond expected boundary
    env.car_orientation = "up"
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(3)  # forward
    assert env.car_rect.top >= 0

def test_render_runs():
    env = ParkingFeature(render_mode="human")
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
