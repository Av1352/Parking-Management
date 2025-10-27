import pytest
from envs.parking_image_env import ParkingImage

def test_env_reset_and_step():
    env = ParkingImage()
    obs, info = env.reset()
    assert obs is not None
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        if terminated or truncated:
            obs, info = env.reset()

def test_image_shape():
    env = ParkingImage()
    obs, info = env.reset()
    assert len(obs.shape) == 3
    assert obs.shape[-1] == 3  # should be H, W, 3

def test_render_runs():
    env = ParkingImage(render_mode="human")
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None