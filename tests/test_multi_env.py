import pytest
from envs.multi_agent.parking_multi_env import ParkingMultiEnv

def test_env_reset_and_step():
    env = ParkingMultiEnv()
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert all(isinstance(v, (list, object)) for v in obs.values())
    agents = list(obs.keys())
    for _ in range(5):
        actions = {agent: env.action_space(agent).sample() for agent in agents if agent in env.agents}
        obs, rewards, terms, trunc, info = env.step(actions)
        assert isinstance(rewards, dict)
        assert set(rewards.keys()).issubset(set(agents))
        if not env.agents:
            obs, info = env.reset()

def test_render_runs():
    env = ParkingMultiEnv(render_mode="human")
    obs, info = env.reset()
    agents = list(env.agents)
    actions = {agent: env.action_space(agent).sample() for agent in agents}
    obs, rewards, terms, trunc, info = env.step(actions)
    assert isinstance(obs, dict)
