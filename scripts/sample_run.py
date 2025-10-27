"""
python sample_run.py --env feature
python sample_run.py --env image
python sample_run.py --env multi
"""
import argparse

from envs.parking_feature_env import ParkingFeature
from envs.parking_image_env import ParkingImage
from envs.parking_multi_env import ParkingMultiEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=["feature", "image", "multi"], default="feature", help="Which environment to run")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
args = parser.parse_args()

if args.env == "feature":
    env = ParkingFeature(render_mode="human")
elif args.env == "image":
    env = ParkingImage(render_mode="human")
elif args.env == "multi":
    env = ParkingMultiEnv(render_mode="human")

for ep in range(args.episodes):
    print(f"Episode {ep + 1}\n---------------------")
    obs, info = env.reset()
    for step in range(args.steps):
        if args.env == "multi":
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            if not env.agents:
                break
        else:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
