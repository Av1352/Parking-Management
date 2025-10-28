import argparse
from stable_baselines3 import PPO, DQN
from envs.feature_based.parking_feature_env import ParkingFeature
from envs.image_based.parking_image_env import ParkingImage
from envs.multi_agent.parking_multi_env import ParkingMultiEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=["feature", "image", "multi"], required=True)
parser.add_argument("--algo", choices=["PPO", "DQN"], required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--episodes", type=int, default=10)
args = parser.parse_args()

if args.env == "feature":
    env = ParkingFeature(render_mode="human")
elif args.env == "image":
    env = ParkingImage(render_mode="human")
elif args.env == "multi":
    env = ParkingMultiEnv(render_mode="human")

if args.algo == "PPO":
    model = PPO.load(args.model_path, env=env)
elif args.algo == "DQN":
    model = DQN.load(args.model_path, env=env)

for ep in range(args.episodes):
    obs, info = env.reset()
    done = False
    step_count = 0
    while not done and step_count < 1000:  # Prevent infinite loops
        if args.env == "multi":
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            done = not env.agents
        else:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        step_count += 1
