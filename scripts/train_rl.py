"""
python train_rl.py --env feature --algo PPO --timesteps 100000
python train_rl.py --env multi --algo DQN --timesteps 1000000
"""
import argparse
from gymnasium.wrappers import TimeLimit, GrayScaleObservation
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from envs.feature_based.parking_feature_env import ParkingFeature
from envs.image_based.parking_image_env import ParkingImage
from envs.multi_agent.parking_multi_env import ParkingMultiEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=["feature", "image", "multi"], required=True)
parser.add_argument("--algo", choices=["PPO", "DQN"], required=True)
parser.add_argument("--timesteps", type=int, default=100000)
parser.add_argument("--n_envs", type=int, default=4)
args = parser.parse_args()

name = f"{args.algo}_{args.env}"
tmp_path = f"./logs/{name}"
new_logger = configure(tmp_path, ["csv", "tensorboard", "log"])
checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path=f"./models/{name}/checkpoint",
    name_prefix=f"{name}", save_replay_buffer=True, save_vecnormalize=True
)

def make_gym_env():
    if args.env == "feature":
        env = ParkingFeature()
        env = TimeLimit(env, 150)
    elif args.env == "image":
        env = ParkingImage()
        env = TimeLimit(env, 400)
        env = GrayScaleObservation(env, keep_dim=True)
    elif args.env == "multi":
        env = ParkingMultiEnv()
        env = TimeLimit(env, 150)
    env = Monitor(env, f"./logs/{name}/monitor")
    return env

vec_env = make_vec_env(make_gym_env, n_envs=args.n_envs)
eval_callback = EvalCallback(vec_env, best_model_save_path=f"./models/{name}/best/",
    log_path=f"./models/{name}/best/", eval_freq=3000,
    deterministic=True, render=False)

if args.algo == "PPO":
    model = PPO("MlpPolicy", vec_env, verbose=1)
elif args.algo == "DQN":
    model = DQN("MlpPolicy", vec_env, verbose=1)

model.set_logger(new_logger)
model.learn(total_timesteps=args.timesteps, callback=[checkpoint_callback, eval_callback], progress_bar=True)
model.save(f"./models/{name}/final_model")