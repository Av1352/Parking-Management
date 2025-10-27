import argparse
from stable_baselines3.common.env_checker import check_env
from envs.parking_feature_env import ParkingFeature
from envs.parking_image_env import ParkingImage
from envs.parking_multi_env import ParkingMultiEnv

parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=["feature", "image", "multi"], required=True)
args = parser.parse_args()

if args.env == "feature":
    env = ParkingFeature()
elif args.env == "image":
    env = ParkingImage()
elif args.env == "multi":
    env = ParkingMultiEnv()

check_env(env)
print(f"Environment '{args.env}' passed check_env()!")