import streamlit as st
import sys
import os

# Ensure repo root is in Python path to allow your env imports!
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# RL Environment and Agent Imports
from envs.feature_based.parking_feature_env import ParkingFeature
import pygame
import numpy as np
from PIL import Image
import imageio

# Stable Baselines3 for RL Agent
from stable_baselines3 import PPO

st.set_page_config(page_title="Parking RL Demo", layout="centered")
st.title("Parking Management RL Agent Demo (Feature-based PPO)")
st.sidebar.title("Demo Controls")

# Sidebar controls
episodes = st.sidebar.slider("Episodes", 1, 5, 1)
steps_per_episode = st.sidebar.slider("Steps per episode", 10, 60, 30)
policy = st.sidebar.selectbox("Policy", ["Trained PPO Agent", "Random Policy"])

ppo_model_path = os.path.join("models", "best_model.zip")

# Try loading agent
model = None
if os.path.exists(ppo_model_path):
    try:
        model = PPO.load(ppo_model_path)
        st.sidebar.success("Trained PPO model loaded!")
    except Exception as e:
        st.sidebar.error(f"Agent load error: {e}")
else:
    st.sidebar.warning("Trained PPO model not found. Using random actions.")

# Utility: Convert pygame surface to image
def get_frame(surface):
    arr = pygame.surfarray.array3d(surface)
    arr = np.transpose(arr, (1, 0, 2))
    return Image.fromarray(arr)

if st.button("Run Demo and Show GIF"):
    env = ParkingFeature(render_mode="human")
    all_frames = []
    all_rewards = []
    with st.spinner("Running episodes..."):
        for ep in range(episodes):
            obs, info = env.reset()
            frames, rewards, terminated, truncated = [], [], False, False
            for step in range(steps_per_episode):
                if policy == "Trained PPO Agent" and model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                frame = get_frame(env.window)
                frames.append(np.array(frame))
                if terminated or truncated:
                    break
            all_frames.extend(frames)
            all_rewards.append(sum(rewards))
        # Save GIF
        gif_path = "demos/parking_demo.gif"
        imageio.mimsave(gif_path, all_frames, duration=0.15)
    st.image(gif_path, caption=f"Parking Simulation GIF ({policy})")
    st.success(f"Episode Rewards: {all_rewards}")

st.markdown("""
---
**How to use this demo:**  
- Select number of episodes/steps and policy ("Trained PPO Agent" or "Random").
- Click "Run Demo and Show GIF".
- The GIF shows the car moving within the parking environment.
- Rewards are displayed for each episode; high rewards indicate successful parking.
""")

st.caption("Powered by RL and Streamlit. For interactive demos, visuals, and custom agents, extend this script as needed!")
