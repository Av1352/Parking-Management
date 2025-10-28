import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.feature_based.parking_feature_env import ParkingFeature
from envs.image_based.parking_image_env import ParkingImage
from envs.multi_agent.parking_multi_env import ParkingMultiEnv
import pygame
import numpy as np
from PIL import Image
import imageio
from stable_baselines3 import PPO

st.set_page_config(page_title="Parking RL Project Demo", layout="wide")
st.title("🚗 Parking Management RL Demo")
st.sidebar.title("Project & Demo Controls")

env_type = st.sidebar.selectbox("Environment", ["feature-based", "image-based", "multi-agent"])
episodes = st.sidebar.slider("Episodes", 1, 5, 1)
steps_per_episode = st.sidebar.slider("Max steps/ep", 10, 60, 30)

# Only show agent policy selection for feature-based environment!
if env_type == "feature-based":
    policy = st.sidebar.selectbox("Policy", ["Trained PPO Agent", "Random Policy"])
else:
    policy = "Random Policy"
    st.sidebar.warning("Only 'Random Policy' available.")

ppo_model_path = os.path.join("models", "best_model.zip")
model = None
if policy == "Trained PPO Agent" and env_type == "feature-based":
    if os.path.exists(ppo_model_path):
        try:
            model = PPO.load(ppo_model_path)
            st.sidebar.success("Loaded PPO agent for feature-based env!")
        except Exception as e:
            st.sidebar.error(f"Agent load error: {e}")
    else:
        st.sidebar.warning("Trained PPO model not found. Using random actions.")

def get_frame(surface):
    arr = pygame.surfarray.array3d(surface)
    arr = np.transpose(arr, (1, 0, 2))
    return Image.fromarray(arr)

st.markdown("""
### 🚩 Demo Instructions

- Select environment type (feature, image, multi-agent).
- For feature-based: pick trained agent or random policy.
- For image/multi-agent: only random actions.
- Click **Run Demo** to see simulation GIF, rewards, and stats.

---
""")

if st.button("Run Demo and Show GIF"):
    EnvClass = ParkingFeature if env_type == "feature-based" else (
        ParkingImage if env_type == "image-based" else ParkingMultiEnv
    )

    env = EnvClass(render_mode="human")
    all_frames, all_rewards = [], []
    success_count, crash_count = 0, 0
    for ep in range(episodes):
        obs, info = env.reset()
        frames, rewards = [], []
        terminated, truncated = False, False
        for step in range(steps_per_episode):
            if policy == "Trained PPO Agent" and env_type == "feature-based" and model:
                agent_obs = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs
                action, _ = model.predict(agent_obs, deterministic=True)
            else:
                if env_type == "multi-agent" and hasattr(env, "agents"):
                    action = {agent: env.action_space(agent).sample() for agent in env.agents}
                else:
                    action = env.action_space.sample()
            result = env.step(action)
            if env_type == "multi-agent":
                obs, rewards_dict, terminated, truncated, info = result
                reward = sum(rewards_dict.values()) if isinstance(rewards_dict, dict) else rewards_dict
                rewards.append(reward)
                if not env.agents:
                    break
            else:
                obs, reward, terminated, truncated, info = result
                rewards.append(reward)
                if hasattr(env, "parked_successfully") and env.parked_successfully:
                    success_count += 1
                if hasattr(env, "collision") and env.collision:
                    crash_count += 1
                if terminated or truncated:
                    break
            frame = get_frame(env.window)
            frames.append(np.array(frame))
        all_frames.extend(frames)
        all_rewards.append(sum(rewards))
    gif_path = f"demos/{env_type}_parking_demo.gif"
    imageio.mimsave(gif_path, all_frames, duration=0.15)
    st.image(gif_path, caption=f"{env_type.capitalize()} Simulation | {policy}")
    st.success(f"Total Rewards per Episode: {all_rewards}")
    st.progress(success_count / episodes if episodes else 0)
    st.write(f"🙌 Successful Parks: {success_count} / {episodes} episodes")
    if crash_count:
        st.write(f"⚠️ Collisions: {crash_count}")

st.markdown("""
---
### 📝 Project Scope & Limitations

- **Trained agent only available for feature-based environment.**
- **Image & Multi-agent environments:** Demo random actions with honest limits.
- **Effort shown:** Multi-env design, agent training, analytics, and full RL infra.
- **Future:** More agent/model training, visual dashboards, and community input welcome!

---
""")
st.caption("RL simulation and web integration work. See README for technical notes, roadmap, and next steps.")
