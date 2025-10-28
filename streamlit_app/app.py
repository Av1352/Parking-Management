import streamlit as st
import sys
import os

# This path makes 'sample_run.py' discoverable when running from repo root or from 'streamlit_app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from scripts.sample_run import ParkingFeature, ParkingImage, ParkingMultiEnv

# App sidebar options
st.sidebar.title("Parking Management RL Dashboard")
mode = st.sidebar.selectbox("Select mode", ["Run Environment", "Train Agent", "Evaluate Agent", "View Logs"])

if mode == "Run Environment":
    st.header("Run Parking Environment")
    env_type = st.selectbox("Environment", ["feature", "image", "multi"])
    episodes = st.slider("Episodes", 1, 10, 1)
    steps = st.slider("Max Steps per episode", 10, 300, 100)

    run_button = st.button("Run now")
    if run_button:
        if env_type == "feature":
            env = ParkingFeature(render_mode="human")
        elif env_type == "image":
            env = ParkingImage(render_mode="human")
        elif env_type == "multi":
            env = ParkingMultiEnv(render_mode="human")

        # Run sample episode(s)
        for ep in range(episodes):
            st.write(f"Episode {ep+1}:")
            obs, info = env.reset()
            for step in range(steps):
                if env_type == "multi":
                    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                    obs, rewards, terminations, truncations, infos = env.step(actions)
                    if not env.agents:
                        break
                else:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        obs, info = env.reset()
            st.write("Episode finished.")

elif mode == "Train Agent":
    st.header("Training (CLI only for now)")
    st.info("To train, use the CLI: `python scripts/train_rl.py --env feature --algo PPO --timesteps 100000`")

elif mode == "Evaluate Agent":
    st.header("Evaluation (CLI only for now)")
    st.info("To evaluate, use the CLI: `python scripts/eval_rl.py --env feature --algo PPO --model_path PATH_TO_MODEL`")

elif mode == "View Logs":
    st.header("Logs & Outputs")
    log_path = "./logs/"
    if os.path.exists(log_path):
        logs = os.listdir(log_path)
        st.write("Available logs:", logs)
    else:
        st.write("No logs found.")

st.sidebar.caption("Parking-Management Suite | RL Dashboard")