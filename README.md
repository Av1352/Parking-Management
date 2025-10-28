# 🚗 Parking Management RL: Multi-Agent & Simulation Demo

**Optimize parking in urban environments using Deep Reinforcement Learning (RL) and simulation!  
Try it live and browse the codebase to see RL applied to both single and multi-agent parking problems.**


---

## 📦 Project Overview

This project explores the use of RL and simulation to solve real-world parking management challenges: congestion, space allocation, and autonomous vehicle coordination.  
It includes three main environments:
- **Feature-based (coordinate/MLP):** Classic RL with fast learning and reliable parking agent (PPO).
- **Image-based (vision):** Uses raw pixels—harder and slower to train, shows the challenge of deep RL for vision.
- **Multi-agent:** Multiple cars in a collaborative setting with PettingZoo API, agents must cooperate to park efficiently.

All environments are visualized and run via a Streamlit app, so you can witness agent learning, performance, and failures in real time.

---

## 🖥️ Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parking-management-rl.streamlit.app/)

- **Feature-based:** shows off a PPO agent trained to park reliably.
- **Image and Multi-Agent:** demo runs with random actions for now—see honest reporting of RL limitations and future plans.

---

## 🏆 Features & Modules

- **Environments:** Modular single-agent (`feature-based`, `image-based`) and multi-agent (`multi-agent`) implementations.
- **RL Agent Training:** PPO/DQN via Stable Baselines3, support for custom reward structures and obs spaces.
- **Streamlit Web App:** Interactive UI for selecting env, episodes, and policy, with visualization via GIF.
- **Effort & Transparency:** All code, results, and limitations clearly shown for learning and review.

---

## 📚 Installation & Usage

1. **Clone the repo**

```bash
git clone https://github.com/Av1352/Parking-Management.git
cd Parking-Management
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit demo**

```bash
streamlit run streamlit_app/app.py
```

4. **Try the environments**  
- "Feature-based" lets you select PPO or random policy.
- "Image-based" and "Multi-agent" currently only use random actions (future RL agent support planned).

---

## 🚦 RL Training

The trained PPO agent model (`models/best_model.zip`) is for the feature-based environment only.  
Training scripts and guidelines for image and multi-agent RL are provided; community contributions are welcome for expanding agent support!

---

## 📊 Project Results & Limitations

- Feature-based PPO agent learns swiftly, parks reliably.  
- Image-based agents struggle (slow learning, needs millions of steps).
- Multi-agent RL is an active area—collaboration, vision, and true scalability remain unsolved.

**Random policy runs are shown for unsolved environment/agent combos; results are honestly reported for full transparency.**

---

## 🤝 Team & Contributions

- **Development:** Modular RL codebase, agent training pipeline, Streamlit visualization, and honest reporting by Av1352.
- **Core references:**  
- Stable Baselines3, Gymnasium, PettingZoo, AgileRL
- Research: Multi-Agent RL, parking optimization, collision avoidance

Contributions (especially for vision/multi-agent RL policies) and feedback are welcome!  
Open issues or pull requests for bugs, new agents, analytic dashboards, or documentation fixes.

---

## 🚀 Roadmap

- Add trained agents for image/multi-agent environments.
- More analytic dashboards (success rate, collision stats, reward curves).
- Interactive agent/parameter/seed selector.
- Extend RL policies for real-world scenarios: crowd-sourced parking, trajectory planning.

---

## 📄 License

MIT License—see LICENSE file.

---

## 💬 Contact & Community

Questions, feedback, and collaborations: open a GitHub issue or contact via portfolio/LinkedIn.

---

**This project is designed for transparency, learning, and real impact—try the live demo, inspect the code, and contribute to advancing autonomous parking with RL!**