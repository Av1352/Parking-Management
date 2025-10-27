# Parking-Management

An industry-optimized suite of RL environments and playable games for car parking using Python, OpenAI Gymnasium, Stable-Baselines3, and Pygame. Includes feature-based, image-based, and multi-agent environments for RL experiments, plus classic playable demos.

## Project Structure

- `envs/` — RL environments (feature, image, multi-agent)
- `utils/` — Shared constants, asset loaders, random helpers
- `assets/` — Game images and sprites
- `games/` — Classic and experimental playable games using Pygame
- `scripts/` — Unified runners for training, evaluation, env checks, and result inspection
- `tests/` — Pytest-based unit and integration tests
- `streamlit_app/` — Streamlit dashboard (future expansion)

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a sample environment:

```bash
python scripts/sample_run.py --env feature
```

Train an RL agent (example):

```bash
python scripts/train_rl.py --env feature --algo PPO --timesteps 100000
```

Run Pygame classic game:

```bash
python games/parking_game.py
```


## Outputs

Training creates logs/output in local folders (`logs/`, `models/`, etc.), which are not tracked in git. Use scripts/view_npz.py for inspecting NPZ result files.

## Streamlit App

Coming soon: interactive Streamlit dashboard for visualization and experiment control!

## License

MIT