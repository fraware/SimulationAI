# Intersection Navigation with Deep Reinforcement Learning

![Project Logo](Project_Logo.png)

## Project Overview

This project develops a reinforcement learning (RL) agent that navigates intersections by choosing turn actions (left, right, straight, back) while following a chosen rule set. The agent uses a Deep Q-Network (DQN) and a simulated environment backed by the Google Maps API (geocoding, directions, nearby places).

- **State Representation:** Location, direction, and nearby streets are encoded into a fixed-size state vector used for decisions.

- **RL Environment:** The environment produces observations from the agent’s location, assigns rewards for rule-compliant turns, and terminates after a set number of intersections.

- **Model:** A DQN with a target network approximates Q-values; the agent acts via an epsilon-greedy policy.

- **Training:** Experience replay and periodic target updates are used for stability.

- **Evaluation:** The trained model is evaluated on test episodes (success rate, average time to destination).

- **Demo:** A script lets you pick a rule (right-only, left-only, alternate) and run a driving simulation with the trained model.

## Installation and Setup

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd inter-sim-rl
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   Or install the package in editable mode with dev dependencies:
   ```sh
   pip install -e ".[dev]"
   ```

3. Set the Google Maps API key (required for training and demo). Replace `<key>` with your key:
   ```sh
   export API_KEY=<key>
   ```
   On Windows (PowerShell):
   ```powershell
   $env:API_KEY = "<key>"
   ```
   You can also set `GOOGLE_MAPS_API_KEY` instead of `API_KEY`.

## Configuration

- **API key:** Set the `API_KEY` or `GOOGLE_MAPS_API_KEY` environment variable. The app raises an error at startup if the key is missing or invalid.

- **Paths and hyperparameters:** Defaults are in `inter_sim_rl.config.Config`. You can override them in code or by extending the config (e.g. model save directory, number of simulations, epsilon, batch size, gamma, target update frequency). Model and output directories default to `models/` and `output/` under the project root.

- **Reproducibility:** Set `Config.seed` to an integer before training to fix random seeds (Python, NumPy, TensorFlow) for reproducible runs.

## Usage

- **Demo (default):** Run the script and choose a rule set; then a trained model is loaded and a driving simulation is run. You must have trained a model once (see below) and set the API key.
  ```sh
  python main.py
  ```

- **Training and evaluation:** To train the DQN and then run evaluation, use:
  ```sh
  python main.py --train
  ```
  This runs the full training loop, saves the model under `models/`, and then runs the test evaluation. Ensure the API key is set.

## Development

- **Run tests:**
  ```sh
  pytest tests -v
  ```

- **Lint:**
  ```sh
  ruff check inter_sim_rl main.py tests
  ```

- **Format:**
  ```sh
  ruff format inter_sim_rl main.py tests
  ```

- **Install dev dependencies:** `pip install -e ".[dev]"` installs pytest and ruff.

## Architecture

- **State:** `StateRepresentation` holds location, direction, nearby streets, and optional address/instruction; `get_state_vector()` returns a fixed-size vector.

- **Environment:** `RLEnvironment` initializes from a starting address, steps via actions (using the Maps API for transitions), and computes rewards from the chosen rule (right, left, or alternate).

- **Agent:** `DQNModel` provides the main and target networks; the training loop in `main.py` collects transitions, samples batches, and updates the DQN with Bellman targets.

- **Demo:** The `simulate_driving_*` functions in `inter_sim_rl.driving_directions` use the same state vector format and action space as training so the saved model can be used for inference.

## Documentation

For component-level details, see the docstrings in `inter_sim_rl` (e.g. `config.py`, `state_representation.py`, `rl_environment.py`, `dqn_model.py`, `driving_directions.py`).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

- Fork the repository.
- Create a new branch.
- Make your changes and commit them.
- Push to your fork and submit a pull request.

## Acknowledgments

- Google Maps API
- TensorFlow

## Contact

If you have any questions, feedback, or issues, feel free to contact us.
