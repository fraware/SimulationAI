# Simulation.AI: Intersection Navigation with Reinforcement Learning

![Project Logo](Project_Logo.png)

## Project Overview

In this project, we aim to develop a reinforcement learning (RL) agent capable of navigating intersections effectively while adhering to specific rules to avoid being blocked and prematurely ending simulations. The agent's goal is to reach a destination by making optimal turn decisions at intersections, even in complex scenarios. The project integrates the Google Maps API for environment simulation, data collection, and real-time learning. Our approach leverages the Deep Q-Network (DQN) algorithm and follows a systematic process:

- **State Representation:** We define a comprehensive state representation that encapsulates the agent's current location, direction, and relevant environment information. This representation forms the basis for the agent's decision-making process.

- **RL Environment:** We create a dynamic RL environment that simulates driving scenarios. The environment generates observations (states) based on the agent's location and provides rewards for each action (turn) taken. Termination conditions are defined to ensure the agent doesn't prematurely conclude simulations.

- **Model Architecture:** Our agent employs the DQN algorithm, which uses a deep neural network to approximate Q-values, representing the desirability of different actions. The network architecture takes the state representation as input and outputs action probabilities for making informed decisions.

- **Data Collection:** Using the Google Maps API, we conduct multiple simulations where the agent interacts with the environment. During each simulation, the agent chooses actions (turns) based on its current state and receives rewards based on successful navigation and adherence to predefined rules.

- **Training Loop:** We implement a training loop that utilizes the collected data to update the DQN model. Experience replay and target networks stabilize training by improving data efficiency and mitigating issues related to time-correlated data.

- **Exploration vs. Exploitation:** To balance exploration and exploitation, we employ exploration strategies such as epsilon-greedy or softmax exploration. These strategies encourage the agent to explore different actions while exploiting learned policies.

- **Reward Design:** We design a reward function that guides the agent to take actions leading to successful navigation and rule adherence. Positive rewards are assigned for correct turns, while negative rewards discourage undesired behavior.

- **Evaluation:** The trained DQN model is evaluated on test simulations to assess its performance. Success rate, average time to destination, and other metrics measure the agent's navigation skills and rule compliance.

- **Fine-Tuning and Online Learning:** Hyperparameters are fine-tuned to optimize the model's performance. The model can be further improved through continuous online learning, adapting to changing conditions, and gaining experience from additional simulations.

- **Deployment and Testing:** The trained DQN agent is deployed in the simulation environment, and its behavior is tested in diverse scenarios. The agent's ability to make informed decisions at intersections without getting blocked is demonstrated, meeting the project's overarching goal.

By combining Google Maps API, Deep Reinforcement Learning algorithms, and careful design of the agent's architecture, reward system, and training process, we aim to develop an intelligent agent capable of navigating intersections effectively and successfully completing simulations without premature termination.

## Installation and Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/intersection-navigation-rl.git
   cd intersection-navigation-rl

2. Install required packages:
   ```txt
   pip install -r requirements.txt

4. Configure environment variables
   ```py
   export API_KEY=your_google_maps_api_key

## Usage
To run the project and train the RL agent, follow these steps:

1. Set up the environment as explained in the "Installation and Setup" section.
2. Run the main simulation script:
   ```sh
   python main.py
3. Follow the prompts to choose the rule set and starting address.

## Documentation
For detailed documentation and explanations of each component, refer to the documentation folder.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
We welcome contributions from the community. To contribute, follow these steps:

- Fork the repository.
- Create a new branch.
- Make your changes and commit them.
- Push to your fork and submit a pull request.

## Acknowledgments
We would like to thank the following libraries and resources:

- Google Maps API
- TensorFlow

## Contact
If you have any questions, feedback, or issues, feel free to contact us.
