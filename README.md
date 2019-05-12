# Load Frequency Control: A Multi-Agent Deep Deterministic Policy Gradient Approach

### Code structure

* `architecture.py` contains the model used to implement the actors and the critics.
* `dynamics.py` contains the implementation of the equations of the environment.
* `rl.py` contains some utils used in the algorithm.
* `train_two_gens.py` contains the code to train the agents.
* `test_two_gens.py` contains the code to test the trained agents.
* `model` folder contains the trained model.
* `rewards` folder contains a `.pickle` file with the cumulative reward performance of the agent.