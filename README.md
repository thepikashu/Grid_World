# Deep Reinforcement Learning

This project explores Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL) by implementing agents from scratch in custom-built environments.
Unlike standard implementations, this project avoids Gym/Gymnasium and builds environments manually using Pygame.

## My Contributions

- Developed standalone GridWorld and Snake environments from scratch using NumPy and Matplotlib to enable high-speed training in headless environments (Google Colab).

- Mitigated the overestimation bias of vanilla DQN by decoupling action selection from action evaluation, leading to more stable convergence.

- Implemented Exponential Epsilon Decay to balance the exploration-exploitation trade-off, significantly smoothing the learning curve compared to linear decay.

- Built a real-time monitoring system using Matplotlib and IPython to track reward plateaus and visualize final agent policies as heatmaps.

## Results & Observations

**Convergence Stability:** The DDQN agent reached a stable reward plateau within 75 episodes, proving that decoupling the target network prevents the "maximization bias" found in standard DQN.

**Reward Shaping Success:** Observed that a step-penalty of -1 was essential for the agent to identify the global minimum path in the GridWorld, successfully navigating around complex obstacles.

**Inference Visualization:** Post-training heatmaps confirm that the agent learned a deterministic optimal policy π∗, consistently reaching the goal from any starting state.

## Algorithms

### Classic RL
- Value Iteration  
- SARSA  
- Q-Learning  

### Deep RL
- DQN  
- DDQN  
- Policy Gradient (REINFORCE)

## Environments

- Grid World  
- Maze Navigation  
- Snake Game (Pygame-based)

## Results & Observations

- Q-learning converges faster in simple environments  
- SARSA is more stable in stochastic settings  
- DQN requires careful tuning to avoid instability  

## Key Learnings

- Reward design is critical for learning  
- Exploration vs exploitation trade-off affects performance  
- Deep RL models are sensitive to hyperparameters

## Author

Yashasvee Taiwade
