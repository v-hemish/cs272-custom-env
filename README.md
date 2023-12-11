# Agriculture Management with Reinforcement Learning

## Project Summary
This project introduces a Reinforcement Learning (RL) environment designed to simulate agriculture management. The core idea is to apply RL algorithms to manage a crop's growth considering various environmental factors and resource limitations. The project utilizes Ray RLlib, a popular framework for RL, to train an intelligent agent that can make decisions to maximize crop yield while efficiently managing resources.

## State Space
The state space in the Agriculture Environment comprises six key factors:
- **Soil Moisture (0-1)**: Reflects the current moisture level in the soil.
- **Nutrient Level (0-1)**: Indicates the level of nutrients available to the crop.
- **Crop Growth (0-1)**: Measures the growth stage of the crop.
- **Weather Condition (0-1)**: Represents the impact of weather on crop growth.
- **Water Reserve (0-1)**: Shows the available water reserve for irrigation.
- **Nutrient Reserve (0-1)**: Displays the remaining nutrient reserves for fertilization.

## Action Space
The agent has three possible actions to choose from:
1. **No Action**: The agent decides not to take any immediate action.
2. **Water the Crop**: This action aims to increase soil moisture.
3. **Fertilize the Crop**: This action is intended to boost the nutrient level.

## Rewards
The reward function is designed to encourage the agent to maximize crop growth while maintaining balance in resource use. It is calculated as the growth of the crop minus the deviation of soil moisture and nutrient levels from their ideal values.

## RL Algorithm 
The project leverages various algorithms from Ray RLlib, such as Proximal Policy Optimization (PPO), Actor-Critic using Kronecker-Factored Trust Region (ACKTR), Soft Actor-Critic (SAC), Behavioral Cloning (BC), Deep Q Networks (DQN), Impala, and Maximum a posteriori Policy Optimisation (MPO). These algorithms offer a range of strategies, from value-based to policy-based methods, providing a comprehensive approach to training the RL agent.

### Key Configuration Parameters
- **Environment**: "AgricultureEnv"
- **Number of GPUs**: 0 (can be adjusted based on system capabilities)
- **Number of Workers**: 7 (for parallel training)
- **Framework**: TensorFlow (flexible to change to PyTorch)
- **Replay Buffer**: Prioritized Replay Buffer with a capacity of 50,000 and customizable alpha, beta, and epsilon parameters.

## Training Process
The training process involves iterative learning, with each iteration improving the agent's policy based on its performance in the environment. The training stops after a predefined number of iterations or once the agent consistently achieves a high reward.

## Starting State [if applicable]
Each episode starts with a predefined neutral state where the soil moisture and nutrient levels are set at 0.5, and the reserves for water and nutrients are full. The crop begins at the initial growth stage.

## Episode End [if applicable]
An episode terminates under two conditions:
1. The crop reaches full growth, indicating a successful harvest.
2. Both water and nutrient reserves fall below critical levels, signifying resource depletion.

## Results
The results section should include detailed analysis and visualizations such as:
- **Learning Curves**: Displaying the agent's performance over time.
- **Resource Usage**: Graphs showing the efficiency in water and nutrient utilization.
- **Comparison to Baselines**: Comparing the RL agent's performance with traditional, non-RL methods.
- **Episode Length Analysis**: Insights into the average duration of episodes and the conditions leading to their termination.

## Conclusion
This project demonstrates the potential of Reinforcement Learning in agriculture management, showcasing how intelligent agents can optimize resource usage and crop growth. The environment and algorithms used here could be further extended to other domains of resource management and environmental control.
