import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.appo import APPO
from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.bc import BC
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms import impala
from ray.rllib.algorithms.marwil import marwil
# from ray.rllib.algorithms.appo import APPO
from agriculture_env import AgricultureEnv  # Make sure to import your custom env

def train_agriculture_env():
    # Register the custom environment with RLlib
    tune.register_env("AgricultureEnv", lambda config: AgricultureEnv())

    # Initialize Ray
    ray.init()

    # Configuration for the training
    config = {
        "env": "AgricultureEnv",
        "num_gpus": 0,
        "num_workers": 7,
        "framework": "tf",
        # Add more configurations as needed
        "replay_buffer_config": {
            "type": "PrioritizedReplayBuffer",
            "capacity": 50000,  # Adjust the capacity as needed
            "prioritized_replay_alpha": 0.6,  # Adjust alpha (how much prioritization is used) as needed
            "prioritized_replay_beta": 0.4,  # Adjust beta (importance-sampling weight) as needed
            "prioritized_replay_eps": 1e-6  # Small value to avoid zero priorities
        }
    }

    # Start training
    stop_criteria = {
        "training_iteration": 50,  # You can modify this as needed
    }

    results = tune.run(DQN, config=config, stop=stop_criteria)

    # Retrieve the best trial using the "max" mode for "episode_reward_mean" metric
    best_trial = results.get_best_trial("episode_reward_mean", mode="max")

    # Optionally save the trained model
    checkpoint_path = results.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
    print(f"Best model checkpoint saved at: {checkpoint_path}")

if __name__ == "__main__":
    train_agriculture_env()