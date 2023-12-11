import gymnasium as gym
# from gymnasium import spaces
from gymnasium.spaces import Discrete,Box
import numpy as np
import random

class AgricultureEnv(gym.Env):
    """
    A more complex agriculture environment with weather variability, water and nutrient reserves, and pests.
    The agent manages a single crop, considering these factors, to maximize growth while efficiently using resources.
    """

    def __init__(self):
        super(AgricultureEnv, self).__init__()

        # Define observation space: [soil moisture, nutrient level, crop growth, weather condition, water reserve, nutrient reserve]
        self.observation_space = Box(low=0, high=1, shape=(6,), dtype=np.float32)

        # Define action space: [0 = no action, 1 = water, 2 = fertilize]
        self.action_space = Discrete(3)

        # Initial conditions
        self.soil_moisture = 0.5
        self.nutrient_level = 0.5
        self.crop_growth = 0.0
        self.weather_condition = 0.5  # Represents weather impact, 0.5 as neutral
        self.water_reserve = 1.0  # Full reserve
        self.fertilizer_reserve = 1.0  # Full reserve
        self.pests = False
        self.reset()

    def step(self, action):
        # Update weather and pests
        self._update_weather()
        self._update_pests()

        # Apply action effects
        if action == 1 and self.water_reserve > 0.1:  # Watering
            self.soil_moisture = min(self.soil_moisture + 0.2, 1)
            self.water_reserve -= 0.1
        elif action == 2 and self.fertilizer_reserve > 0.1:  # Fertilizing
            self.nutrient_level = min(self.nutrient_level + 0.2, 1)
            self.fertilizer_reserve -= 0.1
            
        # Natural decrease in soil moisture and nutrient level
        self.soil_moisture = max(self.soil_moisture - 0.05, 0)
        self.nutrient_level = max(self.nutrient_level - 0.03, 0)

        # Crop growth is influenced by soil moisture, nutrient level, weather, and pests
        growth_factor = (self.soil_moisture + self.nutrient_level + self.weather_condition) / 30
        growth_factor -= 0.1 if self.pests else 0
        self.crop_growth += max(growth_factor, 0)

        # Calculate reward
        reward = self.crop_growth - (abs(0.5 - self.soil_moisture) + abs(0.5 - self.nutrient_level)) / 10

        # Check if it's time to harvest (end of episode)
        done = self.crop_growth >= 1
        if done:
            self.crop_growth=1
        if self.water_reserve >0 and self.water_reserve <0.1 and  self.fertilizer_reserve >0 and self.fertilizer_reserve <0.1:
            done=True
            reward-=1
        

        # Update state
        self.state = np.array([self.soil_moisture, self.nutrient_level, self.crop_growth, self.weather_condition, self.water_reserve, self.fertilizer_reserve])

        return self.state, reward, done,False, {}

    def reset(self,*,seed=None, options=None):
        # Reset the environment to an initial state
        self.soil_moisture = 0.5
        self.nutrient_level = 0.5
        self.crop_growth = 0.0
        self.weather_condition = 0.5
        self.water_reserve = 1.0
        self.fertilizer_reserve = 1.0
        self.pests = False
        self.state = np.array([self.soil_moisture, self.nutrient_level, self.crop_growth, self.weather_condition, self.water_reserve, self.fertilizer_reserve])
        return self.state,{}

    def _update_weather(self):
        # Simulate weather changes
        weather_effect = random.choice([-0.2, 0, 0.2])
        self.weather_condition = max(min(self.weather_condition + weather_effect, 1), 0)

    def _update_pests(self):
        # Simulate pest occurrence
        self.pests = random.random() < 0.1  # 10% chance of pests

    def render(self, mode='console'):
        if mode == 'console':
            print("==== Agriculture Environment State ====")
            print(f" Soil Moisture    : {self.soil_moisture:.2f}")
            print(f" Nutrient Level   : {self.nutrient_level:.2f}")
            print(f" Crop Growth      : {self.crop_growth:.2f}")
            print(f" Weather Condition: {self.weather_condition:.2f}")
            print(f" Water Reserve    : {self.water_reserve:.2f}")
            print(f" Fertilizer Reserve: {self.fertilizer_reserve:.2f}")
            print(f" Pests Present    : {'Yes' if self.pests else 'No'}")
            print("======================================")

    def close(self):
        pass

# Example of how to use the environment
if __name__ == "__main__":
    env = AgricultureEnv()

    for episode in range(10):  # Run 10 episodes for demonstration
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Choose a random action
            state, reward, done, _ = env.step(action)
            env.render()  # Render the state of the environment

    env.close()