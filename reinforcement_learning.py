from stable_baselines3 import DQN, PPO
import matplotlib.pyplot as plt
import gym
from gym import spaces
import numpy as np

import gym
from gym import spaces
import numpy as np

class WineQualityEnv(gym.Env):
    def __init__(self):
        super(WineQualityEnv, self).__init__()

        # Action space: continuous values between -1 (decrease) and 1 (increase) for each of the 11 properties
        self.action_space = spaces.Box(low=np.array([-1] * 11), high=np.array([1] * 11), dtype=np.float32)

        # Observation space: 11 chemical properties (scaled between 0 and 10)
        self.observation_space = spaces.Box(low=np.array([0.0] * 11), high=np.array([10.0] * 11), dtype=np.float32)

        # Initial wine properties (chemical properties)
        self.state = np.random.uniform(0, 10, size=11)  # Initial random state for wine
        self.done = False

    def reset(self):
        # Reset the environment and return the initial state
        self.state = np.random.uniform(0, 10, size=11)
        self.done = False
        return self.state

    def step(self, action):
        # Apply the action to the state (wine chemical properties)
        # The action is a continuous adjustment for each property
        self.state = np.clip(self.state + action, 0, 10)  # Ensure properties stay within the range [0, 10]

        # Calculate the reward based on wine quality (the last property is the quality)
        quality = self.state[-1]  # Last property is quality
        reward = self._calculate_reward(quality)

        # Check if done (e.g., if wine quality reaches a certain threshold)
        if quality <= 0:
            self.done = True

        return self.state, reward, self.done, {}

    def _calculate_reward(self, quality):
        # Reward is based on the quality score (0 to 10)
        # Use a more granular reward system based on the quality
        if quality >= 7:
            return 1  # High reward for excellent wine
        elif quality >= 5:
            return 0.5  # Moderate reward for average wine
        elif quality > 0:
            return -0.5  # Small penalty for low quality
        else:
            return -1  # Large penalty for poor quality

    def render(self):
        # Render the environment (e.g., print current state)
        print(f"Current wine properties: {self.state}")
        print(f"Wine quality: {self.state[-1]}")


# Initialize the custom environment
env = WineQualityEnv()

# Create and train the PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)  # Train the model

# Save the trained model
model.save("ppo_wine_quality_model")

# Test the trained model
state = env.reset()
for _ in range(10):
    action, _states = model.predict(state)
    state, reward, done, info = env.step(action)
    print(f"Action: {action}, Quality: {state[-1]}, Reward: {reward}")
    env.render()
    if done:
        break