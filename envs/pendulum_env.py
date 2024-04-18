import gymnasium as gym

class Pendulum_Env:

    def __init__(self, max_episode_steps=None, render_mode=None): # render_mode='human'
        self.gym_env = gym.make("Pendulum-v1", max_episode_steps=max_episode_steps, render_mode=render_mode)

    def reset(self, seed=None):
        observation, info = self.gym_env.reset(seed=seed)
        return observation
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.gym_env.step(action)
        return observation, reward, terminated, info


        


observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()