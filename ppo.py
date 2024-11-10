import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from highway_env import HighwayEnv  # Asegúrate de que el entorno esté en el path
# Evaluación y visualización
import numpy as np
import pygame
from highway_env import HighwayVisualizer


def make_env():
    return HighwayEnv(num_lanes=4, num_other_vehicles=5)

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_highway_tensorboard/"
)

TIMESTEPS = 100000

model.learn(total_timesteps=TIMESTEPS)

model.save("ppo_highway_model")



def evaluate_agent(model, env, episodes=5):
    visualizer = HighwayVisualizer(env.envs[0])
    clock = pygame.time.Clock()

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            visualizer.render()
            clock.tick(30)

    pygame.quit()

evaluate_agent(model, env)
