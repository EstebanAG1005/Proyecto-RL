import gymnasium as gym
from stable_baselines3 import PPO
from carla_env import CarlaEnv

def main():
    env = CarlaEnv()

    model = PPO(
        'MultiInputPolicy',
        env,
        n_steps=256,
        batch_size=64,
        learning_rate=1e-4,
        verbose=1,
        tensorboard_log="./ppo_carla_tensorboard/"
    )

    model.learn(total_timesteps=10000)

    model.save("ppo_carla_model")

    env.close()

if __name__ == '__main__':
    main()
