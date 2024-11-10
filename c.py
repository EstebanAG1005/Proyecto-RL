import pygame
import numpy as np
from typing import Tuple
from highway_env import HighwayEnv,HighwayVisualizer

def run_simulation():
    # Crear el entorno con la simulación ralentizada
    env = HighwayEnv(num_lanes=4, num_other_vehicles=5)
    visualizer = HighwayVisualizer(env)
    
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, _ = env.reset()
        
        keys = pygame.key.get_pressed()
        acceleration = 0.0
        lane_change = 0.0
        
        if keys[pygame.K_UP]:
            acceleration = 1.0
        if keys[pygame.K_DOWN]:
            acceleration = -1.0
        if keys[pygame.K_LEFT]:
            lane_change = -1.0
        if keys[pygame.K_RIGHT]:
            lane_change = 1.0
        
        action = np.array([acceleration, lane_change])
        obs, reward, done, _, info = env.step(action)
        
        visualizer.render()
        
        clock.tick(30)  # Mantener 30 FPS
        
        if done:
            print(f"Episodio terminado. Reward final: {reward}")
            obs, _ = env.reset()
    
    pygame.quit()

if __name__ == "__main__":
    run_simulation()