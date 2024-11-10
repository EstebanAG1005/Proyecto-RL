import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HighwayEnv(gym.Env):
    def __init__(self, n_lanes=4, road_length=1000, max_speed=10, safe_distance=10):
        super(HighwayEnv, self).__init__()
        
        # Configuración del entorno
        self.n_lanes = n_lanes
        self.road_length = road_length
        self.max_speed = max_speed
        self.safe_distance = safe_distance
        
        # Inicialización de posiciones y velocidades de vehículos
        self.agent_position = [0, n_lanes // 2]  # (pos_x, lane)
        self.agent_speed = 5
        self.vehicles = self.initialize_vehicles()
        
        # Definición del espacio de observación y acciones
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -self.max_speed, 0, 0]),
            high=np.array([self.road_length, self.n_lanes - 1, self.max_speed, self.road_length, 1]),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)  # 0: mantener, 1: acelerar/frenar, 2: cambiar de carril

    def initialize_vehicles(self):
        # Distribuye vehículos en el camino
        vehicles = []
        for lane in range(self.n_lanes):
            # Cada carril tiene un vehículo inicial aleatorio adelante del agente
            position_x = np.random.randint(self.agent_position[0] + 20, self.road_length)
            speed = np.random.randint(1, self.max_speed)
            vehicles.append([position_x, lane, speed])
        return vehicles

    def reset(self):
        self.agent_position = [0, self.n_lanes // 2]
        self.agent_speed = 5
        self.vehicles = self.initialize_vehicles()
        
        return self._get_observation()

    def _get_observation(self):
        # Obtener la posición relativa y velocidad del vehículo más cercano en cada carril
        observation = []
        for lane in range(self.n_lanes):
            lane_vehicles = [v for v in self.vehicles if v[1] == lane]
            if lane_vehicles:
                closest_vehicle = min(lane_vehicles, key=lambda v: v[0] - self.agent_position[0] if v[0] > self.agent_position[0] else self.road_length)
                dist = closest_vehicle[0] - self.agent_position[0]
                relative_speed = closest_vehicle[2] - self.agent_speed
                occupied = 1 if dist < self.safe_distance else 0
            else:
                dist = self.road_length
                relative_speed = 0
                occupied = 0
            observation.extend([dist, relative_speed, occupied])
        
        observation.extend([self.agent_position[0], self.agent_position[1], self.agent_speed])
        return np.array(observation, dtype=np.float32)

    def step(self, action):
        # Aplicar acción: 0 = mantener, 1 = acelerar/frenar, 2 = cambiar de carril
        if action == 1:
            # Acelerar o frenar
            self.agent_speed = min(self.max_speed, max(1, self.agent_speed + np.random.choice([-1, 1])))
        elif action == 2:
            # Cambio de carril si está libre
            new_lane = self.agent_position[1] + np.random.choice([-1, 1])
            if 0 <= new_lane < self.n_lanes:
                adjacent_vehicles = [v for v in self.vehicles if v[1] == new_lane and 0 < v[0] - self.agent_position[0] < self.safe_distance]
                if not adjacent_vehicles:
                    self.agent_position[1] = new_lane

        # Actualizar posición del agente
        self.agent_position[0] += self.agent_speed
        
        # Actualizar posiciones de otros vehículos
        for vehicle in self.vehicles:
            vehicle[0] += vehicle[2]
            if vehicle[0] > self.road_length:
                vehicle[0] = 0
                vehicle[2] = np.random.randint(1, self.max_speed)

        # Calcular recompensa y verificar si el episodio termina
        reward = 1  # Recompensa básica
        done = False
        for vehicle in self.vehicles:
            if vehicle[1] == self.agent_position[1] and 0 < vehicle[0] - self.agent_position[0] < self.safe_distance:
                reward = -100  # Penalización por colisión
                done = True
                break
            elif vehicle[1] == self.agent_position[1] and self.agent_position[0] < vehicle[0] < self.agent_position[0] + self.safe_distance:
                reward -= 10  # Penalización por mantener poca distancia
            
        if self.agent_position[0] >= self.road_length:
            done = True  # Fin del episodio cuando el agente recorre toda la distancia
            
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Agent Position: {self.agent_position}, Speed: {self.agent_speed}")
        for vehicle in self.vehicles:
            print(f"Vehicle Position: {vehicle[0]}, Lane: {vehicle[1]}, Speed: {vehicle[2]}")
