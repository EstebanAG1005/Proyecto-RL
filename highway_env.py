import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Vehicle:
    """Clase para representar vehículos en la simulación."""
    x: float  # posición longitudinal
    y: int    # carril (0 a N-1)
    speed: float
    length: float = 4.5  # longitud del vehículo en metros
    width: float = 2.0   # ancho del vehículo en metros
    
class HighwayEnv(gym.Env):
    """Entorno de simulación de conducción en autopista multicarril."""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, num_lanes: int = 4, num_other_vehicles: int = 5):
        super(HighwayEnv, self).__init__()
        
        # Parámetros de configuración
        self.num_lanes = num_lanes
        self.num_other_vehicles = num_other_vehicles
        self.max_speed = 120.0  # km/h
        self.min_speed = 0.0    # km/h
        self.safe_distance = 20.0  # metros
        self.lane_width = 3.5   # metros
        self.simulation_frequency = 10  # Hz
        self.max_episode_steps = 1000
        self.max_distance = 1000.0  # metros
        
        # Conjunto para rastrear vehículos rebasados
        self.overtaken_vehicles = set()
        
        # Espacio de observación
        # [posición_x, posición_y, velocidad, 
        #  distancias a otros vehículos (num_lanes * 2),
        #  velocidades relativas (num_lanes * 2)]
        # num_features = 3 + self.num_lanes * 4
        # self.observation_space = spaces.Box(
        #     low=-float('inf'),
        #     high=float('inf'), 
        #     shape=(num_features,),
        #     dtype=np.float32
        # )

        num_features = 3 + self.num_lanes * 4
        obs_low = np.array(
            [-np.inf, 0, self.min_speed] + [0.0] * (self.num_lanes * 2) + [-self.max_speed] * (self.num_lanes * 2),
            dtype=np.float32
        )
        obs_high = np.array(
            [np.inf, self.num_lanes - 1, self.max_speed] + [self.max_distance] * (self.num_lanes * 2) + [self.max_speed] * (self.num_lanes * 2),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        
        # Espacio de acción
        # [aceleración (-1 a 1), cambio de carril (-1, 0, 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Estado interno
        self.agent: Optional[Vehicle] = None
        self.vehicles: List[Vehicle] = []
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        """Reinicia el entorno al estado inicial."""
        super().reset(seed=seed)
        
        # Inicializar el agente
        self.agent = Vehicle(
            x=0.0,
            y=np.random.randint(0, self.num_lanes),
            speed=80.0  # velocidad inicial 80 km/h
        )
        
        # Inicializar otros vehículos
        self.vehicles = []
        for _ in range(self.num_other_vehicles):
            vehicle = Vehicle(
                x=np.random.uniform(50, 200),
                y=np.random.randint(0, self.num_lanes),
                speed=np.random.uniform(60, 100)
            )
            self.vehicles.append(vehicle)
            
        # Reiniciar conjunto de vehículos rebasados
        self.overtaken_vehicles = set()
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Ejecuta un paso de simulación."""
        self.current_step += 1
        
        # Desempaquetar acciones
        acceleration = action[0]  # -1 a 1
        lane_change = action[1]   # -1 a 1
        
        # Actualizar estado del agente
        # Actualizar velocidad
        speed_change = acceleration * 5.0  # 5 m/s² máximo
        self.agent.speed = np.clip(
            self.agent.speed + speed_change,
            self.min_speed,
            self.max_speed
        )
        
        # Cambio de carril
        if abs(lane_change) > 0.5:  # Umbral para cambio de carril
            new_lane = self.agent.y + np.sign(lane_change)
            if 0 <= new_lane < self.num_lanes:
                self.agent.y = new_lane
        
        # Actualizar posición
        self.agent.x += self.agent.speed / 3.6  # convertir km/h a m/s
        
        # Actualizar otros vehículos
        for vehicle in self.vehicles:
            vehicle.x += vehicle.speed / 3.6
            
        # Verificar colisiones
        collision = self._check_collisions()
        
        # Calcular recompensa
        reward = self._compute_reward(collision)
        
        # Verificar si se han rebasado todos los vehículos
        all_overtaken = len(self.overtaken_vehicles) == len(self.vehicles)
        
        # Verificar terminación
        done = collision or self.current_step >= self.max_episode_steps or all_overtaken
        
        return self._get_observation(), reward, done, False, {"all_overtaken": all_overtaken}
    
    def _get_observation(self) -> np.ndarray:
        """Constructs the observation vector."""
        obs = [
            self.agent.x,
            self.agent.y,
            self.agent.speed
        ]
        
        # Initialize distances and relative speeds with finite values
        distances_front = [self.max_distance] * self.num_lanes
        distances_back = [self.max_distance] * self.num_lanes
        rel_speeds_front = [0.0] * self.num_lanes
        rel_speeds_back = [0.0] * self.num_lanes
        
        # Calculate distances and relative speeds
        for vehicle in self.vehicles:
            lane = int(vehicle.y)
            if 0 <= lane < self.num_lanes:
                dist = vehicle.x - self.agent.x
                rel_speed = vehicle.speed - self.agent.speed
                if dist >= 0:  # Vehicle ahead
                    if dist < distances_front[lane]:
                        distances_front[lane] = dist
                        rel_speeds_front[lane] = rel_speed
                else:  # Vehicle behind
                    dist = abs(dist)
                    if dist < distances_back[lane]:
                        distances_back[lane] = dist
                        rel_speeds_back[lane] = rel_speed
        
        # Clip distances to max_distance
        distances_front = [min(d, self.max_distance) for d in distances_front]
        distances_back = [min(d, self.max_distance) for d in distances_back]
        
        # Construct the full observation
        obs.extend(distances_front)
        obs.extend(distances_back)
        obs.extend(rel_speeds_front)
        obs.extend(rel_speeds_back)
        
        return np.array(obs, dtype=np.float32)

    def _check_collisions(self) -> bool:
        """Verifica si hay colisiones con otros vehículos."""
        for vehicle in self.vehicles:
            # Solo verificar colisión si están en el mismo carril
            if vehicle.y == self.agent.y:
                # Calcular los límites de cada vehículo
                agent_front = self.agent.x + self.agent.length / 2
                agent_back = self.agent.x - self.agent.length / 2
                vehicle_front = vehicle.x + vehicle.length / 2
                vehicle_back = vehicle.x - vehicle.length / 2
                
                # Hay colisión si los intervalos se solapan
                if not (agent_back > vehicle_front or agent_front < vehicle_back):
                    return True
                    
        return False
    
    def _update_overtaken_vehicles(self):
        """Actualiza el conjunto de vehículos rebasados."""
        for i, vehicle in enumerate(self.vehicles):
            # Si el agente está adelante del vehículo y no está registrado como rebasado
            if self.agent.x > vehicle.x + vehicle.length and i not in self.overtaken_vehicles:
                self.overtaken_vehicles.add(i)
    
    def _compute_reward(self, collision: bool) -> float:
        """Calcula la recompensa basada en el estado actual."""
        # Actualizar vehículos rebasados
        self._update_overtaken_vehicles()
        
        # Penalización por colisión
        if collision:
            return -100.0
            
        # Recompensa base
        reward = 0.0
        
        # Recompensa por mantener velocidad segura
        reward += 0.1 * (self.agent.speed / self.max_speed)
        
        # Penalización por distancia insegura
        min_distance = float('inf')
        for vehicle in self.vehicles:
            if vehicle.y == self.agent.y and vehicle.x > self.agent.x:
                distance = vehicle.x - self.agent.x
                min_distance = min(min_distance, distance)
        
        if min_distance < self.safe_distance:
            reward -= 0.5 * (self.safe_distance - min_distance) / self.safe_distance
        
        # Recompensa por rebasar vehículos
        if len(self.overtaken_vehicles) > 0:
            # Recompensa proporcional al número de vehículos rebasados
            reward += 0.2 * len(self.overtaken_vehicles)
            
        # Recompensa especial por rebasar todos los vehículos
        if len(self.overtaken_vehicles) == len(self.vehicles):
            reward += 1.0
        
        return reward
    
    def render(self):
        """Renderiza el estado actual del entorno."""
        # Implementar visualización si se necesita
        pass
    
    def close(self):
        """Cierra el entorno y libera recursos."""
        pass


class HighwayVisualizer:
    def __init__(self, env: HighwayEnv):
        pygame.init()
        self.env = env
        
        # Configuración de la ventana
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Highway Driving Simulation")
        
        # Colores
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        
        # Dimensiones y escalas
        self.lane_height = 80
        self.car_width = 60
        self.car_height = 40
        self.pixels_per_meter = 3  # Escala: 3 píxeles = 1 metro
        
        # Offset vertical para centrar los carriles
        self.vertical_offset = (self.height - (self.env.num_lanes * self.lane_height)) // 2
        
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convierte coordenadas del mundo a coordenadas de la pantalla."""
        # Posición horizontal relativa al agente, usando la escala de metros a píxeles
        screen_x = int((x - self.env.agent.x) * self.pixels_per_meter + self.width / 3)
        
        # Posición vertical centrada en el carril
        screen_y = int(self.vertical_offset + y * self.lane_height + self.lane_height / 2)
        
        return screen_x, screen_y
        
    def render(self):
        self.screen.fill(self.GRAY)
        
        # Dibujar líneas de carril
        for i in range(self.env.num_lanes + 1):
            y_pos = self.vertical_offset + i * self.lane_height
            pygame.draw.line(self.screen, self.WHITE, (0, y_pos), (self.width, y_pos), 3)
        
        def draw_vehicle(x, y, color):
            car_x, car_y = self.world_to_screen(x, y)
            car_rect = pygame.Rect(
                car_x - self.car_width // 2,
                car_y - self.car_height // 2,
                self.car_width,
                self.car_height
            )
            if 0 <= car_x <= self.width:  # Solo dibujar si está en pantalla
                pygame.draw.rect(self.screen, color, car_rect)
                pygame.draw.rect(self.screen, self.BLACK, car_rect, 2)
        
        # Dibujar otros vehículos
        for vehicle in self.env.vehicles:
            if 0 <= vehicle.y < self.env.num_lanes:
                draw_vehicle(vehicle.x, vehicle.y, self.RED)
        
        # Dibujar el agente
        if 0 <= self.env.agent.y < self.env.num_lanes:
            draw_vehicle(self.env.agent.x, self.env.agent.y, self.WHITE)
        
        # Mostrar información
        font = pygame.font.Font(None, 36)
        info_texts = [
            f"Speed: {self.env.agent.speed:.1f} km/h",
            f"Position: {self.env.agent.x:.1f} m",
            f"Lane: {self.env.agent.y + 1}/{self.env.num_lanes}"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (10, 10 + i * 40))
        
        pygame.display.flip()
