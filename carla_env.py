import gymnasium as gym
from gymnasium import spaces
import carla
import numpy as np
import cv2
import time
from traffic_manager import TrafficManager
from agents.navigation.controller import VehiclePIDController  # Usar el controlador PID que compartiste

class CarlaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CarlaEnv, self).__init__()

        # Conexión con CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)

        # Cargar el mundo
        self.world = self.client.load_world('Town01')

        # Configurar el clima
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=70.0
        )
        self.world.set_weather(weather)

        # Limpiar el mundo de otros actores
        self._clear_world()

        # Inicializar el TrafficManager y generar tráfico
        self.traffic_manager = TrafficManager(self.client, self.world)
        self.traffic_manager.spawn_vehicles(number_of_vehicles=30)
        self.traffic_manager.spawn_walkers(number_of_walkers=50)

        # Definir espacios de acción y observación
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(240, 320, 3),
                dtype=np.uint8
            ),
            'velocity': spaces.Box(
                low=0,
                high=100,
                shape=(1,),
                dtype=np.float32
            )
        })

        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.image = None
        self.collision_history = []

        self._setup_vehicle()

        # Configurar el controlador PID para el vehículo
        args_lateral = {'K_P': 1.5, 'K_D': 0.1, 'K_I': 0.05}  # Ajustar parámetros para mejor control
        args_longitudinal = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0}
        self.vehicle_controller = VehiclePIDController(self.vehicle, args_lateral, args_longitudinal)


        # Establecer la velocidad objetivo
        self.target_speed = 30  # km/h

        # Configurar la ruta
        self._set_up_route()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reiniciar el entorno
        self._setup_vehicle()
        self.collision_history = []  # Vaciar el historial de colisiones
        self._set_up_route()
        observation = self._get_observation()
        info = {}

        return observation, info


    def step(self, action):
        # Verificar si el vehículo existe y está vivo
        if self.vehicle is None or not self.vehicle.is_alive:
            print("El vehículo ha sido destruido. Reiniciando el entorno.")
            observation, info = self.reset()
            reward = -100  # Penalización por destrucción del vehículo
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated, info

        # Obtener la velocidad objetivo desde la acción proporcionada
        target_speed = self.target_speed * action[0]

        # Verificar si quedan waypoints en la ruta
        if len(self.route) > 0:
            target_waypoint = self.route.pop(0)  # Consumir el siguiente waypoint
        else:
            # Si se acaba la ruta, no reiniciar inmediatamente
            observation = self._get_observation()
            reward = 0
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated, {}

        # Utilizar el controlador PID para controlar el vehículo
        try:
            # Verificar nuevamente si el vehículo sigue existiendo antes de aplicar el control
            if self.vehicle.is_alive:
                control = self.vehicle_controller.run_step(target_speed, target_waypoint)
                self.vehicle.apply_control(control)  # Aplicar el control del PID
            else:
                print("El vehículo fue destruido justo antes de aplicar el control.")
                observation, info = self.reset()
                reward = -100  # Penalización por destrucción del vehículo
                terminated = True
                truncated = False
                return observation, reward, terminated, truncated, info

        except RuntimeError as e:
            print(f"Error al aplicar el control: {e}. Reiniciando el entorno.")
            observation, info = self.reset()
            reward = -100  # Penalización por destrucción del vehículo
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated, info

        # Avanzar la simulación
        self.world.tick()

        # Actualizar la cámara del espectador
        self._update_spectator_camera()

        # Obtener observación
        observation = self._get_observation()

        # Calcular recompensa
        reward = self._compute_reward()

        # Verificar si el episodio ha terminado
        terminated = self._is_done()
        truncated = False

        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self._destroy_actors()
        self.traffic_manager.destroy_all()
        cv2.destroyAllWindows()

    def _setup_vehicle(self):
        self._destroy_actors()

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        spawn_points = self.world.get_map().get_spawn_points()

        # Intentar spawnear el vehículo en varios puntos de spawn
        for spawn_point in spawn_points:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                print(f"Vehículo spawneado en {spawn_point}")
                break
        if self.vehicle is None:
            raise Exception("No se pudo spawnear el vehículo en ningún punto.")

        # Configurar sensores
        self._setup_sensors()

        self._set_spectator_camera()
        time.sleep(1)

    def _setup_sensors(self):
        blueprint_library = self.world.get_blueprint_library()

        # Sensor de cámara
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '320')
        camera_bp.set_attribute('image_size_y', '240')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(lambda data: self._process_image(data))

        # Sensor de colisión
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))


    def _set_up_route(self):
        """Define una ruta simple basada en los waypoints del mapa."""
        self.map = self.world.get_map()
        start_waypoint = self.map.get_waypoint(self.vehicle.get_location())

        # Generar waypoints hacia adelante
        self.route = []
        current_waypoint = start_waypoint
        for _ in range(200):  # Ajustar el número de waypoints
            self.route.append(current_waypoint)
            next_waypoints = current_waypoint.next(2.0)  # Avanza 2 metros por waypoint
            if next_waypoints:
                current_waypoint = next_waypoints[0]
            else:
                break

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.image = array

    def _on_collision(self, event):
        actor_we_collided_against = event.other_actor
        impulse = event.normal_impulse
        print(f"Colisión con {actor_we_collided_against.type_id}, impulso: {impulse}")
        self.collision_history.append(event)

        # Evitar el reinicio inmediato tras una colisión para que no destruya al vehículo
        # Puedes ajustar esta lógica si deseas evitar que las colisiones terminen el episodio.
        if len(self.collision_history) > 0:
            print("Colisión detectada. Penalización aplicada, pero no reiniciar.")
            # No reiniciar el entorno por colisión
            # self.reset()  # Desactivar el reinicio automático

    def _get_observation(self):
        if self.image is None:
            img = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            img = self.image
        velocity = np.array([self._get_speed()], dtype=np.float32)
        return {'image': img, 'velocity': velocity}

    def _compute_reward(self):
        reward = 0.0
        # Penalización por colisión
        if len(self.collision_history) > 0:
            reward -= 100
            # No vaciar el historial de colisiones inmediatamente
            # Esto permite que las colisiones no causen un reinicio inmediato, pero sí penalizan el score
            # self.collision_history = []  

        # Penalización por velocidad desviada del objetivo
        speed = self._get_speed()
        speed_error = abs(speed - self.target_speed)
        reward -= speed_error * 0.1

        # Recompensa por estar en la velocidad correcta
        if reward == 0.0:
            reward += 1.0  # Recompensa pequeña por moverse sin incidentes

        return reward


    def _is_done(self):
        # No terminar el episodio por colisión para evitar reinicios constantes
        # Solo terminar si el vehículo se sale de la ruta o el tiempo se acaba
        return False  # Puedes cambiar esta lógica según lo que defina el término del episodio en tu caso.



    def _get_speed(self):
        if self.vehicle is None or not self.vehicle.is_alive:
            return 0.0
        vel = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return speed

    def _destroy_actors(self):
        actors = [self.camera_sensor, self.collision_sensor, self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def _clear_world(self):
        actors = self.world.get_actors()
        for actor in actors:
            if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.'):
                actor.destroy()

    def _set_spectator_camera(self):
        vehicle_transform = self.vehicle.get_transform()
        camera_height = 20.0
        camera_location = carla.Location(
            x=vehicle_transform.location.x,
            y=vehicle_transform.location.y,
            z=vehicle_transform.location.z + camera_height
        )
        camera_rotation = carla.Rotation(pitch=-90, yaw=vehicle_transform.rotation.yaw, roll=0)
        camera_transform = carla.Transform(camera_location, camera_rotation)
        spectator = self.world.get_spectator()
        spectator.set_transform(camera_transform)

    def _update_spectator_camera(self):
        self._set_spectator_camera()
