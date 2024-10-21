import carla
import random

class TrafficManager:
    def __init__(self, client, world):
        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []

    def spawn_vehicles(self, number_of_vehicles=10):  
        vehicle_blueprints = list(self.blueprint_library.filter('vehicle.*'))
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        for spawn_point in spawn_points[:number_of_vehicles]:
            vehicle_bp = random.choice(vehicle_blueprints)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                vehicle.set_autopilot(True)
                self.vehicles_list.append(vehicle)

    def spawn_walkers(self, number_of_walkers=20):  
        walker_blueprints = self.blueprint_library.filter('walker.pedestrian.*')
        batch = []
        walker_speed = []

        for _ in range(number_of_walkers):
            location = self.world.get_random_location_from_navigation()
            if location:
                spawn_point = carla.Transform(location)
                walker_bp = random.choice(walker_blueprints)
                walker_speed.append(1.4)  
                batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        self.client.apply_batch_sync(batch, True)

    def destroy_all(self):
        for i in range(0, len(self.walkers_list)):
            controller = self.world.get_actor(self.all_id[i + len(self.walkers_list)])
            if controller is not None:
                controller.stop()

        actors = self.world.get_actors(self.all_id)
        actors.destroy()

        for vehicle in self.vehicles_list:
            if vehicle is not None:
                vehicle.destroy()
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
