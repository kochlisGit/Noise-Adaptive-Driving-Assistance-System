import random
import time
import carla
import cv2
from nadas.maps.towns.town02 import Town02
from nadas.maps.towns.town03 import Town03
from nadas.maps.towns.town04 import Town04
from nadas.maps.towns.town10 import Town10
from nadas.npcs.vehicle_manager import VehicleNPCManager
from nadas.npcs.walker_manager import WalkerNPCManager
from nadas.sensors.lkas_manager import LKASSensorManager
from nadas.sensors.acc_manager import ACCSensorManager


def main(
        server_ip: str,
        port: int,
        towns: list,
        num_iterations: int,
        sensor_memory_size: int,
        sensor_data_corrupt_prob: float,
        segmentation_noise_ratio: float,
        segmentation_corrupt_portion_size: tuple or None,
        depth_error_rate: float,
        hazard_distance: float,
        use_state_prediction: bool,
        debug: bool,
        store_directory: str or None
):
    vehicle_manager = None
    walker_manager = None
    lkas_sensor_manager = None
    acc_sensor_manager = None

    for _ in range(num_iterations):
        town = random.choice(towns)

        client = carla.Client(server_ip, port)
        client.set_timeout(10)
        world = client.load_world(town.name)
        blueprints_library = world.get_blueprint_library()

        vehicle_bp = carla.BlueprintLibrary = blueprints_library.filter("model3")[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        sensor_spawn_point = carla.Transform(carla.Location(x=0.8, z=1.5), carla.Rotation(pitch=-10))

        if vehicle_manager is None:
            vehicle_manager = VehicleNPCManager(blueprint_library=blueprints_library)

        if walker_manager is None:
            walker_manager = WalkerNPCManager(blueprint_library=blueprints_library)

        if lkas_sensor_manager is None:
            lkas_sensor_manager = LKASSensorManager(
                blueprints_library,
                sensor_memory_size=sensor_memory_size,
                sensor_data_corrupt_prob=sensor_data_corrupt_prob,
                segmentation_noise_ratio=segmentation_noise_ratio,
                use_state_prediction=use_state_prediction,
                debug=debug,
                store_directory=store_directory
            )
        if acc_sensor_manager is None:
            acc_sensor_manager = ACCSensorManager(
                blueprints_library=blueprints_library,
                sensor_memory_size=sensor_memory_size,
                sensor_data_corrupt_prob=sensor_data_corrupt_prob,
                segmentation_corrupt_portion_size=segmentation_corrupt_portion_size,
                segmentation_noise_ratio=segmentation_noise_ratio,
                depth_error_rate=depth_error_rate,
                hazard_distance=hazard_distance,
                use_state_prediction=use_state_prediction,
                debug=debug,
                store_directory=store_directory
            )

        # sensor_manager = random.choice([lkas_sensor_manager, acc_sensor_manager])
        sensor_manager = acc_sensor_manager

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_time = 0.01
        world.apply_settings(settings)

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        world.tick()
        sensor_manager.spawn_sensors(vehicle=vehicle, sp_transform=sensor_spawn_point, world=world)

        vehicle.set_autopilot(True)
        world.tick()

        vehicle_manager.spawn_npcs(client=client, world=world, num_npcs=town.num_vehicles)
        walker_manager.spawn_npcs(client=client, world=world, num_npcs=town.num_walkers)

        start = time.time()
        while time.time() - start < 60:
            world.tick()
            sensor_manager.get_observations()

        vehicle_manager.destroy_npcs(client=client, world=world)
        walker_manager.destroy_npcs(client=client, world=world)
        sensor_manager.destroy_sensors(world=world)

        vehicle.set_autopilot(False)
        world.tick()

        vehicle.destroy()
        world.tick()

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_time = None
        world.apply_settings(settings)
        world.tick()

        if debug:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main(
        server_ip='localhost',
        port=2000,
        towns=[Town02(), Town03(), Town04(), Town10()],
        num_iterations=8,
        sensor_memory_size=5,
        segmentation_corrupt_portion_size=(10, 15),
        depth_error_rate=0.1,
        hazard_distance=7,
        sensor_data_corrupt_prob=0.2,
        segmentation_noise_ratio=0.2,
        use_state_prediction=True,
        debug=True,
        store_directory=None
    )
