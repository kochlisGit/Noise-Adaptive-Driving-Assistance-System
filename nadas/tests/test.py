import random
import time
import carla
import cv2
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
        town = random.choice(towns)
        client = carla.Client(server_ip, port)
        client.set_timeout(10)
        world = client.load_world(town.name)
        wmap = world.get_map()
        spawnpoints = wmap.get_spawn_points()

        for sp in town.spawn_dest_pairs:
            print(spawnpoints[sp[0]].location)


if __name__ == '__main__':
    main(
        server_ip='localhost',
        port=2000,
        towns=[Town10()],
        num_iterations=8,
        sensor_memory_size=5,
        segmentation_corrupt_portion_size=(60, 80),
        depth_error_rate=0.1,
        hazard_distance=7,
        sensor_data_corrupt_prob=0.2,
        segmentation_noise_ratio=0.2,
        use_state_prediction=True,
        debug=True,
        store_directory=None
    )
