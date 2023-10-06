import random
import time
import carla
from nadas.maps.towns.town02 import Town02
from nadas.maps.towns.town03 import Town03
from nadas.maps.towns.town04 import Town04
from nadas.maps.towns.town10 import Town10
from nadas.npcs.vehicle_manager import VehicleNPCManager
from nadas.npcs.walker_manager import WalkerNPCManager


def main(
        server_ip: str,
        port: int,
        towns: list,
        num_iterations: int
):
    vehicle_manager = None
    walker_manager = None

    for _ in range(num_iterations):
        town = random.choice(towns)

        client = carla.Client(server_ip, port)
        client.set_timeout(10)
        world = client.load_world(town.name)
        blueprints_library = world.get_blueprint_library()

        if vehicle_manager is None:
            vehicle_manager = VehicleNPCManager(blueprint_library=blueprints_library)

        if walker_manager is None:
            walker_manager = WalkerNPCManager(blueprint_library=blueprints_library)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_time = 0.01
        world.apply_settings(settings)

        vehicle_manager.spawn_npcs(client=client, world=world, num_npcs=town.num_vehicles)
        walker_manager.spawn_npcs(client=client, world=world, num_npcs=town.num_walkers)

        start = time.time()
        while time.time() - start < 30:
            world.tick()

        vehicle_manager.destroy_npcs(client=client, world=world)
        walker_manager.destroy_npcs(client=client, world=world)

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_time = None
        world.apply_settings(settings)
        world.tick()


if __name__ == '__main__':
    main(
        server_ip='localhost',
        port=2000,
        towns=[Town02(), Town03(), Town04(), Town10()],
        num_iterations=5
    )
