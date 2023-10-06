import carla
import random
from nadas.npcs.npc_manager import NPCManager


class VehicleNPCManager(NPCManager):
    def __init__(self, blueprint_library: carla.BlueprintLibrary):
        super().__init__(blueprint_library=blueprint_library)

        self._vehicle_id_list = None

    def _get_npc_blueprints(self, blueprint_library: carla.BlueprintLibrary) -> list:
        return blueprint_library.filter('vehicle.*')

    def _get_npc_spawn_points(self, world: carla.World, num_npcs: int):
        return random.sample(population=world.get_map().get_spawn_points(), k=num_npcs)

    def _get_npc_actor_ids(self) -> list:
        return self._vehicle_id_list

    def _spawn_npcs(self, client: carla.Client, world: carla.World, blueprints: list, spawn_points: list):
        # Spawn vehicles
        self._vehicle_id_list = super()._spawn_actors(client=client, spawn_points=spawn_points, blueprints=blueprints)
        world.tick()

        # Enable vehicle autopilot
        autopilot_batch = [carla.command.SetAutopilot(actor_id, True) for actor_id in self._vehicle_id_list]
        client.apply_batch_sync(autopilot_batch, True)

    def _destroy_npcs(self, client: carla.Client, world: carla.World, actor_ids: list):
        # Disable vehicle autopilot
        autopilot_batch = [carla.command.SetAutopilot(actor_id, False) for actor_id in actor_ids]
        client.apply_batch_sync(autopilot_batch, True)
        world.tick()

        # Destroy vehicles
        super()._destroy_actors(client=client, actor_ids=actor_ids)
        self._vehicle_id_list = None
