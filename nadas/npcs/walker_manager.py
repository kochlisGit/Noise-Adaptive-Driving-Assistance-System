import carla
import random
from nadas.npcs.npc_manager import NPCManager


class WalkerNPCManager(NPCManager):
    def __init__(self, blueprint_library: carla.BlueprintLibrary):
        super().__init__(blueprint_library=blueprint_library)

        self._walker_id_list = None
        self._walker_controller_id_list = None

    def _get_npc_blueprints(self, blueprint_library: carla.BlueprintLibrary) -> list:
        self._walker_controller_bp = blueprint_library.filter('controller.ai.walker')[0]
        return blueprint_library.filter('walker.pedestrian.*')

    def _get_npc_spawn_points(self, world: carla.World, num_npcs: int):
        return [
            carla.Transform(world.get_random_location_from_navigation(), carla.Rotation())
            for _ in range(num_npcs)
        ]

    def _get_npc_actor_ids(self) -> list:
        return self._walker_id_list

    def _spawn_npcs(self, client: carla.Client, world: carla.World, blueprints: list, spawn_points: list or None):
        # Spawn walkers
        self._walker_id_list = self._spawn_actors(client=client, spawn_points=spawn_points, blueprints=blueprints)
        world.tick()

        # Spawn controllers
        spawn_batch = [
            carla.command.SpawnActor(self._walker_controller_bp, carla.Transform(), parent_id=walker_id)
            for walker_id in self._walker_id_list
        ]
        batch_results = client.apply_batch_sync(spawn_batch, True)
        self._walker_controller_id_list = [result.actor_id for result in batch_results if not result.error]
        world.tick()

        # Enable walker navigation
        walker_controller_list = world.get_actors(self._walker_controller_id_list)
        for controller in walker_controller_list:
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.uniform(1, 3))

    def _destroy_npcs(self, client: carla.Client, world: carla.World, actor_ids: list):
        # Retrieve walkers, disable their controllers, then delete both walkers and controllers
        walker_controller_list = world.get_actors(self._walker_controller_id_list)
        for controller in walker_controller_list:
            controller.stop()

        self._destroy_actors(client=client, actor_ids=self._walker_controller_id_list)
        world.tick()

        self._destroy_actors(client=client, actor_ids=actor_ids)

        self._walker_id_list = None
        self._walker_controller_id_list = None
