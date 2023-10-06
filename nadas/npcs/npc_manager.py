import random
from abc import ABC, abstractmethod
import carla


class NPCManager(ABC):
    def __init__(self, blueprint_library: carla.BlueprintLibrary):
        self._npc_blueprints = self._get_npc_blueprints(blueprint_library=blueprint_library)

    @abstractmethod
    def _get_npc_blueprints(self, blueprint_library: carla.BlueprintLibrary) -> list:
        pass

    @abstractmethod
    def _get_npc_spawn_points(self, world: carla.World, num_npcs: int):
        pass

    @abstractmethod
    def _get_npc_actor_ids(self) -> list or None:
        pass

    @abstractmethod
    def _spawn_npcs(self, client: carla.Client, world: carla.World, blueprints: list, spawn_points: list):
        pass

    @abstractmethod
    def _destroy_npcs(self, client: carla.Client, world: carla.World, actor_ids: list):
        pass

    @staticmethod
    def _spawn_actors(client: carla.Client, spawn_points: list, blueprints: list):
        spawn_batch = [carla.command.SpawnActor(bp, sp) for sp, bp in zip(spawn_points, blueprints)]
        batch_results = client.apply_batch_sync(spawn_batch, True)
        return [result.actor_id for result in batch_results if not result.error]

    @staticmethod
    def _destroy_actors(client: carla.Client, actor_ids: list):
        destroy_batch = [carla.command.DestroyActor(actor_id) for actor_id in actor_ids]
        client.apply_batch_sync(destroy_batch, True)

    def spawn_npcs(self, client: carla.Client, world: carla.World, num_npcs: int):
        if num_npcs < 1:
            return

        actor_ids = self._get_npc_actor_ids()

        if actor_ids is not None:
            raise RuntimeError(f'NPCs should be destroyed before spawning new NPC vehicles, got {len(actor_ids)} NPCs')

        self._spawn_npcs(
            client=client,
            world=world,
            blueprints=random.choices(population=self._npc_blueprints, k=num_npcs),
            spawn_points=self._get_npc_spawn_points(world=world, num_npcs=num_npcs)
        )
        world.tick()

    def destroy_npcs(self, client: carla.Client, world: carla.World):
        actor_ids = self._get_npc_actor_ids()

        if actor_ids is None:
            return

        self._destroy_npcs(client=client, world=world, actor_ids=actor_ids)
        world.tick()
