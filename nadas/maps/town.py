import carla
from abc import ABC, abstractmethod
from nadas.carla.agents.navigation.local_planner import RoadOption


class Town(ABC):
    def __init__(
            self,
            name: str,
            num_vehicles: int,
            num_walkers: int,
            spawn_dest_pairs: list,
            start_locations: list or None
    ):
        self._name = name
        self._num_vehicles = num_vehicles
        self._num_walkers = num_walkers
        self._spawn_dest_pairs = spawn_dest_pairs
        self._start_locations = start_locations

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_vehicles(self) -> int:
        return self._num_vehicles

    @property
    def num_walkers(self) -> int:
        return self._num_walkers

    @property
    def spawn_dest_pairs(self) -> list:
        return self._spawn_dest_pairs

    @property
    def start_locations(self) -> list:
        return self._start_locations

    def generate_route_waypoints(self, world_map: carla.Map) -> list:
        if self.start_locations is None:
            raise NotImplementedError(f'Start locations Not implemented on this map: {self.name}')

        routes_road_option = []
        for i, t_n in enumerate(self._start_locations):
            transform, n_waypoints = t_n
            wp = world_map.get_waypoint(transform.location, project_to_road=True)
            routes_road_option.append([(wp, RoadOption.LANEFOLLOW)])
            for j in range(n_waypoints):
                wp = wp.next(4)[0]
                routes_road_option[i].append((wp, RoadOption.LANEFOLLOW))
        return routes_road_option
