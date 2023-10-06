import math
import random
import carla
from nadas.carla.agents.navigation.basic_agent import BasicAgent


class AgentController(BasicAgent):
    def __init__(
            self,
            vehicle: carla.Vehicle,
            max_brake: float or None = None,
            max_throttle: float or None = None,
            max_steering: float or None = None
    ):
        opt_dict = {
            'ignore_traffic_lights': True,
            'ignore_stop_signs': True,
            'ignore_vehicles': False
        }

        if max_brake is not None:
            opt_dict['max_brake'] = max_brake
        if max_throttle is not None:
            opt_dict['max_throttle'] = max_throttle
        if max_steering is not None:
            opt_dict['max_steering'] = max_steering

        super().__init__(vehicle=vehicle, target_speed=random.choice([20, 30, 50, 60]), opt_dict=opt_dict)
        self._local_planner = super().get_local_planner()
        self._vehicle = vehicle

    def get_pid_control(self) -> carla.VehicleControl:
        # Get planner control
        control = self._local_planner.run_step()

        vel = self._vehicle.get_velocity()
        vehicle_speed = 3.6*math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Check for possible vehicle obstacles
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            control = super().add_emergency_stop(control=control)
            return control

        # Check for possible walker obstacles
        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        max_walker_distance = self._base_vehicle_threshold + vehicle_speed
        affected_by_walker, _, _ = self._vehicle_obstacle_detected(walker_list, max_walker_distance)
        if affected_by_walker:
            control = super().add_emergency_stop(control=control)
            return control

        return control

    def set_route(self, waypoints_list):
        self._local_planner.set_global_plan(current_plan=waypoints_list)

    def destroy(self):
        self._local_planner.reset_vehicle()
