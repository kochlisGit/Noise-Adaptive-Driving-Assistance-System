import math
import carla


def vehicle_in_lane(wheels: list, world_map, lane_id) -> bool:
    # Get the waypoints of the front left and front right wheels
    wheel_fl_waypoint: carla.Waypoint = world_map.get_waypoint(wheels[0].position / 100)
    wheel_fr_waypoint: carla.Waypoint = world_map.get_waypoint(wheels[1].position / 100)

    if wheel_fl_waypoint.lane_id != lane_id:
        # Calculate the distance between the front left wheel and the projected location of the front right wheel on the road
        # If this distance is greater than the lane width of the front right wheel, then the vehicle is outside the lane
        if wheel_fl_waypoint.transform.location.distance_2d(
                world_map.get_waypoint(
                    wheel_fr_waypoint.transform.location,
                    project_to_road=True
                ).transform.location) > wheel_fr_waypoint.lane_width:
            return False

    # If the front right wheel is not in the initial lane, check if it is outside the lane width
    if wheel_fr_waypoint.lane_id != lane_id:
        # Calculate the distance between the front right wheel and the projected location of the front left wheel on the road
        # If this distance is greater than the lane width of the front left wheel, then the vehicle is outside the lane
        if wheel_fr_waypoint.transform.location.distance_2d(
                world_map.get_waypoint(
                    wheel_fl_waypoint.transform.location,
                    project_to_road=True
                ).transform.location
        ) > wheel_fl_waypoint.lane_width:
            return False

    # If both wheels are in the initial lane, return True
    return True


def get_direction_vehicle_road_product(vehicle_vector, road_waypoint: carla.Waypoint) -> float:
    next_waypoint = road_waypoint.next(distance=1.0)[0]
    road_vector = carla.Vector3D(
        x=next_waypoint.transform.location.x - road_waypoint.transform.location.x,
        y=next_waypoint.transform.location.y - road_waypoint.transform.location.y,
        z=road_waypoint.transform.location.z
    )
    return road_vector.dot_2d(vehicle_vector)


def out_of_lane(wheels: list, world_map, lane_id):
    wheel_fl_lane_id: int = world_map.get_waypoint(wheels[0].position/100).lane_id
    wheel_fr_lane_id: int = world_map.get_waypoint(wheels[1].position/100).lane_id
    return lane_id != wheel_fl_lane_id and lane_id != wheel_fr_lane_id


def get_vehicle_speed(vehicle: carla.Vehicle) -> float:
    vel = vehicle.get_velocity()
    return 3.6*math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
