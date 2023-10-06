import carla
import gymnasium
import numpy as np
import random
from nadas.agents.agent import AgentController
from nadas.environments import utils
from nadas.maps.towns.town10 import Town10
from nadas.sensors.acc_manager import ACCSensorManager
from nadas.npcs.vehicle_manager import VehicleNPCManager
from nadas.npcs.walker_manager import WalkerNPCManager


class ACCEnvironment(gymnasium.Env):
    def __init__(self, env_config: dict):
        self._server_ip = env_config['server_ip']
        self._port = env_config['port']
        self._sensor_data_corrupt_prob = env_config['sensor_data_corrupt_prob']
        self._segmentation_corrupt_portion_size = env_config['segmentation_corrupt_portion_size']
        self._depth_error_rate = env_config['depth_error_rate']
        self._use_state_prediction = env_config['use_state_prediction']
        self._town = Town10()
        self._max_steps = env_config['max_steps']
        self._iterations_per_reload = env_config['iterations_per_reload']
        self._action_repeats = env_config['action_repeats']
        self._debug = False if 'debug' not in env_config else env_config['debug']
        self._store_sensor_directory = None if 'store_sensor_directory' not in env_config else \
            env_config['store_sensor_directory']

        self._max_brake = 0.7
        self._max_throttle = 0.7
        self._hazard_distance = 10
        self._min_safety_distance = 6
        self._sensor_memory_size = 10
        self._fixed_delta_time = 0.01
        self._sensor_sp_transform = carla.Transform(carla.Location(x=0.8, z=1.5), carla.Rotation(pitch=-10))

        # Define Action Space
        self.action_space = gymnasium.spaces.Box(
            low=-self._max_brake,
            high=self._max_throttle,
            shape=(1,),
            dtype=np.float32
        )

        # Camera Image Size as observation (Simple)
        self.observation_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(60, 80, 1), dtype=np.float32),
            'control': gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        })

        # Connect carla
        self._client = carla.Client(self._server_ip, self._port)
        self._client.set_timeout(20)
        self._world = self._client.get_world()
        self._world_map = self._world.get_map()
        self._world_spawn_points = self._world_map.get_spawn_points()

        # Get agent routes
        self._start_locations = self._town.start_locations
        self._num_routes = len(self._start_locations)
        self._routes = self._town.generate_route_waypoints(world_map=self._world_map)
        self._current_route_waypoint_list = None

        # Get blueprints
        blueprints_library = self._world.get_blueprint_library()
        self._vehicle_bp = blueprints_library.filter('model3')[0]

        # Initialize Sensor manager
        self._sensor_manager = ACCSensorManager(
            blueprints_library=blueprints_library,
            sensor_memory_size=self._sensor_memory_size,
            sensor_data_corrupt_prob=self._sensor_data_corrupt_prob,
            segmentation_corrupt_portion_size=self._segmentation_corrupt_portion_size,
            depth_error_rate=self._depth_error_rate,
            hazard_distance=self._hazard_distance,
            use_state_prediction=self._use_state_prediction,
            debug=self._debug,
            store_directory=self._store_sensor_directory
        )

        # Initialize NPC managers
        self._vehicle_manager = VehicleNPCManager(blueprint_library=blueprints_library)
        self._walker_manager = WalkerNPCManager(blueprint_library=blueprints_library)

        self._vehicle = None
        self._agent_controller = None
        self._end_location = None
        self._collision = None
        self._obstacle = (None, None)
        self._episode_step = self._max_steps
        self._iteration = 0

    def _synchronize(self):
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_time = self._fixed_delta_time
        self._world.apply_settings(settings)

    def _desynchronize(self):
        settings = self._world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_time = self._fixed_delta_time
        self._world.apply_settings(settings)

    def _spawn_actors(self):
        loc_ids = [i for i in range(self._num_routes)]

        # Spawn vehicle
        while self._vehicle is None:
            random.shuffle(loc_ids)

            for loc_id in loc_ids:
                spawn_point = self._start_locations[loc_id][0]

                # Randomly misplacing vehicle on lane
                sp = carla.Transform(
                    location=carla.Location(
                        x=spawn_point.location.x + np.random.uniform(-0.1, 0.1),
                        y=spawn_point.location.y + np.random.uniform(-0.1, 0.1),
                        z=spawn_point.location.z
                    ),
                    rotation=spawn_point.rotation
                )
                self._vehicle = self._world.try_spawn_actor(self._vehicle_bp, sp)
                self._world.tick()

                if self._vehicle is not None:
                    route = self._routes[loc_id]
                    self._current_route_waypoint_list = route
                    self._end_location = route[-1][0].transform.location
                    break

        # Spawn sensors
        self._sensor_manager.spawn_sensors(
            vehicle=self._vehicle,
            sp_transform=self._sensor_sp_transform,
            world=self._world
        )

        # Spawn NPCs
        if self._iteration % self._iterations_per_reload == 0:
            self._vehicle_manager.spawn_npcs(client=self._client, world=self._world, num_npcs=self._town.num_vehicles)
            self._walker_manager.spawn_npcs(client=self._client, world=self._world, num_npcs=self._town.num_walkers)

    def _init_agent_controller(self, waypoint_list: list):
        self._agent_controller = AgentController(
            vehicle=self._vehicle,
            max_brake=self._max_brake,
            max_throttle=self._max_throttle
        )
        self._agent_controller.set_route(waypoints_list=waypoint_list)

    def _destroy_actors(self):
        # Destroy sensors
        self._sensor_manager.destroy_sensors(world=self._world)

        # Destroy vehicle
        if self._vehicle is not None:
            self._agent_controller.destroy()
            self._vehicle.destroy()
            self._vehicle = None
            self._world.tick()

        # Destroy NPCs
        if self._iteration % self._iterations_per_reload == 0:
            self._vehicle_manager.destroy_npcs(client=self._client, world=self._world)
            self._walker_manager.destroy_npcs(client=self._client, world=self._world)

    def _get_state(self, pid_control: carla.VehicleControl, action: np.ndarray) -> dict:
        sensor_observations = self._sensor_manager.get_observations()

        self._collision = sensor_observations['collision']
        self._obstacle = sensor_observations['obstacle']

        speed = min(100.0, utils.get_vehicle_speed(vehicle=self._vehicle))/100.0
        return {
            'image': sensor_observations['image'],
            'control': np.float32([action, pid_control.throttle, pid_control.brake, speed])
        }

    def get_pid_control(self) -> carla.VehicleControl:
        return self._agent_controller.get_pid_control()

    def get_control(self) -> carla.VehicleControl:
        return self._vehicle.get_control()

    def set_control(self, control: carla.VehicleControl):
        self._vehicle.apply_control(control)

    def reset(self, **kwargs) -> dict:
        self._collision = None
        self._end_location = None
        self._episode_step = 0

        if self._iteration % self._iterations_per_reload == 0:
            if self._debug:
                print('Reloading world...')

            self._world = self._client.reload_world()

        self._synchronize()
        self._world.tick()

        self._destroy_actors()
        self._spawn_actors()

        # Initializing agent controller
        self._init_agent_controller(waypoint_list=self._current_route_waypoint_list)

        self._world.tick()
        return self._get_state(pid_control=self._vehicle.get_control(), action=np.float32([0.0]))

    def _done(self):
        reached_destination = self._vehicle.get_location().distance(self._end_location) < 5.0
        max_steps_reached = self._episode_step >= self._max_steps

        if self._debug:
            if reached_destination:
                print('Reached Destination')
            elif self._collision:
                print('Collided')
            elif max_steps_reached:
                print('Max Steps Reached')

        return self._collision is not None or reached_destination or max_steps_reached

    def _get_reward(self, pid_control: carla.VehicleControl) -> float:
        # Case 1. Vehicle has collided
        if self._collision:
            return -10

        obstacle_actor, distance = self._obstacle

        # Case 2. No Front Obstacle: Return acceleration (Must speed up)
        if obstacle_actor is None:
            return -1.0 if pid_control.throttle == 0.0 else pid_control.throttle/self._max_throttle
        else:
            actor_type = obstacle_actor.type_id[0: 4]

            # Case 3. Front walker: Stop until it passed
            if actor_type == 'walk':
                return -1.0 if pid_control.brake == 0.0 else pid_control.brake/self._max_brake

            # Case 4. Front vehicle
            elif actor_type == 'vehi':
                # Case 4.1 - Vehicle in Range (0, max safety dist)
                front_vehicle_speed = utils.get_vehicle_speed(vehicle=obstacle_actor)
                vehicle_speed = utils.get_vehicle_speed(vehicle=self._vehicle)

                if front_vehicle_speed <= 0.1:
                    if vehicle_speed <= 0.1:
                        if distance <= self._min_safety_distance:
                            return 1.0
                        else:
                            return 1/(distance - self._min_safety_distance + 1)
                    else:
                        return - pid_control.throttle/self._max_throttle
                else:
                    return 1/(abs(distance - self._min_safety_distance) + 1)
            else:
                raise NotImplementedError(f'Not implemented reward function for obstacle Actor {obstacle_actor}')

    def _apply_control(self, pid_control: carla.VehicleControl, action: np.ndarray) -> carla.VehicleControl:
        acceleration = action[0]

        if acceleration >= 0:
            pid_control.brake = 0.0
            pid_control.throttle = float(acceleration)
        elif 0 > acceleration > -0.1:
            pid_control.brake = 0.0
            pid_control.throttle = 0.0
        else:
            pid_control.brake = -float(acceleration)
            pid_control.throttle = 0.0

        if self._debug:
            print(f'Agent Control: {pid_control}')

        self._vehicle.apply_control(pid_control)

        return pid_control

    def step(self, action: np.ndarray) -> tuple:
        assert -1.0 <= action[0] <= 1.0

        self._episode_step += 1
        pid_control = self._agent_controller.get_pid_control()

        if self._debug:
            print(f'PID Control: {pid_control}')

        pid_control = self._apply_control(pid_control=pid_control, action=action)

        for _ in range(self._action_repeats):
            self._world.tick()

        reward = self._get_reward(pid_control=pid_control)
        next_state = self._get_state(pid_control=pid_control, action=action)
        done = self._done()

        if done:
            self._iteration += 1
            self._desynchronize()
            self._world.tick()

        return next_state, reward, done, {}

    def render(self):
        raise NotImplementedError('Do not call this method. Rendering is implemented through CARLA simulator')
