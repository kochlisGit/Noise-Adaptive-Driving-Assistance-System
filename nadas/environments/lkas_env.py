import carla
import gymnasium
import numpy as np
import random
from nadas.agents.agent import AgentController
from nadas.environments import utils
from nadas.maps.towns.town10 import Town10
from nadas.sensors.lkas_manager import LKASSensorManager
from nadas.npcs.vehicle_manager import VehicleNPCManager
from nadas.npcs.walker_manager import WalkerNPCManager


class LKASEnvironment(gymnasium.Env):
    def __init__(self, env_config: dict):
        self._server_ip = env_config['server_ip']
        self._port = env_config['port']
        self._sensor_data_corrupt_prob = env_config['sensor_data_corrupt_prob']
        self._segmentation_noise_ratio = env_config['segmentation_noise_ratio']
        self._use_state_prediction = env_config['use_state_prediction']
        self._town = Town10()
        self._max_steps = env_config['max_steps']
        self._iterations_per_reload = env_config['iterations_per_reload']
        self._action_repeats = env_config['action_repeats']
        self._debug = False if 'debug' not in env_config else env_config['debug']
        self._store_sensor_directory = None if 'store_sensor_directory' not in env_config else \
            env_config['store_sensor_directory']

        self._max_steering = 0.5
        self._sensor_memory_size = 5
        self._fixed_delta_time = 0.01
        self._sensor_sp_transform = carla.Transform(carla.Location(x=0.8, z=1.5), carla.Rotation(pitch=-10))

        # Define Action Space
        self.action_space = gymnasium.spaces.Box(
            low=-self._max_steering,
            high=self._max_steering,
            shape=(1,),
            dtype=np.float32
        )

        # Camera Image Size as observation (Simple)
        self.observation_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(60, 80, 1), dtype=np.float32),
            'control': gymnasium.spaces.Box(low=-self._max_steering, high=self._max_steering, shape=(2,), dtype=np.float32)
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
        self._sensor_manager = LKASSensorManager(
            blueprints_library=blueprints_library,
            sensor_memory_size=self._sensor_memory_size,
            sensor_data_corrupt_prob=self._sensor_data_corrupt_prob,
            segmentation_noise_ratio=self._segmentation_noise_ratio,
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
        self._out_of_lane = False
        self._lane_id = None
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
            max_steering=self._max_steering
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

    def _get_state(self, steer: float) -> dict:
        sensor_observations = self._sensor_manager.get_observations()

        vehicle_speed = utils.get_vehicle_speed(vehicle=self._vehicle)
        self._collision = sensor_observations['collision']
        return {
            'image': sensor_observations['image'],
            'control': np.float32([steer, vehicle_speed/100.0])
        }

    def get_pid_control(self) -> carla.VehicleControl:
        return self._agent_controller.get_pid_control()

    def get_control(self) -> carla.VehicleControl:
        return self._vehicle.get_control()

    def set_control(self, control: carla.VehicleControl):
        self._vehicle.apply_control(control)

    def reset(self, **kwargs) -> dict:
        self._collision = None
        self._out_of_lane = False
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
        self._lane_id = self._world_map.get_waypoint(self._vehicle.get_location(), project_to_road=True).lane_id

        # Initializing agent controller
        self._init_agent_controller(waypoint_list=self._current_route_waypoint_list)

        self._world.tick()
        return self._get_state(steer=0.0)

    def _done(self):
        reached_destination = self._vehicle.get_location().distance(self._end_location) < 5.0
        max_steps_reached = self._episode_step >= self._max_steps

        if self._debug:
            if self._out_of_lane:
                print('Out of Lane')
            elif reached_destination:
                print('Reached Destination')
            elif self._collision:
                print('Collided')
            elif max_steps_reached:
                print('Max Steps Reached')

        return self._collision is not None or reached_destination or self._out_of_lane or max_steps_reached

    def _apply_control(self, pid_control: carla.VehicleControl, steer: float) -> carla.VehicleControl:
        pid_control.steer = steer

        if self._debug:
            print(f'Agent Control: {pid_control}')

        self._vehicle.apply_control(pid_control)
        return pid_control

    def _get_reward(self, pid_control: carla.VehicleControl, planner_steer: float) -> float:
        wheels = self._vehicle.get_physics_control().wheels

        self._out_of_lane = utils.out_of_lane(
            wheels=wheels,
            world_map=self._world_map,
            lane_id=self._lane_id
        )
        if self._out_of_lane or self._collision:
            return -10

        steer_difference = abs(pid_control.steer - planner_steer)
        return -1.0 if pid_control.steer*planner_steer < 0 else 1 - steer_difference

    def step(self, action: np.ndarray) -> tuple:
        steer = float(action[0])

        assert -1.0 <= steer <= 1.0

        self._episode_step += 1
        pid_control = self._agent_controller.get_pid_control()
        planner_steer = round(pid_control.steer, 1)

        if self._debug:
            print(f'PID Control: {pid_control}')

        pid_control = self._apply_control(pid_control=pid_control, steer=steer)

        for _ in range(self._action_repeats):
            self._world.tick()

        reward = self._get_reward(pid_control=pid_control, planner_steer=planner_steer)
        next_state = self._get_state(steer=steer)
        done = self._done()

        if done:
            self._iteration += 1
            self._desynchronize()

        return next_state, reward, done, {}

    def render(self):
        raise NotImplementedError('Do not call this method. Rendering is implemented through CARLA simulator')
