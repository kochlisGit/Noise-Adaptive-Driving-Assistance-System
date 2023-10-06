from abc import ABC, abstractmethod
import carla
import cv2
import random
import time
from nadas.sensors.sensor_types import SensorType


class SensorManager(ABC):
    def __init__(
            self,
            blueprints_library: carla.BlueprintLibrary,
            sensor_memory_size: int,
            sensor_data_corrupt_prob: float,
            use_state_prediction: bool,
            debug: bool = False,
            store_directory: str or None = None
    ):
        self._sensor_memory_size = sensor_memory_size
        self._sensor_data_corrupt_prob = sensor_data_corrupt_prob
        self._use_state_prediction = use_state_prediction
        self._debug = debug
        self._store_directory = store_directory

        self._blueprints_dict = self._get_sensor_blueprints_dict(blueprints_library=blueprints_library)

        self._sensor_list = None
        self._sensor_data_placeholder = None
        self._memory_buffer = None

    @property
    def use_state_prediction(self) -> bool:
        return self._use_state_prediction

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def store_directory(self) -> str or None:
        return self._store_directory

    @abstractmethod
    def _get_sensor_blueprints_dict(self, blueprints_library: carla.BlueprintLibrary) -> dict:
        pass

    @abstractmethod
    def _get_sensor_data_placeholder(self) -> dict:
        pass

    @abstractmethod
    def _get_memory_buffer(self) -> dict:
        pass

    @abstractmethod
    def _spawn_sensors(self, vehicle: carla.Vehicle, sp_transform: carla.Transform, world: carla.World) -> list:
        pass

    @abstractmethod
    def _get_observations(self, sensor_data_placeholder: dict, memory_buffer: dict, timestamp: int) -> dict:
        pass

    def _initialize(self):
        self._sensor_data_placeholder = self._get_sensor_data_placeholder()
        self._memory_buffer = self._get_memory_buffer()

    def _sensor_listener(self, sensor_type: SensorType, data):
        if sensor_type == SensorType.COLLISION_DETECTOR:
            sensor_data = data.other_actor
        elif sensor_type == SensorType.OBSTACLE_DETECTOR:
            sensor_data = (data.other_actor, data.distance)
        else:
            # Corrupt sensor data with some probability
            sensor_data = None if random.uniform(0.0, 1.0) <= self._sensor_data_corrupt_prob else data
        self._sensor_data_placeholder[sensor_type] = sensor_data

    def get_observations(self) -> dict:
        timestamp = int(time.time())
        observations, self._memory_buffer = self._get_observations(
            sensor_data_placeholder=self._sensor_data_placeholder,
            memory_buffer=self._memory_buffer,
            timestamp=timestamp
        )
        self._sensor_data_placeholder = self._get_sensor_data_placeholder()

        if self.debug:
            cv2.waitKey(1)

        return observations

    def spawn_sensors(self, vehicle: carla.Vehicle, sp_transform: carla.Transform, world: carla.World):
        if self._sensor_list is not None:
            raise RuntimeError('Sensors should be destroyed before spawning new sensors')

        self._initialize()
        self._sensor_list = self._spawn_sensors(vehicle=vehicle, sp_transform=sp_transform, world=world)

    def destroy_sensors(self, world: carla.World):
        if self._sensor_list is None:
            return

        for sensor in self._sensor_list:
            sensor.destroy()
        self._sensor_list = None
        world.tick()
