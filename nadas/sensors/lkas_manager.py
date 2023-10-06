import carla
import cv2
import numpy as np
from collections import deque
from nadas.sensors.sensor_manager import SensorManager
from nadas.sensors.sensor_types import SensorType
from nadas.sensors import utils


class LKASSensorManager(SensorManager):
    def __init__(
            self,
            blueprints_library,
            sensor_memory_size: int,
            sensor_data_corrupt_prob: float,
            segmentation_noise_ratio: float,
            use_state_prediction: bool,
            debug: bool = False,
            store_directory: str or None = None
    ):
        self._segmentation_noise_ratio = segmentation_noise_ratio

        self._image_height = 60
        self._image_width = 80
        self._image_fov = 45
        self._sensor_tick = 0.01

        self._tag_colors = {'ROAD_LANES': [50, 234, 157]}

        super().__init__(
            blueprints_library=blueprints_library,
            sensor_memory_size=sensor_memory_size,
            sensor_data_corrupt_prob=sensor_data_corrupt_prob,
            use_state_prediction=use_state_prediction,
            debug=debug,
            store_directory=store_directory
        )

    def _get_sensor_blueprints_dict(self, blueprints_library: carla.BlueprintLibrary) -> dict:
        sensor_blueprints = {}

        # Segmentation camera blueprints
        seg_cam_bp = blueprints_library.find('sensor.camera.semantic_segmentation')
        seg_cam_bp.set_attribute('image_size_x', f'{self._image_width}')
        seg_cam_bp.set_attribute('image_size_y', f'{self._image_height}')
        seg_cam_bp.set_attribute('fov', f'{self._image_fov}')
        seg_cam_bp.set_attribute('sensor_tick', f'{self._sensor_tick}')
        sensor_blueprints[SensorType.SEGMENTATION] = seg_cam_bp

        # Collision Detector blueprint
        collision_bp = blueprints_library.find('sensor.other.collision')
        sensor_blueprints[SensorType.COLLISION_DETECTOR] = collision_bp

        return sensor_blueprints

    def _get_sensor_data_placeholder(self) -> dict:
        return {
            SensorType.SEGMENTATION: None,
            SensorType.COLLISION_DETECTOR: None
        }

    def _get_memory_buffer(self) -> dict:
        lane_memory_buffer = deque(maxlen=self._sensor_memory_size)
        for _ in range(self._sensor_memory_size):
            lane_memory_buffer.append(np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32))
        return {'lane': lane_memory_buffer}

    def _spawn_sensors(self, vehicle: carla.Vehicle, sp_transform: carla.Transform, world: carla.World) -> list:
        sensor_list = []

        sensor_bp = self._blueprints_dict[SensorType.SEGMENTATION]
        sensor = world.spawn_actor(sensor_bp, sp_transform, attach_to=vehicle)
        sensor.listen(lambda data: self._sensor_listener(sensor_type=SensorType.SEGMENTATION, data=data))
        sensor_list.append(sensor)

        sensor_bp = self._blueprints_dict[SensorType.COLLISION_DETECTOR]
        sensor = world.spawn_actor(sensor_bp, sp_transform, attach_to=vehicle)
        sensor.listen(lambda data: self._sensor_listener(sensor_type=SensorType.COLLISION_DETECTOR, data=data))
        sensor_list.append(sensor)

        world.tick()
        return sensor_list

    def _get_lane_mask(self, data, memory_buffer: deque, timestamp: int) -> np.ndarray:
        if data is None:
            if self.use_state_prediction:
                return utils.estimate_corrupted_sensor_data(
                    measurements=memory_buffer,
                    memory_size=self._sensor_memory_size
                )
            else:
                return np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32)
        else:
            # Process segmentation colors and get lane mask
            data.convert(carla.ColorConverter.CityScapesPalette)
            image = np.reshape(data.raw_data, newshape=(self._image_height, self._image_width, 4))[:, :, :3]

            if self.debug:
                cv2.imshow('Segmentation', cv2.resize(image, dsize=(self._image_width, self._image_height)))
            if self.store_directory is not None:
                cv2.imwrite(f'{self._store_directory}/segmentation_{timestamp}.png', image)

            lane_ids = (image == self._tag_colors['ROAD_LANES']).all(axis=2)
            lane_mask = np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32)
            lane_mask[lane_ids] = 1.0

            # Apply noise to lane mask
            return utils.apply_segmentation_noise(
                image=lane_mask,
                noise_ratio=self._segmentation_noise_ratio
            )

    def _get_observations(self, sensor_data_placeholder: dict, memory_buffer: dict, timestamp: int) -> tuple:
        data = sensor_data_placeholder[SensorType.SEGMENTATION]
        lane_buffer = memory_buffer['lane']
        lane_mask = self._get_lane_mask(data=data, memory_buffer=lane_buffer, timestamp=timestamp)
        lane_buffer.appendleft(lane_mask)
        memory_buffer['lane'] = lane_buffer

        if self.debug:
            cv2.imshow('Observation', cv2.resize(lane_mask, dsize=(self._image_width, self._image_height)))

            collision = self._sensor_data_placeholder[SensorType.COLLISION_DETECTOR]
            if collision:
                print(f'Collision: {collision}')
        if self.store_directory is not None:
            cv2.imwrite(f'{self._store_directory}/observation_{timestamp}.png', lane_mask)

        observations = {
            'collision': self._sensor_data_placeholder[SensorType.COLLISION_DETECTOR],
            'image': lane_mask
        }
        return observations, memory_buffer
