import carla
import cv2
import numpy as np
from collections import deque
from nadas.sensors.sensor_manager import SensorManager
from nadas.sensors.sensor_types import SensorType
from nadas.sensors import utils


class ACCSensorManager(SensorManager):
    def __init__(
            self,
            blueprints_library,
            sensor_memory_size: int,
            sensor_data_corrupt_prob: float,
            segmentation_corrupt_portion_size: tuple or None,
            depth_error_rate: float,
            hazard_distance: float,
            use_state_prediction: bool,
            debug: bool = False,
            store_directory: str or None = None
    ):
        self._segmentation_corrupt_portion_size = segmentation_corrupt_portion_size
        self._depth_error_rate = depth_error_rate
        self._hazard_distance = hazard_distance

        self._image_height = 60
        self._image_width = 80
        self._image_fov = 45
        self._sensor_tick = 0.01

        self._tag_colors = {
            'VEHICLES': [142, 0, 0],
            'WALKERS': [60, 20, 220]
        }

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

        # Depth camera blueprints
        dep_cam_bp = blueprints_library.find('sensor.camera.depth')
        dep_cam_bp.set_attribute('image_size_x', f'{self._image_width}')
        dep_cam_bp.set_attribute('image_size_y', f'{self._image_height}')
        dep_cam_bp.set_attribute('fov', f'{self._image_fov}')
        dep_cam_bp.set_attribute('sensor_tick', f'{self._sensor_tick}')
        sensor_blueprints[SensorType.DEPTH] = dep_cam_bp

        # Obstacle Detector blueprint
        obstacle_bp = blueprints_library.find('sensor.other.obstacle')
        obstacle_bp.set_attribute('distance', f'{self._hazard_distance}')
        obstacle_bp.set_attribute('only_dynamics', f'{True}')
        sensor_blueprints[SensorType.OBSTACLE_DETECTOR] = obstacle_bp

        # Collision Detector blueprint
        collision_bp = blueprints_library.find('sensor.other.collision')
        sensor_blueprints[SensorType.COLLISION_DETECTOR] = collision_bp

        return sensor_blueprints

    def _get_sensor_data_placeholder(self) -> dict:
        return {
            SensorType.SEGMENTATION: None,
            SensorType.DEPTH: None,
            SensorType.COLLISION_DETECTOR: None,
            SensorType.OBSTACLE_DETECTOR: (None, None)
        }

    def _get_memory_buffer(self) -> dict:
        actor_buffer = deque(maxlen=self._sensor_memory_size)
        depth_buffer = deque(maxlen=self._sensor_memory_size)
        for _ in range(self._sensor_memory_size):
            actor_buffer.append(np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32))
            depth_buffer.append(np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32))
        return {'actor': actor_buffer, 'depth': depth_buffer}

    def _spawn_sensors(self, vehicle: carla.Vehicle, sp_transform: carla.Transform, world: carla.World) -> list:
        sensor_list = []

        sensor_bp = self._blueprints_dict[SensorType.SEGMENTATION]
        sensor = world.spawn_actor(sensor_bp, sp_transform, attach_to=vehicle)
        sensor.listen(lambda data: self._sensor_listener(sensor_type=SensorType.SEGMENTATION, data=data))
        sensor_list.append(sensor)

        sensor_bp = self._blueprints_dict[SensorType.DEPTH]
        sensor = world.spawn_actor(sensor_bp, sp_transform, attach_to=vehicle)
        sensor.listen(lambda data: self._sensor_listener(sensor_type=SensorType.DEPTH, data=data))
        sensor_list.append(sensor)

        sensor_bp = self._blueprints_dict[SensorType.COLLISION_DETECTOR]
        sensor = world.spawn_actor(sensor_bp, sp_transform, attach_to=vehicle)
        sensor.listen(lambda data: self._sensor_listener(sensor_type=SensorType.COLLISION_DETECTOR, data=data))
        sensor_list.append(sensor)

        sensor_bp = self._blueprints_dict[SensorType.OBSTACLE_DETECTOR]
        sensor = world.spawn_actor(sensor_bp, sp_transform, attach_to=vehicle)
        sensor.listen(lambda data: self._sensor_listener(sensor_type=SensorType.OBSTACLE_DETECTOR, data=data))
        sensor_list.append(sensor)

        world.tick()
        return sensor_list

    def _get_actor_mask_and_ids(self, data, memory_buffer: deque, timestamp: int) -> tuple:
        if data is None:
            if self.use_state_prediction:
                actor_mask = utils.estimate_corrupted_sensor_data(
                    measurements=memory_buffer,
                    memory_size=self._sensor_memory_size
                )
                actor_mask_ids = actor_mask > 0
            else:
                actor_mask = np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32)
                actor_mask_ids = None
        else:
            data.convert(carla.ColorConverter.CityScapesPalette)
            image = np.reshape(data.raw_data, newshape=(self._image_height, self._image_width, 4))[:, :, :3]

            # Apply noise to an area of segmentation image
            if self._segmentation_corrupt_portion_size is not None:
                image = utils.corrupt_image_area(
                    image=image.copy(),
                    portion_size=self._segmentation_corrupt_portion_size
                )

            if self.debug:
                cv2.imshow('Segmentation', cv2.resize(image, dsize=(self._image_width, self._image_height)))
            if self.store_directory is not None:
                cv2.imwrite(f'{self._store_directory}/segmentation_{timestamp}.png', image)

            actor_mask_ids = (
                    (image == self._tag_colors['VEHICLES']) |
                    (image == self._tag_colors['WALKERS'])
            ).all(axis=2)
            actor_mask = np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32)
            actor_mask[actor_mask_ids] = 1.0
        return actor_mask, actor_mask_ids

    def _get_depth_values(self, data, memory_buffer: deque, timestamp: int) -> np.ndarray:
        if data is None:
            if self.use_state_prediction:
                return utils.estimate_corrupted_sensor_data(
                    measurements=memory_buffer,
                    memory_size=self._sensor_memory_size
                )
            else:
                return np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32)
        else:
            data.convert(carla.ColorConverter.LogarithmicDepth)
            depth_pixels = np.reshape(
                data.raw_data,
                newshape=(self._image_height, self._image_width, 4)
            )[:, :, :1]/255.0

            # Apply noise to depth pixels
            if self._depth_error_rate > 0.0:
                depth_pixels = utils.apply_depth_noise(
                    image=depth_pixels,
                    error_percentage=self._depth_error_rate
                )
            if self._debug:
                cv2.imshow('Depth', cv2.resize(depth_pixels, dsize=(self._image_width, self._image_height)))
            if self._store_directory is not None:
                cv2.imwrite(f'{self._store_directory}/depth_{timestamp}.png', depth_pixels)

            # Invert colors, so that objects near to 1.0 are closer and 0.0 is noise or open space
            return 1 - depth_pixels

    def _get_observations(self, sensor_data_placeholder: dict, memory_buffer: dict, timestamp: int) -> tuple:
        actor_buffer = memory_buffer['actor']
        actor_mask, actor_mask_ids = self._get_actor_mask_and_ids(
            data=sensor_data_placeholder[SensorType.SEGMENTATION],
            memory_buffer=actor_buffer,
            timestamp=timestamp
        )
        actor_buffer.appendleft(actor_mask)
        memory_buffer['actor'] = actor_buffer

        depth_buffer = memory_buffer['depth']
        depth_values = self._get_depth_values(
            data=sensor_data_placeholder[SensorType.DEPTH],
            memory_buffer=depth_buffer,
            timestamp=timestamp
        )
        depth_buffer.appendleft(depth_values)
        memory_buffer['depth'] = depth_buffer

        observation = np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32)
        if actor_mask_ids is not None:
            observation[actor_mask_ids] = depth_values[actor_mask_ids]

        if self.debug:
            cv2.imshow('Observation', cv2.resize(observation, dsize=(self._image_width, self._image_height)))

            collision_info = self._sensor_data_placeholder[SensorType.COLLISION_DETECTOR]
            if collision_info:
                print(f'Collision: {collision_info}')

            obstacle_info = self._sensor_data_placeholder[SensorType.OBSTACLE_DETECTOR]
            if obstacle_info[0] is not None:
                print(f'Obstacle {obstacle_info[0]} detected at distance: {obstacle_info[1]}')

        if self.store_directory is not None:
            cv2.imwrite(f'{self._store_directory}/observation_{timestamp}.png', observation)

        observations = {
            'collision': self._sensor_data_placeholder[SensorType.COLLISION_DETECTOR],
            'obstacle': self._sensor_data_placeholder[SensorType.OBSTACLE_DETECTOR],
            'image': observation
        }
        return observations, memory_buffer
