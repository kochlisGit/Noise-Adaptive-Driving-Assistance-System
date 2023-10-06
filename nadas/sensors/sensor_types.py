from enum import Enum


class SensorType(Enum):
    SEGMENTATION = 0
    DEPTH = 1
    RADAR = 2
    COLLISION_DETECTOR = 3
    OBSTACLE_DETECTOR = 4
