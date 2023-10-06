import random
import cv2
import numpy as np
from nadas.environments.lkas_env import LKASEnvironment
from nadas.maps.towns.town10 import Town10


def main(
        server_ip: str,
        port: int,
        num_iterations: int,
        sensor_data_corrupt_prob: float,
        segmentation_noise_ratio: float,
        segmentation_corrupt_portion_size: tuple,
        depth_error_rate: float,
        use_state_prediction: bool,
        max_steps: int,
        iterations_per_reload: int,
        action_repeats: int,
        debug: bool,
        store_sensor_directory: str or None
):
    env = LKASEnvironment(env_config={
        'server_ip': server_ip,
        'port': port,
        'sensor_data_corrupt_prob': sensor_data_corrupt_prob,
        'segmentation_noise_ratio': segmentation_noise_ratio,
        'segmentation_corrupt_portion_size': segmentation_corrupt_portion_size,
        'depth_error_rate': depth_error_rate,
        'use_state_prediction': use_state_prediction,
        'town': Town10(),
        'max_steps': max_steps,
        'iterations_per_reload': iterations_per_reload,
        'action_repeats': action_repeats,
        'debug': debug,
        'store_sensor_directory': store_sensor_directory
    })

    for _ in range(num_iterations):
        done = False
        env.reset()

        while not done:
            action = random.choices(
                [np.float32([0.0]), np.float32([0.2]), np.float32([-0.2])],
                weights=[0.35, 0.3, 0.3],
                k=1
            )[0]
            print(f'Action = {action}')

            _, reward, done, _ = env.step(action=action)
            print(f'Reward = {reward}')

    if debug:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(
        server_ip='localhost',
        port=2000,
        num_iterations=5,
        sensor_data_corrupt_prob=0.2,
        segmentation_noise_ratio=0.2,
        segmentation_corrupt_portion_size=(60, 80),
        depth_error_rate=0.1,
        max_steps=1024,
        use_state_prediction=True,
        iterations_per_reload=2,
        action_repeats=2,
        debug=True,
        store_sensor_directory=None
    )
