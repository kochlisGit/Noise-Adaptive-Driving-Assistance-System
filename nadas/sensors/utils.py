import random
import numpy as np
from sklearn.cluster import KMeans


def apply_segmentation_noise(image: np.ndarray, noise_ratio: float) -> np.ndarray:
    height, width = image.shape[: 2]
    num_pixels = height*width
    num_noisy_pixels = int(num_pixels*noise_ratio)

    random_noise_ids = np.random.choice(a=num_pixels, size=num_noisy_pixels, replace=False)
    rows, cols = np.unravel_index(indices=random_noise_ids, shape=(height, width))
    image[rows, cols] = 0
    return image


def apply_depth_noise(image: np.ndarray, error_percentage: float) -> np.ndarray:
    noise = np.random.uniform(low=-error_percentage, high=error_percentage)
    noisy_image = image + noise
    return noisy_image.clip(min=0.0, max=1.0)


def corrupt_image_area(image: np.ndarray, portion_size: tuple):
    height, width = image.shape[: 2]
    portion_height, portion_width = portion_size

    y1 = random.randint(0, height - portion_height - 1)
    x1 = random.randint(0, width - portion_width - 1)
    y2 = y1 + portion_height - 1
    x2 = y2 + portion_width - 1
    image[y1: y2, x1: x2, :] = 1.0
    return image


def get_normalized_clustered_radar_points(
        radar_measurements: np.ndarray,
        radar_range: float,
        h_fov: float,
        v_fov: float,
        n_clusters: int
) -> np.ndarray:
    # Apply scaling before K-Means
    max_values = np.float32([h_fov, v_fov, radar_range, 100])
    x_scaled = radar_measurements/max_values

    if n_clusters > x_scaled.shape[0]:
        # Fill missing values in case observations are missing
        missing_measurements = np.zeros(shape=(n_clusters - x_scaled.shape[0], 4), dtype=np.float32)
        return np.vstack((x_scaled, missing_measurements))
    else:
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, max_iter=50, n_init=10, random_state=0)
        kmeans.fit(x_scaled)
        centers = kmeans.cluster_centers_

        if centers.shape[0] < n_clusters:
            missing_measurements = np.zeros(shape=(n_clusters - centers.shape[0], 4), dtype=np.float32)
            centers = np.vstack((centers, missing_measurements))
        return centers


def estimate_corrupted_sensor_data(measurements, memory_size: int) -> np.ndarray:
    # Temporal Differencing to estimate next state
    weights = np.linspace(start=1.0, stop=1.0/memory_size, num=memory_size)[1:]
    temporal_differences = np.diff(measurements, axis=0)
    average_diff = np.average(temporal_differences, axis=0, weights=weights)
    return measurements[0] + average_diff
