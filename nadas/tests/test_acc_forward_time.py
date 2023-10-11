import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert
import tensorflow_probability as tfp
import numpy as np
import time
from collections import deque

tfd = tfp.distributions


def add_time_dimension(
        padded_inputs,
        *,
        seq_lens,
        framework: str = "tf",
        time_major: bool = False,
):
    """Adds a time dimension to padded inputs.

    Args:
        padded_inputs: a padded batch of sequences. That is,
            for seq_lens=[1, 2, 2], then inputs=[A, *, B, B, C, C], where
            A, B, C are sequence elements and * denotes padding.
        seq_lens: A 1D tensor of sequence lengths, denoting the non-padded length
            in timesteps of each rollout in the batch.
        framework: The framework string ("tf2", "tf", "torch").
        time_major: Whether data should be returned in time-major (TxB)
            format or not (BxT).

    Returns:
        TensorType: Reshaped tensor of shape [B, T, ...] or [T, B, ...].
    """

    # Sequence lengths have to be specified for LSTM batch inputs. The
    # input batch must be padded to the max seq length given here. That is,
    # batch_size == len(seq_lens) * max(seq_lens)
    if framework in ["tf2", "tf"]:
        assert time_major is False, "time-major not supported yet for tf!"
        padded_batch_size = tf.shape(padded_inputs)[0]
        # Dynamically reshape the padded batch to introduce a time dimension.
        new_batch_size = tf.shape(seq_lens)[0]
        time_size = padded_batch_size // new_batch_size
        new_shape = tf.concat(
            [
                tf.expand_dims(new_batch_size, axis=0),
                tf.expand_dims(time_size, axis=0),
                tf.shape(padded_inputs)[1:],
            ],
            axis=0,
        )
        return tf.reshape(padded_inputs, new_shape)
    else:
        assert framework == "torch", "`framework` must be either tf or torch!"
        padded_batch_size = padded_inputs.shape[0]

        # Dynamically reshape the padded batch to introduce a time dimension.
        new_batch_size = seq_lens.shape[0]
        time_size = padded_batch_size // new_batch_size
        batch_major_shape = (new_batch_size, time_size) + padded_inputs.shape[1:]
        padded_outputs = padded_inputs.view(batch_major_shape)

        if time_major:
            # Swap the batch and time dimensions
            padded_outputs = padded_outputs.transpose(0, 1)
        return padded_outputs


class ACC:
    def __init__(self):
        self._memory_buffer_size = 10
        self._memory_buffer = deque(maxlen=self._memory_buffer_size)
        self._tag_colors = {
            'VEHICLES': [142, 0, 0],
            'WALKERS': [60, 20, 220]
        }
        self._image_height = 60
        self._image_width = 80
        for _ in range(self._memory_buffer_size):
            self._memory_buffer.append(
                np.zeros(shape=(self._image_height, self._image_width, 1))
            )

    def estimate_corrupted_sensor_data(self) -> np.ndarray:
        # Temporal Differencing to estimate next state
        weights = np.linspace(start=1.0, stop=1.0 / self._memory_buffer_size, num=self._memory_buffer_size)[1:]
        temporal_differences = np.diff(self._memory_buffer, axis=0)
        average_diff = np.average(temporal_differences, axis=0, weights=weights)
        return self._memory_buffer[0] + average_diff

    def _get_actor_mask_and_ids(self, data) -> tuple:
        if data is None:
            actor_mask = self.estimate_corrupted_sensor_data()
            actor_mask_ids = actor_mask > 0
        else:
            image = np.reshape(data, newshape=(self._image_height, self._image_width, 4))[:, :, :3]

            actor_mask_ids = (
                    (image == self._tag_colors['VEHICLES']) |
                    (image == self._tag_colors['WALKERS'])
            ).all(axis=2)

            actor_mask = np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32)
            actor_mask[actor_mask_ids] = 1.0
        return actor_mask, actor_mask_ids

    def _get_depth_values(self, data) -> np.ndarray:
        if data is None:
            return self.estimate_corrupted_sensor_data()
        else:
            depth_pixels = np.reshape(
                data,
                newshape=(self._image_height, self._image_width, 4)
            )[:, :, :1] / 255.0

            # Invert colors, so that objects near to 1.0 are closer and 0.0 is noise or open space
            return 1 - depth_pixels

    def get_observation(self, seg_data, depth_data) -> np.ndarray:
        actor_mask, actor_mask_ids = self._get_actor_mask_and_ids(data=seg_data)
        self._memory_buffer.appendleft(actor_mask)

        depth_values = self._get_depth_values(data=depth_data)
        self._memory_buffer.appendleft(depth_values)

        observation = np.zeros(shape=(self._image_height, self._image_width, 1), dtype=np.float32)
        if actor_mask_ids is not None:
            observation[actor_mask_ids] = depth_values[actor_mask_ids]

        return observation


image_inputs = tf.keras.layers.Input(shape=(60, 80, 1), name='image')

h_image = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', use_bias=False)(image_inputs)
h_image = tf.keras.layers.BatchNormalization()(h_image)
h_image = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, activation='relu', use_bias=False)(h_image)
h_image = tf.keras.layers.BatchNormalization()(h_image)
h_image = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', use_bias=False)(h_image)
h_image = tf.keras.layers.BatchNormalization()(h_image)
h_image = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', use_bias=False)(h_image)
h_image = tf.keras.layers.BatchNormalization()(h_image)
h_image = tf.keras.layers.Flatten()(h_image)

action_inputs = tf.keras.layers.Input(shape=2, name='control')
out = tf.keras.layers.Concatenate(axis=-1)([h_image, action_inputs])
out = tf.keras.layers.Dense(units=128, activation='tanh')(out)

base_model = tf.keras.Model(inputs=[image_inputs, action_inputs], outputs=out, name='base_model')

rnn_input = tf.keras.layers.Input(shape=(None, 128), name='inputs')
state_in_h = tf.keras.layers.Input(shape=(64,), name='h')
state_in_c = tf.keras.layers.Input(shape=(64,), name='c')
seq_in = tf.keras.layers.Input(shape=(), name='seq_in', dtype=tf.int32)

# Send to LSTM cell
lstm_out, state_h, state_c = tf.keras.layers.LSTM(
    64,
    return_sequences=True,
    return_state=True,
    name='lstm'
)(inputs=rnn_input, mask=tf.sequence_mask(seq_in), initial_state=[state_in_h, state_in_c])

x = tf.keras.layers.Dense(units=128, activation='tanh')(lstm_out)
policy_out = tf.keras.layers.Dense(units=1, activation=None, name='policy_out')(x)
value_out = tf.keras.layers.Dense(units=1, activation=None, name='value_out')(x)
rnn_model = tf.keras.Model(
    inputs=[rnn_input, seq_in, state_in_h, state_in_c],
    outputs=[policy_out, value_out, state_h, state_c],
    name='rnn_model'
)

base_model.build(input_shape=[(None, 60, 80, 1), (None, 2)])
rnn_model.build(input_shape=(None, 128))



forward_times = []

s = tfd.Sample(tfd.Normal(loc=0, scale=1), sample_shape=5)

acc = ACC()

for i in range(200):
    if np.random.random() < 0.3:
        seg_data = None
    else:
        # Generate random data
        seg_data = np.random.randint(0, 256, size=(60 * 80 * 4))
        num_pixels = 50  # Number of pixels to set to [50, 234, 157]
        pixel_indices = np.random.choice(60 * 80, size=num_pixels, replace=False).astype(int)
        for pixel_i in pixel_indices:
            seg_data[pixel_i * 4: (pixel_i + 1) * 4] = [142, 0, 0, 255]
        pixel_indices = np.random.choice(60 * 80, size=num_pixels, replace=False).astype(int)
        for pixel_i in pixel_indices:
            seg_data[pixel_i * 4: (pixel_i + 1) * 4] = [60, 20, 220, 255]

    if np.random.random() < 0.3:
        depth_data = None
    else:
        # Generate random data
        depth_data = np.random.uniform(0, 1, size=(60 * 80 * 4))

    image_input = np.random.uniform(low=0, high=1, size=(1, 60, 80, 1))
    control_input = np.random.uniform(low=0, high=1, size=(1, 2))
    lstm_input = np.random.uniform(low=0, high=1, size=(1, 1, 128))
    seq_input = np.ones(shape=1) * 10
    state = [
        np.random.uniform(low=0, high=1, size=(1, 64)),
        np.random.uniform(low=0, high=1, size=(1, 64))
    ]

    start = time.time()

    lane_mask = acc.get_observation(seg_data, depth_data)
    s.sample()
    out = base_model.predict([image_input, control_input])
    out = add_time_dimension(padded_inputs=out, seq_lens=seq_in)
    rnn_model.predict([lstm_input, seq_input] + state)

    end = time.time()

    forward_times.append(end - start)

    print(i, forward_times[-1])

print()
print(np.average(forward_times))
