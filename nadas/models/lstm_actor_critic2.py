import numpy as np
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension


class LSTMActorCritic(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name
        )
        self.cell_size = 64 if 'cell_size' not in model_config else model_config['cell_size']

        orig_space = getattr(obs_space, 'original_space', obs_space)

        image_inputs = tf.keras.layers.Input(shape=orig_space['image'].shape, name='image')

        h_image = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', use_bias=False)(image_inputs)
        h_image = tf.keras.layers.BatchNormalization()(h_image)
        h_image = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, activation='relu', use_bias=False)(h_image)
        h_image = tf.keras.layers.BatchNormalization()(h_image)
        h_image = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', use_bias=False)(h_image)
        h_image = tf.keras.layers.BatchNormalization()(h_image)
        h_image = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', use_bias=False)(h_image)
        h_image = tf.keras.layers.BatchNormalization()(h_image)
        h_image = tf.keras.layers.Flatten()(h_image)

        action_inputs = tf.keras.layers.Input(shape=orig_space['control'].shape, name='control')
        out = tf.keras.layers.Concatenate(axis=-1)([h_image, action_inputs])

        self.base_model = tf.keras.Model(inputs=[image_inputs, action_inputs], outputs=out, name='base_model')

        rnn_input = tf.keras.layers.Input(shape=self.base_model.output_shape, name='inputs')
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name='h')
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name='c')
        seq_in = tf.keras.layers.Input(shape=(), name='seq_in', dtype=tf.int32)

        # Send to LSTM cell
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size,
            return_sequences=True,
            return_state=True,
            name='lstm'
        )(inputs=rnn_input, mask=tf.sequence_mask(seq_in), initial_state=[state_in_h, state_in_c])

        x = tf.keras.layers.Dense(units=128, activation='tanh')(lstm_out)
        policy_out = tf.keras.layers.Dense(units=self.num_outputs, activation=None, name='policy_out')(x)
        value_out = tf.keras.layers.Dense(units=1, activation=None, name='value_out')(x)
        self.rnn_model = tf.keras.Model(
            inputs=[rnn_input, seq_in, state_in_h, state_in_c],
            outputs=[policy_out, value_out, state_h, state_c],
            name='rnn_model'
        )

    def forward_rnn(self, inputs, state, seq_lens):
        x = self.base_model(inputs)
        x = add_time_dimension(padded_inputs=x, seq_lens=seq_lens, framework='tf')
        model_out, self._value_out, h, c = self.rnn_model([x, seq_lens] + state)
        return model_out, [h, c]

    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.forward_rnn(input_dict['obs'], state, seq_lens)
        return tf.reshape(model_out, [-1, self.num_outputs]), state

    def get_initial_state(self):
        return [
            np.random.uniform(low=-0.001, high=0.001, size=(self.cell_size,)),
            np.random.uniform(low=-0.001, high=0.001, size=(self.cell_size,)),
        ]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

