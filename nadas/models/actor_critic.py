import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


class ActorCritic(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name
        )
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
        x = tf.keras.layers.Dense(units=128, activation='tanh')(out)

        policy_out = tf.keras.layers.Dense(units=num_outputs, activation=None, name='policy_out')(x)
        value_out = tf.keras.layers.Dense(units=1, activation=None, name='value_out')(x)

        self.base_model = tf.keras.Model(
            inputs=[image_inputs, action_inputs],
            outputs=[policy_out, value_out],
            name='base_model'
        )

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict['obs'])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
