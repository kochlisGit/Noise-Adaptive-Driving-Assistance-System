import gymnasium
import numpy as np
from collections import deque
from nadas.environments.acc_env import ACCEnvironment
from nadas.environments.lkas_env import LKASEnvironment


class FrameStackingEnv(gymnasium.Env):
    def __init__(self, env_config: dict):
        assert 'env' in env_config

        self._env = env_config['env'](env_config=env_config)
        self._num_frames = env_config['num_frames']

        assert self._num_frames > 1

        if isinstance(self._env, ACCEnvironment):
            self.observation_space = gymnasium.spaces.Dict({
                'image': gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(60, 80, self._num_frames), dtype=np.float32),
                'control': gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(4*self._num_frames,), dtype=np.float32)
            })
        elif isinstance(self._env, LKASEnvironment):
            self.observation_space = gymnasium.spaces.Dict({
                'image': gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(60, 80, self._num_frames), dtype=np.float32),
                'control': gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2*self._num_frames,), dtype=np.float32)
            })
        else:
            raise NotImplementedError(f'Not implemented environment {type(self._env)}')

        self.action_space = self._env.action_space

        self._image_stack = deque(maxlen=self._num_frames)
        self._control_stack = deque(maxlen=self._num_frames)

        for i in range(self._num_frames):
            self._image_stack.append(np.zeros(shape=self._env.observation_space['image'].shape, dtype=np.float32))
            self._control_stack.append(np.zeros(shape=self._env.observation_space['control'].shape, dtype=np.float32))

    def _add_state_stack(self, state: dict) -> dict:
        image = state['image']
        self._image_stack.append(image)

        control = state['control']
        self._control_stack.append(control)

        return {
            'image': np.dstack(self._image_stack),
            'control': np.hstack(self._control_stack)
        }

    def reset(self, **kwargs) -> dict:
        state = self._env.reset(**kwargs)
        return self._add_state_stack(state=state)

    def step(self, action: np.ndarray) -> tuple:
        state, reward, done, info = self._env.step(action=action)
        return self._add_state_stack(state=state), reward, done, info

    def render(self):
        self._env.render()
