import collections
from itertools import zip_longest
from typing import Tuple

import numpy as np

try:
    import gym
    import cv2
    from gym.wrappers import FrameStack, LazyFrames
except ImportError as err:
    raise err


class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self, **kwargs):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info
    
    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class CropObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, slices: Tuple[slice, ...]):
        super(CropObservation, self).__init__(env)
        self.cropped_slice = slices
        new_shape = list()
        for s, old_s in zip_longest(slices, env.observation_space.shape):
            if s:
                s_start = s.start if s.start else 0
                s_stop = s.stop if s.stop else old_s
                new_shape.append(s_stop - s_start)
            else:
                new_shape.append(old_s)
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=tuple(new_shape),
                                                dtype=env.observation_space.dtype)
    
    def observation(self, observation):
        return self.process(observation)
    
    def process(self, frame) -> np.ndarray:
        result = frame[self.cropped_slice]
        return result.astype(np.uint8)


class TransposeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, axes: Tuple[int, ...] = ()):
        super(TransposeObservation, self).__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=old_shape[::-1],
                                                dtype=env.observation_space.dtype)
        self.axes = axes
    
    def observation(self, observation: np.ndarray):
        return observation.transpose(self.axes) if self.axes else observation.transpose()


class StackFrame(gym.ObservationWrapper):
    def __init__(self, env, stacks: int, shape: Tuple, stack_axis: int):
        super().__init__(env)
        self.stacks = stacks
        self.stack_axis = stack_axis
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=shape, dtype=env.observation_space.dtype)
        self.buffer = np.zeros(shape, dtype=env.observation_space.dtype)
    
    def reset(self, **kwargs):
        super(StackFrame, self).reset(**kwargs)
        self.buffer = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return self.observation(self.env.reset())
    
    def observation(self, observation):
        slices = [slice(None, None) for _ in self.observation_space.shape]
        move_old_slice = move_new_slice = slices
        move_old_slice[self.stack_axis] = slice(None, -1)
        move_new_slice[self.stack_axis] = slice(1, None)
        set_slice = (slice(None, None) for _ in self.observation_space.shape[:-1])
        self.buffer[tuple(move_old_slice)] = self.buffer[tuple(move_new_slice)]
        self.buffer[tuple(set_slice)] = observation.squeeze()
        return self.buffer.astype(dtype=self.observation_space.dtype)


class NoopReset(gym.Wrapper):
    def __init__(self, env, max_noops: int):
        super().__init__(env)
        self.max_noops = max_noops
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.env.unwrapped.np_random.randint(1, self.max_noops + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                self.env.reset(**kwargs)
        return obs
