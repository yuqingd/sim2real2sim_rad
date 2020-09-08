from abc import ABC

import numpy as np
import gym
import mujoco_py

camera_0 = {
    'trackbodyid': -1,
    'distance': 1.5,
    'lookat': np.array((0.0, 0.6, 0)),
    'elevation': -60.0,
    'azimuth': 90
}

camera_1 = {
    'trackbodyid': -1,
    'distance': 1.5,
    'lookat': np.array((0.0, 0.6, 0)),
    'elevation': -60.0,
    'azimuth': 135
}

camera_2 = {
    'trackbodyid': -1,
    'distance': 1.5,
    'lookat': np.array((0.0, 0.6, 0)),
    'elevation': -60.0,
    'azimuth': 180
}

camera_3 = {
    'trackbodyid': -1,
    'distance': 1.5,
    'lookat': np.array((0.0, 0.6, 0)),
    'elevation': -60.0,
    'azimuth': 225
}

camera_4 = {
    'trackbodyid': -1,
    'distance': 1.5,
    'lookat': np.array((0.0, 0.6, 0)),
    'elevation': -60.0,
    'azimuth': 270
}

camera_5 = {
    'trackbodyid': -1,
    'distance': 1.5,
    'lookat': np.array((0.0, 0.6, 0)),
    'elevation': -60.0,
    'azimuth': 315
}

camera_6 = {
    'trackbodyid': -1,
    'distance': 1.5,
    'lookat': np.array((0.0, 0.6, 0)),
    'elevation': -60.0,
    'azimuth': 0
}

camera_7 = {
    'trackbodyid': -1,
    'distance': 1.5,
    'lookat': np.array((0.0, 0.6, 0)),
    'elevation': -60.0,
    'azimuth': 45
}

all_cameras = [camera_0, camera_1, camera_2, camera_3, camera_4, camera_5, camera_6, camera_7]


class MetaWorldEnv(gym.Env, ABC):
    def __init__(self, env, cameras, from_pixels=True, height=100, width=100, channels_first=True, offscreen=True):
        self._env = env
        self.cameras = cameras
        self.from_pixels = from_pixels
        self.height = height
        self.width = width
        self.channels_first = channels_first
        self.offscreen = offscreen

        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        shape = [3 * len(cameras), height, width] if channels_first else [height, width, 3 * len(cameras)]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

        action = self.action_space.sample()
        self.reset()
        self._state_obs, _reward, done, _info = self.step(action)
        assert not done

    @property
    def observation_space(self):
        if self.from_pixels:
            return self._observation_space
        else:
            return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def seed(self, seed=None):
        return self._env.seed(seed)

    def reset_model(self):
        self._env.reset()

    def viewer_setup(self, camera_id=0):
        for key, value in all_cameras[camera_id].items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def _get_obs(self):
        if self.from_pixels:
            imgs = []
            for c in self.cameras:
                imgs.append(self.render(mode='rgb_array', camera_id=c))
            if self.channels_first:
                return np.concatenate(imgs, axis=0)
            else:
                return np.concatenate(imgs, axis=2)
        else:
            return self._state_obs

    def step(self, action):
        self._state_obs, reward, done, info = self._env.step(action)
        if self._env.curr_path_length >= self._env.max_path_length:
            done = True
        return self._get_obs(), reward, done, info

    def reset(self):
        self._state_obs = self._env.reset()
        return self._get_obs()

    def set_state(self, qpos, qvel):
        self._env.set_state(qpos, qvel)

    @property
    def dt(self):
        return self._env.dt

    @property
    def _max_episode_steps(self):
        return self._env.max_path_length

    def do_simulation(self, ctrl, n_frames):
        self._env.do_simulatiaon(ctrl, n_frames)

    def render(self, mode='human', camera_id=0, height=None, width=None):
        if mode == 'human':
            self._env.render()

        if height is None:
            height = self.height
        if width is None:
            width = self.width

        if mode == 'rgb_array':
            viewer = self._get_viewer(camera_id)
            viewer.render(width, height, camera_id=-1)
            # window size used for old mujoco-py:
            data = viewer.read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            data = data[::-1, :, :]
            if self.channels_first:
                data = data.transpose((2, 0, 1))
            return data

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._env.close()

    def _get_viewer(self, camera_id):
        if self.viewer is None:
            if self.offscreen:
                from mujoco_py import GlfwContext
                GlfwContext(offscreen=True)
            self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        self.viewer_setup(camera_id)
        return self.viewer

    def get_body_com(self, body_name):
        return self._env.get_body_com(body_name)

    def state_vector(self):
        return self._env.state_vector
