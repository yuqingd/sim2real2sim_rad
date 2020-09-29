""" Kitchen environment for long horizon manipulation """
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from environments.kitchen.adept_envs.adept_envs import robot_env
from environments.kitchen.adept_envs.adept_envs.utils.configurable import configurable
from gym import spaces
from dm_control.mujoco import engine


@configurable(pickleable=True)
class KitchenV0(robot_env.RobotEnv):

    # CALIBRATION_PATHS = {
    #     'default':
    #     os.path.join(os.path.dirname(__file__), 'xarm/xarm_config.xml')
    # }
    # Converted to velocity actuation
    k = os.listdir()
    robot_class = (os.path.dirname(__file__).replace("/", ".") + '.xarm.xarm_robot:Robot_VelAct')[1:]
    robot_class = 'environments.kitchen.adept_envs.adept_envs.xarm.xarm_robot:Robot_VelAct'
    ROBOTS = {'robot': robot_class}
    KITCHEN_MODEl = os.path.join(
        os.path.dirname(__file__),
        'xarm/assets/kitchen_xarm.xml')
    KITCHEN_MODEl_NOKETTLE = os.path.join(
        os.path.dirname(__file__),
        'xarm/assets/kitchen_xarm_nokettle.xml')
    KITCHEN_MODEl_KETTLE = os.path.join(
        os.path.dirname(__file__),
        'xarm/assets/kitchen_kettleonly.xml')
    ROPE_MODEL = os.path.join(
        os.path.dirname(__file__),
        'xarm/assets/rope_xarm_real.xml')
    N_DOF_ROBOT = 13
    N_DOF_OBJECT = 21

    def __init__(self, robot_params={}, frame_skip=1, distance=2.5, azimuth=60, elevation=-30,
                 task_type='reach_microwave', init_range=.2, minimal=False):
        self.goal_concat = True
        self.obs_dict = {}
        self.robot_noise_ratio = 0.1  # 10% as per robot_config specs
        self.goal = np.zeros((30,))
        self.init_range = init_range

        if minimal:
            MODEL = self.KITCHEN_MODEl_KETTLE
        elif 'rope' in task_type:
            MODEL = self.ROPE_MODEL
        elif 'open_microwave' in task_type:
            MODEL = self.KITCHEN_MODEl_NOKETTLE
        else:
            MODEL = self.KITCHEN_MODEl

        super().__init__(
            MODEL,
            robot=self.make_robot(
                n_jnt=self.N_DOF_ROBOT,  #root+robot_jnts
                n_obj=self.N_DOF_OBJECT,
                **robot_params),
            frame_skip=frame_skip,
            camera_settings=dict(
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
            ),
        )

        self.act_mid = np.zeros(self.N_DOF_ROBOT)
        # self.act_amp = 2.0 * np.ones(self.N_DOF_ROBOT)
        self.act_amp = np.ones(self.N_DOF_ROBOT)

        act_lower = -1*np.ones((self.N_DOF_ROBOT,))
        act_upper =  1*np.ones((self.N_DOF_ROBOT,))
        self.action_space = spaces.Box(act_lower, act_upper)

        obs_upper = 8. * np.ones(self.obs_dim)
        obs_lower = -obs_upper
        self.observation_space = spaces.Box(obs_lower, obs_upper)

    def _get_reward_n_score(self, obs_dict):
        raise NotImplementedError()

    def step(self, a, b=None):
        a = np.clip(a, -1.0, 1.0)

        if not self.initializing:
            a = self.act_mid + a * self.act_amp  # mean center and scale
        else:
            self.goal = self._get_task_goal()  # update goal if init

        self.robot.step(
            self, a, step_duration=self.skip * self.model.opt.timestep)

        # observations
        obs = self._get_obs()

        #rewards
        reward_dict, score = self._get_reward_n_score(self.obs_dict)

        # termination
        done = False

        # finalize step
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
            'rewards': reward_dict,
            'score': score,
            'images': np.asarray(self.render(mode='rgb_array'))
        }
        # self.render()
        return obs, reward_dict['r_total'], done, env_info

    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        self.obs_dict = {}
        self.obs_dict['t'] = t
        self.obs_dict['qp'] = qp
        self.obs_dict['qv'] = qv
        self.obs_dict['obj_qp'] = obj_qp
        self.obs_dict['obj_qv'] = obj_qv
        self.obs_dict['goal'] = self.goal
        if self.goal_concat:
            return np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp'], self.obs_dict['goal']])

    def reset_model(self):
        reset_pos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        self.goal = self._get_task_goal()  #sample a new goal on reset
        return self._get_obs()

    def evaluate_success(self, paths):
        # score
        mean_score_per_rollout = np.zeros(shape=len(paths))
        for idx, path in enumerate(paths):
            mean_score_per_rollout[idx] = np.mean(path['env_infos']['score'])
        mean_score = np.mean(mean_score_per_rollout)

        # success percentage
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            num_success += bool(path['env_infos']['rewards']['bonus'][-1])
        success_percentage = num_success * 100.0 / num_paths

        # fuse results
        return np.sign(mean_score) * (
            1e6 * round(success_percentage, 2) + abs(mean_score))

    def close_env(self):
        self.robot.close()

    def set_goal(self, goal):
        self.goal = goal

    def _get_task_goal(self):
        return self.goal

    # Only include goal
    @property
    def goal_space(self):
        len_obs = self.observation_space.low.shape[0]
        env_lim = np.abs(self.observation_space.low[0])
        return spaces.Box(low=-env_lim, high=env_lim, shape=(len_obs//2,))

    def convert_to_active_observation(self, observation):
        return observation

class KitchenTaskRelaxV1(KitchenV0):
    """Kitchen environment with proper camera and goal setup"""

    def __init__(self, task_type='reach_microwave', distance=2.5, azimuth=60, elevation=-30, minimal=False):
        self.task_type = task_type
        super(KitchenTaskRelaxV1, self).__init__(distance=distance, azimuth=azimuth, elevation=elevation,
                                                 task_type=task_type, minimal=minimal)


    def _get_reward_n_score(self, obs_dict):
        reward_dict = {}
        reward_dict['true_reward'] = 0.
        reward_dict['bonus'] = 0.
        reward_dict['r_total'] = 0.
        score = 0.
        return reward_dict, score
