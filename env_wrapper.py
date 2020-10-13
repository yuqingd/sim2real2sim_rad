from dm_control import suite
from environments.metaworld.envs.mujoco import env_dict as ed
import metaworld
import numpy as np
import gym
from mw_wrapper import MetaWorldEnv
import random
from dm_control.rl.control import PhysicsError
from dm_control import suite
from kitchen_env import Kitchen
import cv2

class DR_Env:
  def __init__(self, env, cameras, height=64, width=64, mean_only=False, dr_list=[], simple_randomization=False, dr_shape=None,
               real_world=False, dr=None, use_state="None", use_img=True, name="task_name",
               grayscale=False, dataset_step=None, range_scale=.1, prop_range_scale=False, state_concat=False,
               prop_initial_range=False):

    self._env = env

    self._size = (height, width)

    self.mean_only = mean_only
    self.dr_list = dr_list
    self.simple_randomization = simple_randomization
    self.dr_shape = dr_shape
    self.real_world = real_world
    self.use_state = use_state
    self.use_img = use_img
    self.dr = dr
    self.grayscale = grayscale
    self.name = name
    self.dataset_step = dataset_step
    self.range_scale = range_scale
    self.prop_range_scale = prop_range_scale
    self.state_concat = state_concat

    if prop_initial_range:
        self.initial_range = {}
        for param in dr_list:
            eps = 1e-3
            self.initial_range[param] = max(dr[param] * range_scale, eps)
    else:
        self.initial_range = None
    self.apply_dr()


  def __getattr__(self, attr):
    orig_attr = self._env.__getattribute__(attr)

    if callable(orig_attr):
      def hooked(*args, **kwargs):
        result = orig_attr(*args, **kwargs)
        return result
      return hooked
    else:
      return orig_attr

  def set_dataset_step(self, step):
    self.dataset_step = step

  def set_dr(self, dr):
    self.dr = dr


  def seed(self, seed=None):
      self._env.seed(seed)

  def set_real(self, param, param_name):
      value = self.real_dr_params[param_name]
      param[:] = value

  def update_dr_param(self, param, param_name, eps=1e-3, indices=None):
    if param_name in self.dr:
      if self.initial_range is not None:
          mean = self.dr[param_name]
          range = self.initial_range[param_name]
      elif self.mean_only:
        mean = self.dr[param_name]
        if self.prop_range_scale:
            range = max(self.range_scale * mean, eps)
        else:
            range = self.range_scale
      else:
        mean, range = self.dr[param_name]
        range = max(range, eps)  # TODO: consider re-adding anneal_range_scale
      new_value = np.random.uniform(low=max(mean - range, eps), high=max(mean + range, 2 * eps))
      if indices is None:
        param[:] = new_value
      else:
        try:
          for i in indices:
            param[i:i+1] = new_value
        except:
          param[indices:indices+1] = new_value

      self.sim_params += [new_value]
      self.distribution_mean += [mean]
      self.distribution_range += [range]

  def apply_dr(self):
    self.sim_params = []
    self.distribution_mean = []
    self.distribution_range = []


  @property
  def observation_space(self):
    spaces = {}
    if self.use_state:
      spaces['state'] = self._env.observation_space
    # spaces['state'] = gym.spaces.Discrete(spaces['state'])
    spaces['image'] = gym.spaces.Box(
       0, 255, self._size + (3,), dtype=np.uint8)
    #spaces['image'] = self._env.observation_space
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def env_step(self, action):
      obs, reward, done, info = self._env.step(action)
      if len(obs.shape) == 3:
          state = np.array([0])
      else:
          state = obs
          obs = self.render(mode='rgb_array')
      return obs, state, reward, done, info

  def step(self, action):

    obs, state, reward, done, info = self.env_step(action)
    obs_dict = {}

    if self.use_img:
        obs_dict['image'] = obs
    if self.use_state:
        if self.state_concat:
            obs_dict['state'] = np.concatenate([state, self.get_dr()])
        else:
            obs_dict['state'] = state
    obs_dict['real_world'] = 1.0 if self.real_world else 0.0
    obs_dict['sim_params'] = np.array(self.sim_params, dtype=np.float32)
    if not (self.dr is None) and not self.real_world:
      obs_dict['dr_params'] = self.get_dr()
    if 'success' in info:
        obs_dict['success'] = 1.0 if info['success'] else 0.0
    obs_dict['distribution_mean'] = np.array(self.distribution_mean, dtype=np.float32)
    obs_dict['distribution_range'] = np.array(self.distribution_range, dtype=np.float32)

    return obs_dict, reward, done, info

  def get_dr(self):
    return np.array([], dtype=np.float32)

  def env_reset(self):
      state_obs = self._env.reset()
      img_obs = self.render(mode='rgb_array')
      return state_obs, img_obs

  def reset(self):
    self.apply_dr()
    state_obs, img_obs = self.env_reset()

    obs_dict = {}

    if self.use_img:
      obs_dict['image'] = img_obs
    if self.use_state:
        if self.state_concat:
            obs_dict['state'] = np.concatenate([state_obs, self.get_dr()])
        else:
            obs_dict['state'] = state_obs

    obs_dict['real_world'] = 1.0 if self.real_world else 0.0
    if not (self.dr is None) and not self.real_world:
      obs_dict['dr_params'] = self.get_dr()
    obs_dict['success'] = 0.0
    obs_dict['sim_params'] = np.array(self.sim_params, dtype=np.float32)
    obs_dict['distribution_mean'] = np.array(self.distribution_mean, dtype=np.float32)
    obs_dict['distribution_range'] = np.array(self.distribution_range, dtype=np.float32)
    return obs_dict

  def get_state(self, state_obs):
    return state_obs

  def render(self, size=None, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    kwargs['mode'] = 'rgb_array'
    height, width = self._size
    return self._env.render(width=width, height=height, **kwargs)


class DR_MetaWorldEnv(DR_Env):  # TODO: consider passing through as kwargs
    def __init__(self, env, cameras, height=64, width=64, mean_only=False, dr_list=[], simple_randomization=False,
                 dr_shape=None, real_world=False, dr=None, use_state="None", use_img=True, name="task_name",
                 grayscale=False, delay_steps=0, range_scale=.1, **kwargs):
        env = MetaWorldEnv(env, from_pixels=use_img, cameras=cameras, height=height, width=width)
        super().__init__(env, cameras,
                         height=height, width=width,
                         mean_only=mean_only,
                         dr_list=dr_list,
                         simple_randomization=simple_randomization,
                         dr_shape=dr_shape,
                         real_world=real_world,
                         dr=dr,
                         use_state=use_state,
                         use_img=use_img,
                         name=name,
                         grayscale=grayscale,
                         range_scale=range_scale,
                         **kwargs)

    def render(self, mode='rgb_array', camera_id=2, **kwargs):
        obs = super().render(mode, camera_id=camera_id, **kwargs)
        return obs


    @property
    def observation_space(self):
        spaces = {}
        if self.use_state is not "None":
            if self.use_state == 'all':
                spaces['state'] = self._env.observation_space
            else:
                spaces['state'] = 3
        spaces['state'] = gym.spaces.Discrete(spaces['state'])
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    def get_state(self, state_obs):
        if self.use_state == 'robot':
            return state_obs[:3]  # Only include robot state (endeffector pos)
        elif self.use_state == 'all':
            return state_obs
        else:
            raise NotImplementedError(self.use_state)

    def apply_dr(self):
        self.sim_params = []
        self.distribution_mean = []
        self.distribution_range = []
        if self.dr is None or self.real_world:
            self.sim_params = self.get_dr()
            self.distribution_mean = self.get_dr()
            self.distribution_range = np.zeros(self.dr_shape, dtype=np.float32)
            return

        model = self._env._env.sim.model
        geom_dict = model._geom_name2id
        body_dict = model._body_name2id
        robot_geom = [
            geom_dict['right_arm_base_link_geom'],
            geom_dict['right_l0_geom'],
            geom_dict['right_l1_geom'],
            geom_dict['right_l2_geom'],
            geom_dict['right_l3_geom'],
            geom_dict['right_l4_geom'],
            geom_dict['right_l5_geom'],
            geom_dict['right_l6_geom'],
            geom_dict['right_hand_geom'],
            geom_dict['head_geom'],
        ]
        table_geom = geom_dict['tableTop']

        dr_update_dict_common = {
            # Table
            "table_friction": (model.geom_friction[table_geom: table_geom + 1], None),
            "table_r": (model.geom_rgba[table_geom: table_geom + 1, 0], None),
            "table_g": (model.geom_rgba[table_geom: table_geom + 1, 1], None),
            "table_b": (model.geom_rgba[table_geom: table_geom + 1, 2], None),

            # Robot
            'robot_r': (model.geom_rgba[:, 0], robot_geom),
            'robot_g': (model.geom_rgba[:, 1], robot_geom),
            'robot_b': (model.geom_rgba[:, 2], robot_geom),
            'robot_friction': (model.geom_friction[:, 0], robot_geom),
        }

        if self.name in ['stick-pull', 'stick-push']:
            stick_body = body_dict['stick']
            stick_geom = geom_dict['stick_geom_1']

            object_body = body_dict['object']
            object_geom_body = [
                geom_dict['object_geom_1'],
            ]
            object_geom_handle = [
                geom_dict['object_geom_handle_1'],
                geom_dict['object_geom_handle_2'],
                geom_dict['handle']
            ]
            object_geom = object_geom_body + object_geom_handle

            dr_update_dict = {
                # Stick
                "stick_mass": (model.body_mass[stick_body: stick_body + 1], None),
                "stick_friction": (model.geom_friction[stick_geom: stick_geom + 1, 0], None),
                "stick_r": (model.geom_rgba[stick_geom: stick_geom + 1, 0], None),
                "stick_g": (model.geom_rgba[stick_geom: stick_geom + 1, 1], None),
                "stick_b": (model.geom_rgba[stick_geom: stick_geom + 1, 2], None),

                # Object
                "object_mass": (model.body_mass[object_body: object_body + 1], None),
                "object_friction": (model.geom_friction[:, 0], object_geom),
                "object_body_r": (model.geom_rgba[:, 0], object_geom_body),
                "object_body_g": (model.geom_rgba[:, 1], object_geom_body),
                "object_body_b": (model.geom_rgba[:, 2], object_geom_body),
                "object_handle_r": (model.geom_rgba[:, 0], object_geom_handle),
                "object_handle_g": (model.geom_rgba[:, 1], object_geom_handle),
                "object_handle_b": (model.geom_rgba[:, 2], object_geom_handle),
            }
            dr_update_dict.update(dr_update_dict_common)

            for dr_param in self.dr_list:
                arr, indices = dr_update_dict[dr_param]
                self.update_dr_param(arr, dr_param, indices=indices)

        elif 'basketball' in self.name:
            basket_goal_geom = [
                geom_dict['handle'],
                geom_dict['basket_goal_geom_1'],
                geom_dict['basket_goal_geom_2'],
            ]
            backboard_geom = [
                geom_dict['basket_goal'],
            ]
            basket_geom = basket_goal_geom + backboard_geom

            object_body = body_dict['obj']
            object_geom = [geom_dict['objGeom']]

            dr_update_dict = {
                # Stick
                "basket_friction": (model.geom_rgba[:, 2], basket_geom),
                "basket_goal_r": (model.geom_rgba[:, 0], basket_goal_geom),
                "basket_goal_g": (model.geom_rgba[:, 1], basket_goal_geom),
                "basket_goal_b": (model.geom_rgba[:, 2], basket_goal_geom),
                "backboard_r": (model.geom_rgba[:, 0], backboard_geom),
                "backboard_g": (model.geom_rgba[:, 1], backboard_geom),
                "backboard_b": (model.geom_rgba[:, 2], backboard_geom),

                # Object
                "object_mass": (model.body_mass[object_body: object_body + 1], None),
                "object_friction": (model.geom_friction[:, 0], object_geom),
                "object_r": (model.geom_rgba[:, 0], object_geom),
                "object_g": (model.geom_rgba[:, 1], object_geom),
                "object_b": (model.geom_rgba[:, 2], object_geom),
            }
            dr_update_dict.update(dr_update_dict_common)
        else:
            self.sim_params = self.get_dr()
            self.distribution_mean = self.get_dr()
            self.distribution_range = np.zeros(self.dr_shape, dtype=np.float32)
            return

        for dr_param in self.dr_list:
            arr, indices = dr_update_dict[dr_param]
            self.update_dr_param(arr, dr_param, indices=indices)

    def get_dr(self):
        model = self._env._env.sim.model
        if self.simple_randomization:
            if 'stick-pull' or 'stick-push' in self.name:
                cylinder_body = model.body_name2id('cylinder')
                return np.array([model.body_mass[cylinder_body]])
            elif 'basketball' in self.name:
                microwave_index = model.body_name2id('microdoorroot')
                return np.array([model.body_mass[microwave_index]])
            else:
                raise NotImplementedError

        geom_dict = model._geom_name2id
        body_dict = model._body_name2id
        robot_geom = [
            geom_dict['right_arm_base_link_geom'],
            geom_dict['right_l0_geom'],
            geom_dict['right_l1_geom'],
            geom_dict['right_l2_geom'],
            geom_dict['right_l3_geom'],
            geom_dict['right_l4_geom'],
            geom_dict['right_l5_geom'],
            geom_dict['right_l6_geom'],
            geom_dict['right_hand_geom'],
            geom_dict['head_geom'],
        ]
        table_geom = geom_dict['tableTop']

        dr_update_dict_common = {
            # Table
            "table_friction": model.geom_friction[table_geom, 0],
            "table_r": model.geom_rgba[table_geom, 0],
            "table_g": model.geom_rgba[table_geom, 1],
            "table_b": model.geom_rgba[table_geom, 2],

            # Robot
            'robot_r': model.geom_rgba[robot_geom[0], 0],
            'robot_g': model.geom_rgba[robot_geom[0], 1],
            'robot_b': model.geom_rgba[robot_geom[0], 2],
            'robot_friction': model.geom_friction[robot_geom[0], 0],
        }

        if self.name in ['stick-pull', 'stick-push']:
            model = self._env._env.sim.model
            geom_dict = model._geom_name2id
            body_dict = model._body_name2id

            stick_body = body_dict['stick']
            stick_geom = geom_dict['stick_geom_1']

            object_body = body_dict['object']
            object_geom_body = [
                geom_dict['object_geom_1'],
            ]
            object_geom_handle = [
                geom_dict['object_geom_handle_1'],
                geom_dict['object_geom_handle_2'],
                geom_dict['handle']
            ]
            object_geom = object_geom_body + object_geom_handle

            dr_update_dict = {
                # Stick
                "stick_mass": model.body_mass[stick_body],
                "stick_friction": model.geom_friction[stick_geom, 0],
                "stick_r": model.geom_rgba[stick_geom, 0],
                "stick_g": model.geom_rgba[stick_geom, 1],
                "stick_b": model.geom_rgba[stick_geom, 2],

                # Object
                "object_mass": model.body_mass[object_body],
                "object_friction": model.geom_friction[object_geom[0], 0],
                "object_body_r": model.geom_rgba[object_geom_body[0], 0],
                "object_body_g": model.geom_rgba[object_geom_body[0], 1],
                "object_body_b": model.geom_rgba[object_geom_body[0], 2],
                "object_handle_r": model.geom_rgba[object_geom_handle[0], 0],
                "object_handle_g": model.geom_rgba[object_geom_handle[0], 1],
                "object_handle_b": model.geom_rgba[object_geom_handle[0], 2],
            }
            dr_update_dict.update(dr_update_dict_common)

            dr_list = []
            for dr_param in self.dr_list:
                dr_list.append(dr_update_dict[dr_param])
            arr = np.array(dr_list)
        elif 'basketball' in self.name:
            basket_goal_geom = [
                geom_dict['handle'],
                geom_dict['basket_goal_geom_1'],
                geom_dict['basket_goal_geom_2'],
            ]
            backboard_geom = [
                geom_dict['basket_goal'],
            ]
            basket_geom = basket_goal_geom + backboard_geom

            object_body = body_dict['obj']
            object_geom = [geom_dict['objGeom']]

            dr_update_dict = {
                # Stick
                "basket_friction": model.geom_rgba[basket_geom[0], 2],
                "basket_goal_r": model.geom_rgba[basket_goal_geom[0], 0],
                "basket_goal_g": model.geom_rgba[basket_goal_geom[0], 1],
                "basket_goal_b": model.geom_rgba[basket_goal_geom[0], 2],
                "backboard_r": model.geom_rgba[basket_goal_geom[0], 0],
                "backboard_g": model.geom_rgba[basket_goal_geom[0], 1],
                "backboard_b": model.geom_rgba[basket_goal_geom[0], 2],

                # Object
                "object_mass": model.body_mass[object_body],
                "object_friction": model.geom_friction[object_geom[0], 0],
                "object_r": model.geom_rgba[object_geom[0], 0],
                "object_g": model.geom_rgba[object_geom[0], 1],
                "object_b": model.geom_rgba[object_geom[0], 2],
            }
            dr_update_dict.update(dr_update_dict_common)

            dr_list = []
            for dr_param in self.dr_list:
                dr_list.append(dr_update_dict[dr_param])
            arr = np.array(dr_list)
        else:
            arr = np.zeros(self.dr_shape, dtype=np.float32)

        arr = arr.astype(np.float32)
        return arr

class DR_Kitchen(DR_Env):
    def __init__(self, env, cameras, height=100, width=100, mean_only=False, dr_list=[], simple_randomization=False,
                 dr_shape=None, real_world=False, dr=None, use_state="None", use_img=True, name="task_name",
                 grayscale=False, domain_name="", range_scale=.1, **kwargs):
        super().__init__(env, cameras,
                         height=height, width=width,
                         mean_only=mean_only,
                         dr_list=dr_list,
                         simple_randomization=simple_randomization,
                         dr_shape=dr_shape,
                         real_world=real_world,
                         dr=dr,
                         use_state=use_state,
                         use_img=use_img,
                         name=name,
                         grayscale=grayscale,
                         range_scale=range_scale,
                         **kwargs)

    def __getattr__(self, attr):
        orig_attr = self._env.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr

    @property
    def observation_space(self):
        spaces = {}

        if self.use_state is not "None":
            state_shape = 4 if self.use_gripper else 3  # 2 for fingers, 3 for end effector position
            state_shape = self.goal.shape[0] + state_shape
            if self.use_state == 'all':
                state_shape += 3
            spaces['state'] = gym.spaces.Box(np.array([-float('inf')] * state_shape),
                                             np.array([-float('inf')] * state_shape))
        else:
            spaces['state'] = gym.spaces.Box(np.array([-float('inf')] * self.goal.shape[0]),
                                             np.array([float('inf')] * self.goal.shape[0]))
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)


    def apply_dr(self):
        self.sim_params = []
        self.distribution_mean = []
        self.distribution_range = []
        if self.dr is None or self.real_world:
            self.sim_params = self.get_dr()
            self.distribution_mean = self.get_dr()
            self.distribution_range = np.zeros(self.dr_shape, dtype=np.float32)
            return  # TODO: start using XPOS_INDICES or equivalent for joints.

        if 'rope' in self.task:
            xarm_viz_indices = 2
            model = self._env._env.sim.model
            # cylinder
            cylinder_viz = model.geom_name2id('cylinder_viz')
            cylinder_body = model.body_name2id('cylinder')

            # box
            box_viz_1 = model.geom_name2id('box_viz_1')
            box_viz_2 = model.geom_name2id('box_viz_2')
            box_viz_3 = model.geom_name2id('box_viz_3')
            box_viz_4 = model.geom_name2id('box_viz_4')
            box_viz_5 = model.geom_name2id('box_viz_5')
            # try:
            #     box_viz_6 = model.geom_name2id('box_viz_6')
            #     box_viz_7 = model.geom_name2id('box_viz_7')
            #     box_viz_8 = model.geom_name2id('box_viz_8')
            # except:
            #     pass

            dr_update_dict = {
                'joint1_damping': (model.dof_damping[0:1], None),
                'joint2_damping': (model.dof_damping[1:2], None),
                'joint3_damping': (model.dof_damping[2:3], None),
                'joint4_damping': (model.dof_damping[3:4], None),
                'joint5_damping': (model.dof_damping[4:5], None),
                'joint6_damping': (model.dof_damping[5:6], None),
                'joint7_damping': (model.dof_damping[6:7], None),
                'robot_r': (model.geom_rgba[:, 0], xarm_viz_indices),
                'robot_g': (model.geom_rgba[:, 1], xarm_viz_indices),
                'robot_b': (model.geom_rgba[:, 2], xarm_viz_indices),
                'cylinder_r': (model.geom_rgba[:, 0], cylinder_viz),
                'cylinder_g': (model.geom_rgba[:, 1], cylinder_viz),
                'cylinder_b': (model.geom_rgba[:, 2], cylinder_viz),
                'cylinder_mass': (model.body_mass[cylinder_body:cylinder_body + 1], None),

                'box1_r': (model.geom_rgba[:, 0], box_viz_1),
                'box1_g': (model.geom_rgba[:, 1], box_viz_1),
                'box1_b': (model.geom_rgba[:, 2], box_viz_1),
                'box2_r': (model.geom_rgba[:, 0], box_viz_2),
                'box2_g': (model.geom_rgba[:, 1], box_viz_2),
                'box2_b': (model.geom_rgba[:, 2], box_viz_2),
                'box3_r': (model.geom_rgba[:, 0], box_viz_3),
                'box3_g': (model.geom_rgba[:, 1], box_viz_3),
                'box3_b': (model.geom_rgba[:, 2], box_viz_3),
                'box4_r': (model.geom_rgba[:, 0], box_viz_4),
                'box4_g': (model.geom_rgba[:, 1], box_viz_4),
                'box4_b': (model.geom_rgba[:, 2], box_viz_4),
                'box5_r': (model.geom_rgba[:, 0], box_viz_5),
                'box5_g': (model.geom_rgba[:, 1], box_viz_5),
                'box5_b': (model.geom_rgba[:, 2], box_viz_5),
                # 'box6_r': (model.geom_rgba[:, 0], box_viz_6),
                # 'box6_g': (model.geom_rgba[:, 1], box_viz_6),
                # 'box6_b': (model.geom_rgba[:, 2], box_viz_6),
                # 'box7_r': (model.geom_rgba[:, 0], box_viz_7),
                # 'box7_g': (model.geom_rgba[:, 1], box_viz_7),
                # 'box7_b': (model.geom_rgba[:, 2], box_viz_7),
                # 'box8_r': (model.geom_rgba[:, 0], box_viz_8),
                # 'box8_g': (model.geom_rgba[:, 1], box_viz_8),
                # 'box8_b': (model.geom_rgba[:, 2], box_viz_8),

                'rope_damping': (model.tendon_damping, None),
                'rope_friction': (model.tendon_frictionloss, None),
                'rope_stiffness': (model.tendon_stiffness, None),

                'lighting': (model.light_diffuse[:3], None),
            }
            for dr_param in self.dr_list:
                arr, indices = dr_update_dict[dr_param]
                self.update_dr_param(arr, dr_param, indices=indices)

        else:
            model = self._env._env.sim.model
            geom_dict = model._geom_name2id
            stove_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "stove_collision" in name]
            stove_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "stove_viz" in name]
            xarm_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "xarm_viz" in name]

            dr_update_dict = {
                'joint1_damping': (model.dof_damping[0:1], None),
                'joint2_damping': (model.dof_damping[1:2], None),
                'joint3_damping': (model.dof_damping[2:3], None),
                'joint4_damping': (model.dof_damping[3:4], None),
                'joint5_damping': (model.dof_damping[4:5], None),
                'joint6_damping': (model.dof_damping[5:6], None),
                'joint7_damping': (model.dof_damping[6:7], None),

                'knob_mass': (model.body_mass, [22, 24, 26, 28]),
                'lighting': (model.light_diffuse[:3], None),

                'robot_r': (model.geom_rgba[:, 0], xarm_viz_indices),
                'robot_g': (model.geom_rgba[:, 1], xarm_viz_indices),
                'robot_b': (model.geom_rgba[:, 2], xarm_viz_indices),
                'stove_r': (model.geom_rgba[:, 0], stove_viz_indices),
                'stove_g': (model.geom_rgba[:, 1], stove_viz_indices),
                'stove_b': (model.geom_rgba[:, 2], stove_viz_indices),
                'stove_friction': (model.geom_friction[:, 0], stove_collision_indices),

            }

            if self.has_microwave:
                microwave_index = model.body_name2id('microdoorroot')
                microwave_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "microwave_viz" in name]
                microwave_collision_indices = [geom_dict[name] for name in geom_dict.keys() if
                                               "microwave_collision" in name]
                dr_update_dict_k = {
                    'microwave_r': (model.geom_rgba[:, 0], microwave_viz_indices),
                    'microwave_g': (model.geom_rgba[:, 1], microwave_viz_indices),
                    'microwave_b': (model.geom_rgba[:, 2], microwave_viz_indices),
                    'microwave_friction': (model.geom_friction[:, 0], microwave_collision_indices),
                    'microwave_mass': (model.body_mass[microwave_index: microwave_index + 1], None),
                }
                dr_update_dict.update(dr_update_dict_k)

            if self.has_cabinet:
                cabinet_index = model.body_name2id('slidelink')
                cabinet_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "cabinet_viz" in name]
                cabinet_collision_indices = [geom_dict[name] for name in geom_dict.keys() if
                                             "cabinet_collision" in name]
                dr_update_dict_k = {
                    'cabinet_r': (model.geom_rgba[:, 0], cabinet_viz_indices),
                    'cabinet_g': (model.geom_rgba[:, 1], cabinet_viz_indices),
                    'cabinet_b': (model.geom_rgba[:, 2], cabinet_viz_indices),
                    'cabinet_friction': (model.geom_friction[:, 0], cabinet_collision_indices),
                    'cabinet_mass': (model.body_mass[cabinet_index: cabinet_index + 1], None),
                }
                dr_update_dict.update(dr_update_dict_k)

            # Kettle
            if self.has_kettle:
                kettle_index = model.body_name2id('kettleroot')
                kettle_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "kettle_viz" in name]

                dr_update_dict_k = {
                    'kettle_r': (model.geom_rgba[:, 0], kettle_viz_indices),
                    'kettle_g': (model.geom_rgba[:, 1], kettle_viz_indices),
                    'kettle_b': (model.geom_rgba[:, 2], kettle_viz_indices),
                    'kettle_friction': (model.geom_friction[:, 0], kettle_viz_indices),
                    'kettle_mass': (model.body_mass[kettle_index: kettle_index + 1], None),

                }
                dr_update_dict.update(dr_update_dict_k)

            # Actually Update
            for dr_param in self.dr_list:
                arr, indices = dr_update_dict[dr_param]
                self.update_dr_param(arr, dr_param, indices=indices)

    def get_dr(self):
        model = self._env._env.sim.model
        if self.simple_randomization:
            if 'rope' in self.task:
                cylinder_body = model.body_name2id('cylinder')
                return np.array([model.body_mass[cylinder_body]])
            elif 'open_microwave' in self.task:
                microwave_index = model.body_name2id('microdoorroot')
                return np.array([model.body_mass[microwave_index]])
            elif 'open_cabinet' in self.task:
                cabinet_index = model.body_name2id('slidelink')
                return np.array([model.body_mass[cabinet_index]])
            else:
                kettle_index = model.body_name2id('kettleroot')
                return np.array([model.body_mass[kettle_index]])
        if 'rope' in self.task:
            cylinder_viz = model.geom_name2id('cylinder_viz')
            cylinder_body = model.body_name2id('cylinder')
            box_viz_1 = model.geom_name2id('box_viz_1')
            box_viz_2 = model.geom_name2id('box_viz_2')
            box_viz_3 = model.geom_name2id('box_viz_3')
            box_viz_4 = model.geom_name2id('box_viz_4')
            box_viz_5 = model.geom_name2id('box_viz_5')
            # box_viz_6 = model.geom_name2id('box_viz_6')
            # box_viz_7 = model.geom_name2id('box_viz_7')
            # box_viz_8 = model.geom_name2id('box_viz_8')
            xarm_viz_indices = 2  # [geom_dict[name] for name in geom_dict.keys() if "xarm_viz" in name]
            model = model

            dr_update_dict = {
                'joint1_damping': model.dof_damping[0],
                'joint2_damping': model.dof_damping[1],
                'joint3_damping': model.dof_damping[2],
                'joint4_damping': model.dof_damping[3],
                'joint5_damping': model.dof_damping[4],
                'joint6_damping': model.dof_damping[5],
                'joint7_damping': model.dof_damping[6],
                'robot_r': model.geom_rgba[xarm_viz_indices, 0],
                'robot_g': model.geom_rgba[xarm_viz_indices, 1],
                'robot_b': model.geom_rgba[xarm_viz_indices, 2],
                'cylinder_r': model.geom_rgba[cylinder_viz, 0],
                'cylinder_g': model.geom_rgba[cylinder_viz, 1],
                'cylinder_b': model.geom_rgba[cylinder_viz, 2],
                'cylinder_mass': model.body_mass[cylinder_body],

                'box1_r': model.geom_rgba[box_viz_1, 0],
                'box1_g': model.geom_rgba[box_viz_1, 1],
                'box1_b': model.geom_rgba[box_viz_1, 2],
                'box2_r': model.geom_rgba[box_viz_2, 0],
                'box2_g': model.geom_rgba[box_viz_2, 1],
                'box2_b': model.geom_rgba[box_viz_2, 2],
                'box3_r': model.geom_rgba[box_viz_3, 0],
                'box3_g': model.geom_rgba[box_viz_3, 1],
                'box3_b': model.geom_rgba[box_viz_3, 2],
                'box4_r': model.geom_rgba[box_viz_4, 0],
                'box4_g': model.geom_rgba[box_viz_4, 1],
                'box4_b': model.geom_rgba[box_viz_4, 2],
                'box5_r': model.geom_rgba[box_viz_5, 0],
                'box5_g': model.geom_rgba[box_viz_5, 1],
                'box5_b': model.geom_rgba[box_viz_5, 2],
                # 'box6_r': model.geom_rgba[box_viz_6, 0],
                # 'box6_g': model.geom_rgba[box_viz_6, 1],
                # 'box6_b': model.geom_rgba[box_viz_6, 2],
                # 'box7_r': model.geom_rgba[box_viz_7, 0],
                # 'box7_g': model.geom_rgba[box_viz_7, 1],
                # 'box7_b': model.geom_rgba[box_viz_7, 2],
                # 'box8_r': model.geom_rgba[box_viz_8, 0],
                # 'box8_g': model.geom_rgba[box_viz_8, 1],
                # 'box8_b': model.geom_rgba[box_viz_8, 2],

                'rope_damping': model.tendon_damping[0],
                'rope_friction': model.tendon_frictionloss[0],
                'rope_stiffness': model.tendon_stiffness[0],

                'lighting': model.light_diffuse[0, 0],
            }

            dr_list = []
            for dr_param in self.dr_list:
                dr_list.append(dr_update_dict[dr_param])
            arr = np.array(dr_list)

        else:
            geom_dict = model._geom_name2id
            stove_collision_indices = [geom_dict[name] for name in geom_dict.keys() if
                                       "stove_collision" in name][0]
            stove_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "stove_viz" in name][0]
            xarm_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "xarm_viz" in name][0]
            model = model

            dr_update_dict = {
                'joint1_damping': model.dof_damping[0],
                'joint2_damping': model.dof_damping[1],
                'joint3_damping': model.dof_damping[2],
                'joint4_damping': model.dof_damping[3],
                'joint5_damping': model.dof_damping[4],
                'joint6_damping': model.dof_damping[5],
                'joint7_damping': model.dof_damping[6],

                'lighting': model.light_diffuse[0, 0],

                'robot_r': model.geom_rgba[xarm_viz_indices, 0],
                'robot_g': model.geom_rgba[xarm_viz_indices, 1],
                'robot_b': model.geom_rgba[xarm_viz_indices, 2],
                'stove_r': model.geom_rgba[stove_viz_indices, 0],
                'stove_g': model.geom_rgba[stove_viz_indices, 1],
                'stove_b': model.geom_rgba[stove_viz_indices, 2],
                'stove_friction': model.geom_friction[stove_collision_indices, 0],
            }

            if self.has_cabinet:
                cabinet_index = model.body_name2id('slidelink')
                cabinet_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "cabinet_viz" in name][0]
                cabinet_collision_indices = \
                [geom_dict[name] for name in geom_dict.keys() if "cabinet_collision" in name][0]
                dr_update_dict_k = {
                    'cabinet_r': model.geom_rgba[cabinet_viz_indices, 0],
                    'cabinet_g': model.geom_rgba[cabinet_viz_indices, 1],
                    'cabinet_b': model.geom_rgba[cabinet_viz_indices, 2],
                    'cabinet_friction': model.geom_friction[cabinet_collision_indices, 0],
                    'cabinet_mass': model.body_mass[cabinet_index],
                    'knob_mass': model.body_mass[22],
                }
                dr_update_dict.update(dr_update_dict_k)

            if self.has_microwave:
                microwave_index = model.body_name2id('microdoorroot')
                microwave_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "microwave_viz" in name][0]
                microwave_collision_indices = \
                [geom_dict[name] for name in geom_dict.keys() if "microwave_collision" in name][0]
                dr_update_dict_k = {
                    'microwave_r': model.geom_rgba[microwave_viz_indices, 0],
                    'microwave_g': model.geom_rgba[microwave_viz_indices, 1],
                    'microwave_b': model.geom_rgba[microwave_viz_indices, 2],
                    'microwave_friction': model.geom_friction[microwave_collision_indices, 0],
                    'microwave_mass': model.body_mass[microwave_index],
                }
                dr_update_dict.update(dr_update_dict_k)

            if self.has_kettle:
                kettle_index = model.body_name2id('kettleroot')
                kettle_viz_indices = [geom_dict[name] for name in geom_dict.keys() if "kettle_viz" in name][0]
                kettle_collision_indices = [geom_dict[name] for name in geom_dict.keys() if "kettle_collision" in name][
                    0]
                dr_update_dict_k = {
                    'kettle_r': model.geom_rgba[kettle_viz_indices, 0],
                    'kettle_g': model.geom_rgba[kettle_viz_indices, 1],
                    'kettle_b': model.geom_rgba[kettle_viz_indices, 2],
                    'kettle_friction': model.geom_friction[kettle_collision_indices, 0],
                    'kettle_mass': model.body_mass[kettle_index],
                }
                dr_update_dict.update(dr_update_dict_k)

            dr_list = []
            for dr_param in self.dr_list:
                dr_list.append(dr_update_dict[dr_param])
            arr = np.array(dr_list)
        arr = arr.astype(np.float32)
        return arr


class DR_DMCEnv(DR_Env):
    def __init__(self, env, cameras, height=64, width=64, mean_only=False, dr_list=[], simple_randomization=False,
                 dr_shape=None, real_world=False, dr=None, use_state="None", use_img=True, name="task_name",
                 grayscale=False, domain_name="", range_scale=.1, real_dr_params=None, **kwargs):
        self.domain_name = domain_name
        super().__init__(env, cameras,
                         height=height, width=width,
                         mean_only=mean_only,
                         dr_list=dr_list,
                         simple_randomization=simple_randomization,
                         dr_shape=dr_shape,
                         real_world=real_world,
                         dr=dr,
                         use_state=use_state,
                         use_img=use_img,
                         name=name,
                         grayscale=grayscale,
                         range_scale=range_scale, **kwargs)
        # Set sim params to the desired value.
        if real_dr_params is not None:
            self.real_dr_params = real_dr_params
            self.apply_dr(set_real=True)

    def render(self, mode, **kwargs):
        obs = super().render(mode, **kwargs)
        obs = np.transpose(obs, (2, 0, 1))
        return obs

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    def apply_dr(self, set_real=False):
        self.sim_params = []
        self.distribution_mean = []
        self.distribution_range = []
        if (not set_real) and (self.dr is None or self.real_world):
            self.sim_params = self.get_dr()
            self.distribution_mean = self.get_dr()
            self.distribution_range = np.zeros(self.dr_shape, dtype=np.float32)
            return

        model = self._env.physics.model
        if 'ball_in_cup' in self.domain_name:
            dr_update_dict = {
                "cup_mass": model.body_mass[1:2],
                "ball_mass": model.body_mass[2:3],
                "cup_damping": model.dof_damping[0:2],
                "ball_damping": model.dof_damping[2:4],
                "actuator_gain": model.actuator_gainprm[:, 0],
                "cup_r": model.geom_rgba[1:6, 0],
                "cup_g": model.geom_rgba[1:6, 1],
                "cup_b": model.geom_rgba[1:6, 2],
                "ball_r": model.geom_rgba[6:7, 0],
                "ball_g": model.geom_rgba[6:7, 1],
                "ball_b": model.geom_rgba[6:7, 2],
                "ground_r": model.geom_rgba[0:1, 0],
                "ground_g": model.geom_rgba[0:1, 1],
                "ground_b": model.geom_rgba[0:1, 2],
            }
        elif "walker" in self.domain_name:
            dr_update_dict = {
                "torso_mass": model.body_mass[1:2],
                "right_thigh_mass": model.body_mass[2:3],
                "right_leg_mass": model.body_mass[3:4],
                "right_foot_mass": model.body_mass[4:5],
                "left_thigh_mass": model.body_mass[5:6],
                "left_leg_mass": model.body_mass[6:7],
                "left_foot_mass": model.body_mass[7:8],
                "right_hip": model.dof_damping[3:4],
                "right_knee": model.dof_damping[4:5],
                "right_ankle": model.dof_damping[5:6],
                "left_hip": model.dof_damping[6:7],
                "left_knee": model.dof_damping[7:8],
                "left_ankle": model.dof_damping[8:9],
                "ground_r": model.geom_rgba[0:1, 0],
                "ground_g": model.geom_rgba[0:1, 1],
                "ground_b": model.geom_rgba[0:1, 2],
                "body_r": model.geom_rgba[1:8, 0],
                "body_g": model.geom_rgba[1:8, 1],
                "body_b": model.geom_rgba[1:8, 2],
            }
        elif "cheetah" in self.domain_name:
            dr_update_dict = {
                "torso_mass": model.body_mass[1:2],
                "bthigh_mass": model.body_mass[2:3],
                "bshin_mass": model.body_mass[3:4],
                "bfoot_mass": model.body_mass[4:5],
                "fthigh_mass": model.body_mass[5:6],
                "fshin_mass": model.body_mass[6:7],
                "ffoot_mass": model.body_mass[7:8],
                "bthigh_damping": model.dof_damping[3:4],
                "bshin_damping": model.dof_damping[4:5],
                "bfoot_damping": model.dof_damping[5:6],
                "fthigh_damping": model.dof_damping[6:7],
                "fshin_damping": model.dof_damping[7:8],
                "ffoot_damping": model.dof_damping[8:9],
                "ground_r": model.geom_rgba[0:1, 0],
                "ground_g": model.geom_rgba[0:1, 1],
                "ground_b": model.geom_rgba[0:1, 2],
                "body_r": model.geom_rgba[1:9, 0],
                "body_g": model.geom_rgba[1:9, 1],
                "body_b": model.geom_rgba[1:9, 2],
            }
        elif "finger" in self.domain_name:
            dr_update_dict = {
                "proximal_mass": model.body_mass[0:1],
                "distal_mass": model.body_mass[1:2],
                "spinner_mass": model.body_mass[2:3],
                "proximal_damping": model.dof_damping[0:1],
                "distal_damping": model.dof_damping[1:2],
                "hinge_damping": model.dof_damping[2:3],
                "ground_r": model.geom_rgba[0:1, 0],
                "ground_g": model.geom_rgba[0:1, 1],
                "ground_b": model.geom_rgba[0:1, 2],
                "finger_r": model.geom_rgba[2:4, 0],
                "finger_g": model.geom_rgba[2:4, 1],
                "finger_b": model.geom_rgba[2:4, 2],
                "hotdog_r": model.geom_rgba[5:7, 0],
                "hotdog_g": model.geom_rgba[5:7, 1],
                "hotdog_b": model.geom_rgba[5:7, 2],
            }
        # Actually Update
        for dr_param in self.dr_list:
            arr = dr_update_dict[dr_param]
            if set_real:
                self.set_real(arr, dr_param)
            else:
                self.update_dr_param(arr, dr_param)

    def get_dr(self):
        model = self._env.physics.model
        if "ball_in_cup" in self.domain_name:
            dr_update_dict = {
                "cup_mass": model.body_mass[1],
                "ball_mass": model.body_mass[2],
                "cup_damping": model.dof_damping[0],
                "ball_damping": model.dof_damping[2],
                "actuator_gain": model.actuator_gainprm[0, 0],
                "cup_r": model.geom_rgba[1, 0],
                "cup_g": model.geom_rgba[1, 1],
                "cup_b": model.geom_rgba[1, 2],
                "ball_r": model.geom_rgba[6, 0],
                "ball_g": model.geom_rgba[6, 1],
                "ball_b": model.geom_rgba[6, 2],
                "ground_r": model.geom_rgba[0, 0],
                "ground_g": model.geom_rgba[0, 1],
                "ground_b": model.geom_rgba[0, 2],
            }
        elif "walker" in self.domain_name:
            dr_update_dict = {
                "torso_mass": model.body_mass[1],
                "right_thigh_mass": model.body_mass[2],
                "right_leg_mass": model.body_mass[3],
                "right_foot_mass": model.body_mass[4],
                "left_thigh_mass": model.body_mass[5],
                "left_leg_mass": model.body_mass[6],
                "left_foot_mass": model.body_mass[7],
                "right_hip": model.dof_damping[3],
                "right_knee": model.dof_damping[4],
                "right_ankle": model.dof_damping[5],
                "left_hip": model.dof_damping[6],
                "left_knee": model.dof_damping[7],
                "left_ankle": model.dof_damping[8],
                "ground_r": model.geom_rgba[0, 0],
                "ground_g": model.geom_rgba[0, 1],
                "ground_b": model.geom_rgba[0, 2],
                "body_r": model.geom_rgba[1, 0],
                "body_g": model.geom_rgba[1, 1],
                "body_b": model.geom_rgba[1, 2],
            }
        elif "cheetah" in self.domain_name:
            dr_update_dict = {
                "torso_mass": model.body_mass[1],
                "bthigh_mass": model.body_mass[2],
                "bshin_mass": model.body_mass[3],
                "bfoot_mass": model.body_mass[4],
                "fthigh_mass": model.body_mass[5],
                "fshin_mass": model.body_mass[6],
                "ffoot_mass": model.body_mass[7],
                "bthigh_damping": model.dof_damping[3],
                "bshin_damping": model.dof_damping[4],
                "bfoot_damping": model.dof_damping[5],
                "fthigh_damping": model.dof_damping[6],
                "fshin_damping": model.dof_damping[7],
                "ffoot_damping": model.dof_damping[8],
                "ground_r": model.geom_rgba[0, 0],
                "ground_g": model.geom_rgba[0, 1],
                "ground_b": model.geom_rgba[0, 2],
                "body_r": model.geom_rgba[1, 0],
                "body_g": model.geom_rgba[1, 1],
                "body_b": model.geom_rgba[1, 2],
            }
        elif "finger" in self.domain_name:
            dr_update_dict = {
                "proximal_mass": model.body_mass[0],
                "distal_mass": model.body_mass[1],
                "spinner_mass": model.body_mass[2],
                "proximal_damping": model.dof_damping[0],
                "distal_damping": model.dof_damping[1],
                "hinge_damping": model.dof_damping[2],
                "ground_r": model.geom_rgba[0, 0],
                "ground_g": model.geom_rgba[0, 1],
                "ground_b": model.geom_rgba[0, 2],
                "finger_r": model.geom_rgba[2, 0],
                "finger_g": model.geom_rgba[2, 1],
                "finger_b": model.geom_rgba[2, 2],
                "hotdog_r": model.geom_rgba[5, 0],
                "hotdog_g": model.geom_rgba[5, 1],
                "hotdog_b": model.geom_rgba[5, 2],
            }

        dr_list = []
        for dr_param in self.dr_list:
            dr_list.append(dr_update_dict[dr_param])
        arr = np.array(dr_list)

        arr = arr.astype(np.float32)
        return arr

class DR_Dummy(DR_Env):
    def __init__(self, env, cameras, **kwargs):
        self.square_size = 4
        self.speed_multiplier = 3
        self.square_r = 0.5
        self.square_g = 0.5
        self.square_b = 0.0
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render.modes': []}
        self.timestep = 1
        self._max_episode_steps = 200
        super().__init__(env, cameras, **kwargs)

    def get_state(self):
        return np.array([self.square_size, self.square_r, self.square_g, self.square_b, self.square_x, self.square_y])

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def get_dr(self):
        return np.array([self.square_size, self.speed_multiplier, self.square_r, self.square_g, self.square_b])

    def update_dr_param(self, param_name, eps=1e-3):
        if self.mean_only:
            mean = self.dr[param_name]
            if self.prop_range_scale:
                range = max(self.range_scale * mean, eps)
            else:
                range = self.range_scale
        else:
            mean, range = self.dr[param_name]
            range = max(range, eps)
        new_value = np.random.uniform(low=max(mean - range, eps), high=max(mean + range, 2 * eps))
        self.sim_params += [new_value]
        self.distribution_mean += [mean]
        self.distribution_range += [range]
        return new_value

    def apply_dr(self):
        self.sim_params = []
        self.distribution_mean = []
        self.distribution_range = []
        if self.dr is None or self.real_world:
            self.sim_params = np.zeros(self.dr_shape)
            self.distribution_mean = self.get_dr()
            self.distribution_range = np.zeros(self.dr_shape, dtype=np.float32)
            return
        self.square_size = self.update_dr_param('square_size')
        self.speed_multiplier = self.update_dr_param('speed_multiplier')
        self.square_r = self.update_dr_param('square_r')
        self.square_g = self.update_dr_param('square_g')
        self.square_b = self.update_dr_param('square_b')


    @property
    def observation_space(self):
        spaces = {}
        spaces['image'] = gym.spaces.Box(
            0, 255,  (3,) + self._size, dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)

    def env_step(self, action):
        x_update = action[0] * self.speed_multiplier
        y_update = action[0] * self.speed_multiplier
        self.square_x += x_update
        self.square_y += y_update
        self.square_x = max(self.square_x, 0)
        self.square_y = max(self.square_y, 0)
        self.square_x = min(self.square_x, 63)
        self.square_y = min(self.square_y, 63)
        obs = self.render()
        reward = - (np.abs(self.square_x) + np.abs(self.square_y))
        done = self.timestep >= self._max_episode_steps
        self.timestep += 1
        info = {}
        state = self.get_state()
        return obs, state, reward, done, info

    def env_reset(self):
        self.square_x = np.random.uniform(low=10., high=50.)
        self.square_y = np.random.uniform(low=10., high=50.)
        obs =  self.render()
        state = self.get_state()
        self.timestep = 1
        return state, obs

    def render(self, mode, size=None, *args, **kwargs):
        if size is None:
            size = self._size
        rgb_array = np.zeros((64, 64, 3))
        x_start = max(0, int(np.floor(self.square_x - self.square_size)))
        y_start = max(0, int(np.floor(self.square_y - self.square_size)))
        x_end = min(63, int(np.floor(self.square_x + self.square_size)))
        y_end = min(63, int(np.floor(self.square_y + self.square_size)))
        rgb_array[y_start:y_end, x_start:x_end] = np.clip(np.array([self.square_r, self.square_g, self.square_b]), 0, 1)
        if size is not None:
            rgb_array = cv2.resize(rgb_array, size)
        return np.transpose(rgb_array, (2, 0, 1))


def make(domain_name, task_name, seed, from_pixels, height, width, cameras=range(1),
         visualize_reward=False, frame_skip=None, mean_only=False,  dr_list=[], simple_randomization=False, dr_shape=None,
               real_world=False, dr=None, use_state="None", use_img=True,
                grayscale=False, delay_steps=0, range_scale=.1, prop_range_scale=False, state_concat=False,
         real_dr_params=None, prop_initial_range=False):
    # DMC
    if 'dmc' in domain_name:
        domain_name_root = domain_name[4:]  # Task name is formatted as dmc_walker.  Now just walker
        import dmc2gym
        # env = suite.load(domain_name_root, task_name, task_kwargs={'random': seed})
        env = dmc2gym.make(
            domain_name=domain_name_root,
            task_name=task_name,
            seed=seed,
            visualize_reward=visualize_reward,
            from_pixels=False,
            height=height,
            width=width,
            frame_skip=frame_skip
        )
        env = DR_DMCEnv(env, cameras=cameras, height=height, width=width, mean_only=mean_only,
                              dr_list=dr_list, simple_randomization=simple_randomization, dr_shape=dr_shape,
                              name=task_name, domain_name=domain_name_root,
                              real_world=real_world, dr=dr, use_state=use_state, use_img=use_img, grayscale=grayscale,
                              range_scale=range_scale, prop_range_scale=prop_range_scale, state_concat=state_concat,
                              real_dr_params=real_dr_params, prop_initial_range=prop_initial_range)  # TODO: apply these to all envs
        return env
    elif 'metaworld' in domain_name:
        if task_name + '-v1' in ed.ALL_V1_ENVIRONMENTS.keys():
            env_class = ed.ALL_V1_ENVIRONMENTS[task_name + '-v1']
        elif task_name + '-v2' in ed.ALL_V2_ENVIRONMENTS.keys():
            env_class = ed.ALL_V2_ENVIRONMENTS[task_name + '-v2']
        else:
            raise KeyError("Task name not found. " + str(task_name))

        env = env_class()

        task = random.choice(metaworld.ML1(task_name + '-v1').train_tasks)
        env.set_task(task)
        env.seed(seed)
        env = DR_MetaWorldEnv(env, cameras=cameras, height=height, width=width, mean_only=mean_only,
                   dr_list=dr_list, simple_randomization=simple_randomization, dr_shape=dr_shape, name=task_name,
                   real_world=real_world, dr=dr, use_state=use_state, use_img=use_img, grayscale=grayscale,
                   range_scale=range_scale, prop_initial_range=prop_initial_range)
        return env
    elif 'kitchen' in domain_name:
        env = Kitchen(dr=dr, mean_only=mean_only,
                          early_termination=False,
                          use_state=use_state,
                          real_world=real_world,
                          dr_list=dr_list,
                          task=task_name,
                          simple_randomization=False,
                          step_repeat=50,
                          control_version='mocap_ik',
                          step_size=0.01,
                          initial_randomization_steps=3,
                          minimal=False,
                          grayscale=grayscale,
                          time_limit=200,
                          delay_steps=delay_steps)
        env = DR_Kitchen(env, cameras=cameras, height=height, width=width, mean_only=mean_only,
                       dr_list=dr_list, simple_randomization=simple_randomization, dr_shape=dr_shape, name=task_name,
                       real_world=real_world, dr=dr, use_state=use_state, use_img=use_img, grayscale=grayscale,
                       range_scale=range_scale, prop_initial_range=prop_initial_range)
        return env
    elif 'dummy' in domain_name:
        inner_env = None
        env = DR_Dummy(inner_env, cameras=cameras, height=height, width=width, mean_only=mean_only,
                        dr_list=dr_list, simple_randomization=simple_randomization, dr_shape=dr_shape,
                        name=task_name,
                        real_world=real_world, dr=dr, use_state=use_state, use_img=use_img, grayscale=grayscale,
                        range_scale=range_scale, prop_range_scale=prop_range_scale, state_concat=state_concat,)
        return env
    else:
        raise KeyError("Domain name not found. " + str(domain_name))


def generate_shell_commands(domain_name, task_name, cameras, observation_type, encoder_type, work_dir,
                            pre_transform_image_size, agent, data_augs=None,
                            seed=-1, critic_lr=1e-3, actor_lr=1e-3,
                            eval_freq=10000, batch_size=128, num_train_steps=1000000, cuda=None,
                            save_tb=True, save_video=True, save_model=False):
    if cuda is not None:
        command = 'CUDA_VISIBLE_DEVICES=' + cuda + ' python train.py'
    else:
        command = 'python train.py'
    command += ' --domain_name ' + domain_name
    if task_name is not None:
        command += ' --task_name ' + task_name
    command += ' --cameras ' + cameras
    command += ' --observation_type ' + observation_type
    command += ' --encoder_type ' + encoder_type
    command += ' --work_dir ' + work_dir
    command += ' --pre_transform_image_size ' + pre_transform_image_size
    command += ' --image_size 84'
    command += ' --agent ' + agent
    if data_augs is not None:
        command += ' --data_augs' + data_augs
    command += ' --seed ' + str(seed)
    command += ' --critic_lr ' + str(critic_lr)
    command += ' --actor_lr ' + str(actor_lr)
    command += ' --eval_freq ' + str(eval_freq)
    command += ' --batch_size ' + str(batch_size)
    command += ' --num_train_steps ' + str(num_train_steps)
    if save_tb:
        command += ' --save_tb'
    if save_video:
        command += ' --save_video'
    if save_model:
        command += ' --save_model'


from abc import ABC
class EnvWrapper(gym.Env, ABC):
    def __init__(self, env, cameras, from_pixels=True, height=100, width=100, channels_first=True):
        camera_0 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 90}
        camera_1 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 135}
        camera_2 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 180}
        camera_3 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 225}
        camera_4 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 270}
        camera_5 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 315}
        camera_6 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 0}
        camera_7 = {'trackbodyid': -1, 'distance': 1.5, 'lookat': np.array((0.0, 0.6, 0)),
                    'elevation': -45.0, 'azimuth': 45}
        self.all_cameras = [camera_0, camera_1, camera_2, camera_3, camera_4, camera_5, camera_6, camera_7]

        self._env = env
        self.cameras = cameras
        self.from_pixels = from_pixels
        self.height = height
        self.width = width
        self.channels_first = channels_first

        self.special_reset = None
        self.special_reset_save = None
        self.hybrid_obs = False
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        shape = [3 * len(cameras), height, width] if channels_first else [height, width, 3 * len(cameras)]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

        self._state_obs = None
        self.change_camera()
        self.reset()

    def change_camera(self):
        return

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
        for key, value in self.all_cameras[camera_id].items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def set_hybrid_obs(self, mode):
        self.hybrid_obs = mode

    def _get_obs(self):
        if self.from_pixels:
            imgs = []
            for c in self.cameras:
                imgs.append(self.render(mode='rgb_array', camera_id=c))
            if self.channels_first:
                pixel_obs = np.concatenate(imgs, axis=0)
            else:
                pixel_obs = np.concatenate(imgs, axis=2)
            if self.hybrid_obs:
                return [pixel_obs, self._get_hybrid_state()]
            else:
                return pixel_obs
        else:
            return self._get_state_obs()

    def _get_state_obs(self):
        return self._state_obs

    def _get_hybrid_state(self):
        return self._state_obs

    @property
    def hybrid_state_shape(self):
        if self.hybrid_obs:
            return self._get_hybrid_state().shape
        else:
            return None

    def step(self, action):

        self._state_obs, reward, done, info = self._env.step(action)
        return self._get_obs(), reward, done, info

    def reset(self, save_special_steps=False):
        self._state_obs = self._env.reset()
        return self._get_obs()

    def set_state(self, qpos, qvel):
        self._env.set_state(qpos, qvel)

    @property
    def dt(self):
        if hasattr(self._env, 'dt'):
            return self._env.dt
        else:
            return 1

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
            if isinstance(self, GymEnvWrapper):
                self._env.unwrapped._render_callback()
            viewer = self._get_viewer(camera_id)
            # Calling render twice to fix Mujoco change of resolution bug.
            viewer.render(width, height, camera_id=-1)
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
            from mujoco_py import GlfwContext
            import mujoco_py
            GlfwContext(offscreen=True)
            self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        self.viewer_setup(camera_id)
        return self.viewer

    def get_body_com(self, body_name):
        return self._env.get_body_com(body_name)

    def state_vector(self):
        return self._env.state_vector


class MetaWorldEnvWrapper(EnvWrapper):
    def change_camera(self):
        for c in self.all_cameras:
            c['lookat'] = np.array((1.3, 0.75, 0.4))
            c['distance'] = 1.2
        # Zoomed out cameras
        camera_8 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                    'elevation': -45.0, 'azimuth': 135}
        camera_9 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                    'elevation': -45.0, 'azimuth': 225}
        # Gripper head camera
        camera_10 = {'trackbodyid': -1, 'distance': 0.2, 'lookat': np.array((1.3, 0.75, 0.4)),
                     'elevation': -90, 'azimuth': 0}
        self.all_cameras.append(camera_8)
        self.all_cameras.append(camera_9)
        self.all_cameras.append(camera_10)

    @property
    def _max_episode_steps(self):
        return self._env.max_path_length

    def step(self, action):
        self._state_obs, reward, done, info = self._env.step(action)
        if self._env.curr_path_length >= self._env.max_path_length:
            done = True
        return self._get_obs(), reward, done, info


class GymEnvWrapper(EnvWrapper):
    def change_camera(self):
        for c in self.all_cameras:
            c['lookat'] = np.array((1.3, 0.75, 0.4))
            c['distance'] = 1.2
        # Zoomed out cameras
        camera_8 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                    'elevation': -45.0, 'azimuth': 135}
        camera_9 = {'trackbodyid': -1, 'distance': 1.8, 'lookat': np.array((1.3, 0.75, 0.4)),
                    'elevation': -45.0, 'azimuth': 225}
        # Gripper head camera
        camera_10 = {'trackbodyid': -1, 'distance': 0.2, 'lookat': np.array((1.3, 0.75, 0.4)),
                     'elevation': -90, 'azimuth': 0}
        self.all_cameras.append(camera_8)
        self.all_cameras.append(camera_9)
        self.all_cameras.append(camera_10)

    def update_tracking_cameras(self):
        gripper_pos = self._state_obs['observation'][:3].copy()
        self.all_cameras[10]['lookat'] = gripper_pos

    def _get_obs(self):
        self.update_tracking_cameras()
        return super()._get_obs()

    @property
    def _max_episode_steps(self):
        return self._env._max_episode_steps

    def set_special_reset(self, mode):
        self.special_reset = mode

    def register_special_reset_move(self, action, reward):
        # self.update_tracking_cameras()
        # img = self.render(mode='rgb_array', camera_id=10)
        # import matplotlib.pyplot as plt
        # plt.imshow(img.transpose(1, 2, 0))
        # plt.show()
        if self.special_reset_save is not None:
            self.special_reset_save['obs'].append(self._get_obs())
            self.special_reset_save['act'].append(action)
            self.special_reset_save['reward'].append(reward)

    def go_to_pos(self, pos):
        grip_pos = self._state_obs['observation'][:3]
        action = np.zeros(4)
        for i in range(10):
            if np.linalg.norm(grip_pos - pos) < 0.02:
                break
            action[:3] = (pos - grip_pos) * 10
            self._state_obs, r, d, i = self._env.step(action)
            self.register_special_reset_move(action, r)
            grip_pos = self._state_obs['observation'][:3]

    def raise_gripper(self):
        grip_pos = self._state_obs['observation'][:3]
        raised_pos = grip_pos.copy()
        raised_pos[2] += 0.1
        self.go_to_pos(raised_pos)

    def open_gripper(self):
        action = np.array([0, 0, 0, 1])
        for i in range(2):
            self._state_obs, r, d, i = self._env.step(action)
            self.register_special_reset_move(action, r)

    def close_gripper(self):
        action = np.array([0, 0, 0, -1])
        for i in range(2):
            self._state_obs, r, d, i = self._env.step(action)
            self.register_special_reset_move(action, r)

    def reset(self, save_special_steps=False):
        self._state_obs = self._env.reset()
        if save_special_steps:
            self.special_reset_save = {'obs': [], 'act': [], 'reward': []}
            self.special_reset_save['obs'].append(self._get_obs())
        if self.special_reset == 'close' and self._env.has_object:
            obs = self._state_obs['observation']
            goal = self._state_obs['desired_goal']
            obj_pos = obs[3:6]
            goal_distance = np.linalg.norm(obj_pos - goal)
            desired_reset_pos = obj_pos + (obj_pos - goal) / goal_distance * 0.06
            desired_reset_pos_raised = desired_reset_pos.copy()
            desired_reset_pos_raised[2] += 0.1
            self.raise_gripper()
            self.go_to_pos(desired_reset_pos_raised)
            self.go_to_pos(desired_reset_pos)
        elif self.special_reset == 'grip' and self._env.has_object and not self._env.block_gripper:
            obs = self._state_obs['observation']
            obj_pos = obs[3:6]
            above_obj = obj_pos.copy()
            above_obj[2] += 0.1
            self.open_gripper()
            self.raise_gripper()
            self.go_to_pos(above_obj)
            self.go_to_pos(obj_pos)
            self.close_gripper()
        return self._get_obs()

    def _get_state_obs(self):
        obs = np.concatenate([self._state_obs['observation'],
                              self._state_obs['achieved_goal'],
                              self._state_obs['desired_goal']])
        return obs

    def _get_hybrid_state(self):
        grip_pos = self._env.sim.data.get_site_xpos('robot0:grip')
        dt = self._env.sim.nsubsteps * self._env.sim.model.opt.timestep
        grip_velp = self._env.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = gym.envs.robotics.utils.robot_get_obs(self._env.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        robot_info = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel])
        hybrid_obs_list = []
        if 'robot' in self.hybrid_obs:
            hybrid_obs_list.append(robot_info)
        if 'goal' in self.hybrid_obs:
            hybrid_obs_list.append(self._state_obs['desired_goal'])
        return np.concatenate(hybrid_obs_list)

    @property
    def observation_space(self):
        shape = self._get_state_obs().shape
        return gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype='float32')


class RealEnvWrapper(GymEnvWrapper):
    def render(self, mode='human', camera_id=0, height=None, width=None):
        if mode == 'human':
            self._env.render()

        if height is None:
            height = self.height
        if width is None:
            width = self.width

        if mode == 'rgb_array':
            data = self._env.render(mode='rgb_array', height=height, width=width)
            if self.channels_first:
                data = data.transpose((2, 0, 1))
            if camera_id == 8:
                data = data[3:]
            return data

    def _get_obs(self):
        return self.render(mode='rgb_array', height=self.height, width=self.width)

    def _get_state_obs(self):
        return self._get_obs()

    def reset(self, save_special_steps=False):
        self._state_obs = self._env.reset(rand_pos=True)
        return self._get_obs()
