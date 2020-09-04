from dm_control import suite
from metaworld.envs.mujoco import env_dict as ed
import numpy as np
from mw_wrapper import MetaWorldEnv


class DR_MetaWorldEnv:
  def __init__(self, env, cameras, height=64, width=64, mean_only=False, early_termination=False, dr_list=[], simple_randomization=False, dr_shape=None, outer_loop_version=0,
               real_world=False, dr=None, use_state="None", use_img=True, use_depth=False,
               dataset_step=None, grayscale=False):

    self._env = MetaWorldEnv(env, from_pixels=use_img, cameras=cameras, height=height, width=width)

    self._size = (height, width)

    self.mean_only = mean_only
    self.early_termination = early_termination
    self.dr_list = dr_list
    self.simple_randomization = simple_randomization
    self.dr_shape = dr_shape
    self.outer_loop_version = outer_loop_version
    self.real_world = real_world
    self.use_state = use_state
    self.use_img = use_img
    self.dr = dr
    self.use_depth = use_depth
    self.dataset_step = dataset_step
    self.grayscale = grayscale

    self.apply_dr()

  def set_dr(self, dr):
    self.dr = dr

  def set_dataset_step(self, step):
    self.dataset_step = step

  def update_dr_param(self, param, param_name, eps=1e-3, indices=None):
    if param_name in self.dr:
      if self.mean_only:
        mean = self.dr[param_name]
        range = max(0.1 * mean, eps) #TODO: tune this?
      else:
        mean, range = self.dr[param_name]
        range = max(range, eps)
      new_value = np.random.uniform(low=max(mean - range, eps), high=max(mean + range, 2 * eps))
      if indices is None:
        param[:] = new_value
      else:
        try:
          for i in indices:
            param[i:i+1] = new_value
        except:
          param[indices:indices+1] = new_value

      self.sim_params += [mean]
      self.distribution_mean += [mean]
      self.distribution_range += [range]

  def apply_dr(self):
    self.sim_params = []
    self.distribution_mean = []
    self.distribution_range = []
    if self.dr is None or self.real_world:
      self.sim_params = self.get_dr()
      self.distribution_mean = self.get_dr()
      self.distribution_range = np.zeros(self.dr_shape, dtype=np.float32)
      return

    model = self._env.sim.model
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

      for dr_param in self.dr_list:
        arr, indices = dr_update_dict[dr_param]
        self.update_dr_param(arr, dr_param, indices=indices)

    else:
      raise NotImplementedError


  @property
  def observation_space(self):
    spaces = {}
    if self.use_state is not "None":
      if self.use_state == 'all':
        spaces['state'] = self._env.observation_space
      else:
        spaces['state'] = 3
    spaces['image'] = gym.spaces.Box(
      0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):

    obs, reward, done, info = self._env.step(action)
    obs_dict = {}

    if self.use_img:
      obs_dict['image'] = obs
    else:
      obs_dict['state'] = obs
    info['discount'] = 1.0
    obs_dict['real_world'] = 1.0 if self.real_world else 0.0
    obs_dict['sim_params'] = np.array(self.sim_params, dtype=np.float32)
    if not (self.dr is None) and not self.real_world:
      obs_dict['dr_params'] = self.get_dr()
    obs_dict['success'] = 1.0 if info['success'] else 0.0
    obs_dict['distribution_mean'] = np.array(self.distribution_mean, dtype=np.float32)
    obs_dict['distribution_range'] = np.array(self.distribution_range, dtype=np.float32)

    if not self.early_termination:
      done = False

    return obs_dict, reward, done, info

  def get_dr(self):
    if self.simple_randomization:
      if 'stick-pull' or 'stick-push' in self.name:
        cylinder_body = self._env.sim.model.body_name2id('cylinder')
        return np.array([self._env.sim.model.body_mass[cylinder_body]])
      elif 'basketball' in self.name:
        microwave_index = self._env.sim.model.body_name2id('microdoorroot')
        return np.array([self._env.sim.model.body_mass[microwave_index]])
      else:
        raise NotImplementedError

    model = self._env.sim.model
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
      model = self._env.sim.model
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

    arr = arr.astype(np.float32)
    return arr

  def reset(self):
    obs = self._env.reset()
    self.apply_dr()

    obs_dict = {}

    if self.use_img:
      obs_dict['image'] = obs
    else:
      obs_dict['state'] = obs

    obs_dict['real_world'] = 1.0 if self.real_world else 0.0
    if not (self.dr is None) and not self.real_world:
      obs_dict['dr_params'] = self.get_dr()
    obs_dict['success'] = 0.0
    obs_dict['sim_params'] = np.array(self.sim_params, dtype=np.float32)
    obs_dict['distribution_mean'] = np.array(self.distribution_mean, dtype=np.float32)
    obs_dict['distribution_range'] = np.array(self.distribution_range, dtype=np.float32)
    return obs_dict

  def get_state(self, state_obs):
    if self.use_state == 'robot':
      return state_obs[:3]   # Only include robot state (endeffector pos)
    elif self.use_state == 'all':
      return state_obs
    else:
      raise NotImplementedError(self.use_state)

  def render(self, size=None, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")

    height, width = self._size
    return self._env.render(mode='rgb_array', width=width, height=height)


def make(domain_name, task_name, seed, from_pixels, height, width, cameras=range(1),
         visualize_reward=False, frame_skip=None, mean_only=False, early_termination=False, dr_list=[], simple_randomization=False, dr_shape=None, outer_loop_version=0,
               real_world=False, dr=None, use_state="None", use_img=True, use_depth=False,
               dataset_step=None, grayscale=False):
    if domain_name in suite._DOMAINS:
        import dmc2gym
        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=seed,
            visualize_reward=visualize_reward,
            from_pixels=from_pixels,
            height=height,
            width=width,
            frame_skip=frame_skip
        )
        return env
    elif domain_name in ed.ALL_V1_ENVIRONMENTS.keys():
        env_class = ed.ALL_V1_ENVIRONMENTS[domain_name]
    elif domain_name in ed.ALL_V2_ENVIRONMENTS.keys():
        env_class = ed.ALL_V2_ENVIRONMENTS[domain_name]
    else:
        raise KeyError("Domain name not found.")

    if task_name is not None:
        env = env_class(task_type=task_name)
    else:
        env = env_class()

    env = DR_MetaWorldEnv(env, from_pixels=from_pixels, cameras=cameras, height=height, width=width, mean_only=mean_only, early_termination=early_termination,
               dr_list=dr_list, simple_randomization=simple_randomization, dr_shape=dr_shape, outer_loop_version=outer_loop_version,
               real_world=real_world, dr=dr, use_state=use_state, use_img=use_img, use_depth=use_depth, dataset_step=dataset_step, grayscale=grayscale)
    env.seed(seed)
    return env


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
