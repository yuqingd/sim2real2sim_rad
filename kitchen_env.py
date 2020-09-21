import numpy as np
from environments.kitchen.adept_envs.adept_envs.kitchen_multitask_v0 import KitchenTaskRelaxV1
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control.rl.control import PhysicsError
import gym

GOAL_DIM = 30
ARM_DIM = 13
XPOS_INDICES = {
    'arm': [4, 5, 6, 7, 8, 9, 10], #Arm,
    'end_effector': [10],
    'gripper': [11, 12, 13, 14, 15, 16], #Gripper
    'knob_burner1': [22, 23],
    'knob_burner2': [24, 25],
    'knob_burner3': [26, 27],
    'knob_burner4': [28, 29],
    'light_switch': [31, 32],
    'slide': [38],
    'hinge': [41],
    'microwave': [44],
    'kettle': [47],
    'kettle_root': [48],
}

class Kitchen:
  def __init__(self, task='reach_kettle', size=(100, 100), real_world=False, dr=None, mean_only=False,
               early_termination=False, use_state="None", step_repeat=200, dr_list=[],
               step_size=0.05, simple_randomization=False, time_limit=200,
               control_version='mocap_ik', distance=2., azimuth=50, elevation=-40,
               initial_randomization_steps=1, minimal=False, dataset_step=None, grayscale=False):
    if 'rope' in task:
      distance = 1.5
      azimuth = 20
      elevation = -20
    if 'cabinet' in task:
      distance = 2.5
      azimuth = 120
      elevation = -40
    if 'open_microwave' in task:
      distance = 1.5
      azimuth = 140
      elevation = -30

    if minimal:
      global XPOS_INDICES
      XPOS_INDICES = {
        'arm': [4, 5, 6, 7, 8, 9, 10],  # Arm,
        'end_effector': [10],
        'gripper': [11, 12, 13, 14, 15, 16],  # Gripper
        'kettle': [20],
        'kettle_root': [21],

      }


    self._env = KitchenTaskRelaxV1(distance=distance, azimuth=azimuth, elevation=elevation, task_type=task, minimal=minimal)
    self.task = task
    self._size = size
    self.early_termination = early_termination
    self.mean_only = mean_only
    self.cur_step_fraction = 0
    self.real_world = real_world
    self.use_state = use_state
    self.dr = dr
    self.step_repeat = step_repeat
    self.step_size = step_size
    self.dr_list = dr_list
    if 'pick' in task or  'microwave' in task:
      self.use_gripper = True
    else:
      self.use_gripper = False
    self.end_effector_name = 'end_effector'
    self.mocap_index = 3
    self.end_effector_index = self._env.sim.model._site_name2id['end_effector']
    if 'rope' in task:
      self.cylinder_index = 5
      self.box_with_hole_index = 6

    self.arm_njnts = 7
    self.simple_randomization = simple_randomization
    self.control_version = control_version
    self.initial_randomization_steps = initial_randomization_steps
    if 'push_kettle' in task:
      self.initial_randomization_steps += 3

    self.has_kettle = False if 'open_microwave' in task else True
    self.has_microwave = False if minimal else True
    self.has_cabinet = False if minimal else True
    self.dataset_step = dataset_step
    self.grayscale = grayscale
    self._max_episode_steps = time_limit
    self.reward_range = (-float('inf'), float('inf'))
    # TODO: maybe update this.  Currently just placeholder values.
    self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 80}

    self.setup_task()

  def setup_task(self):
    self.timesteps = 0
    init_xpos = self._env.sim.data.body_xpos
    #randomize kettle location
    if 'push' in self.task or 'slide' in self.task:
      kettle_loc = init_xpos[XPOS_INDICES['kettle']]
      kettle_loc[:,:2] += np.random.normal(0, 5, (2,))
      kettle_loc[:,:2] = np.clip(kettle_loc[:,:2], [-0.5, -0.1], [0.5, 1.0])
      self._env.sim.model.body_pos[XPOS_INDICES['kettle']] = kettle_loc
      self._env.sim.forward()

    elif 'rope' in self.task:
      body_id = self._env.sim.model.body_name2id('boxes_with_hole')
      box_loc = self._env.sim.model.body_pos[body_id]
      box_loc[2] += np.random.normal(0, .05) #move box height only
      self._env.sim.model.body_pos[body_id] = box_loc
      self._env.sim.forward()

    if 'reach' in self.task:
      self.set_workspace_bounds('full_workspace')

      if self.task == 'reach_microwave':
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['microwave']])
      elif self.task == 'reach_slide':
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['slide']])
      elif self.task == 'reach_kettle':
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal[-1] += 0.1  # goal in middle of kettle
      else:
        raise NotImplementedError

    elif 'push' in self.task:
      self.set_workspace_bounds('stove_area')

      if self.task == 'push_kettle_burner': #single goal test, push to back burner
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal[1] += 0.5
      else:
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal += np.random.normal(loc=0, scale=0.3) #randomly select goal location in workspace
        self.goal[1] += 0.4 # move forward in y
        self.goal = np.clip(self.goal, self.end_effector_bound_low, self.end_effector_bound_high)
        self.goal[-1] = np.squeeze(init_xpos[XPOS_INDICES['kettle']])[-1] #set z pos to be same as kettle, since we only want to push in x,y

    elif 'slide' in self.task:
      self.set_workspace_bounds('front_stove_area')
      self.slide_d1 = None

      #decrease friction for sliding
      self._env.sim.model.geom_friction[220:225, :] = 0.002
      self._env.sim.model.geom_friction[97:104, :] = 0.002

      if self.task == 'slide_kettle_burner': #single goal test, slide to back burner
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal[1] += .5
      else:
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal += np.random.normal(loc=0, scale=0.2) #randomly select goal location in workspace
        self.goal[1] += 0.4  # move forward in y
        self.goal = np.clip(self.goal, [-.5, 0.45, 0], [.5, 1, 0])
        self.goal[-1] = np.squeeze(init_xpos[XPOS_INDICES['kettle']])[-1]  # set z pos to be same as kettle, since we only want to push in x,y

    elif 'pick' in self.task:
      self.set_workspace_bounds('full_workspace')
      self.use_gripper = True
      self.orig_kettle_height = np.squeeze(init_xpos[XPOS_INDICES['kettle']])[-1]
      self.pick_d1 = None

      if self.task == 'pick_kettle_burner': #single goal test, slide to back burner
        self.goal = np.squeeze(init_xpos[XPOS_INDICES['kettle']])
        self.goal[1] += 0.5
      else:
        self.goal = np.random.uniform(low=[-1, 0, 0], high=[0, 1, 0]) #randomly select goal location in workspace OUTSIDE of end effector reach
        self.goal[-1] = np.squeeze(init_xpos[XPOS_INDICES['kettle']])[-1] #set z pos to be same as kettle, since we only want to slide in x,y

    elif 'rope' in self.task:
      self.set_workspace_bounds('no_restrictions')
      self.goal = self._env.sim.data.site_xpos[self.box_with_hole_index]

    elif 'open_microwave' in self.task:
      self.set_workspace_bounds('full_workspace')
      self.goal = np.squeeze(init_xpos[XPOS_INDICES['microwave']]).copy()
      self.goal += 0.5
    elif 'open_cabinet' in self.task:
      self.set_workspace_bounds('full_workspace')
      img_orig = self.render(size=(512, 512)).copy()

      goal = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door']].copy()
      self.goal = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door']].copy()
      self.goal[0] = 0.18
      self.step(np.array([0, 0, 0]))
      end_effector = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['end_effector']].copy()
      ratio_to_goal = 0.6
      partway = ratio_to_goal * goal + (1 - ratio_to_goal) * end_effector
      for i in range(60):
        diff = partway - self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['end_effector']].copy()
        diff = diff / self.step_size
        self.step(diff)

    else:
      raise NotImplementedError(self.task)


    self.randomize_start()

  def randomize_start(self):
    # Randomize start position
    x = np.random.uniform(-1, 1)
    y = 0
    z = np.random.uniform(0, 1)
    direction = np.array([x, y, z])

    for i in range(self.initial_randomization_steps):
      self.step(direction)


  def get_reward(self):
    xpos = self._env.sim.data.body_xpos
    if 'reach' in self.task:
      end_effector = np.squeeze(xpos[XPOS_INDICES['end_effector']])
      reward = -np.linalg.norm(end_effector - self.goal)
      done = np.abs(reward) < 0.25
    elif 'push' in self.task:
      end_effector = np.squeeze(xpos[XPOS_INDICES['end_effector']])
        # two stage reward, first get to kettle, then kettle to goal
      kettle = np.squeeze(xpos[XPOS_INDICES['kettle']])
      kettlehandle = kettle.copy()
      #kettlehandle[-1] += 0.1  # goal in middle of kettle

      d1 = np.linalg.norm(end_effector - kettlehandle)
      d2 = np.linalg.norm(kettle - self.goal)
      done = np.abs(d2) < 0.25

      reward = -(d1 + d2)

    elif 'slide' in self.task:
      end_effector = np.squeeze(xpos[XPOS_INDICES['end_effector']])
        # two stage reward, first get to kettle, then kettle to goal
      kettle = np.squeeze(xpos[XPOS_INDICES['kettle']])
      kettlehandle = kettle.copy()
      #kettlehandle[-1] += 0.  # goal in middle of kettle

      d1 = np.linalg.norm(end_effector - kettlehandle)
      if d1 < 0.35 and self.slide_d1 is None: #TODO: tune threshold for hitting kettle
        self.slide_d1 = d1


      d2 = np.linalg.norm(kettle - self.goal)
      done = np.abs(d2) < 0.2

      if self.slide_d1 is not None:
        reward = -(self.slide_d1 + d2)
      else:
        reward = -(d1 + d2)



    elif 'pick' in self.task:
      #three stage reward, first reach kettle, pick up kettle, then goal
      end_effector = np.squeeze(xpos[XPOS_INDICES['end_effector']])

      kettle = np.squeeze(xpos[XPOS_INDICES['kettle']])
      kettlehandle = kettle.copy()
      kettlehandle[-1] += 0.15  # goal in middle of kettle
      kettle_height = kettle[-1]

      d1 = np.linalg.norm(end_effector - kettlehandle) #reach handle

      if d1 < 0.1 and self.pick_d1 is None: #TODO: tune this
        d1 = d1 - self._env.data.ctrl[self.arm_njnts] #TODO: scale gripper contribution to reward
        self.pick_d1 = d1

      d3 = np.linalg.norm(kettle[:2] - self.goal[:2]) #xy distance to goal

      if self.pick_d1 is not None:
        d2 = np.linalg.norm(self.orig_kettle_height - kettle_height) #distance kettle has been lifted
        if d2 > 0.02: #TODO: tune this
          #then we have lifted it
          d2 = -d2
        else:
          if d3 > 0.25: #TODO: tune this
          #then we haven't lifted it and it is far from goal, restart
            self.pick_d1 = None
            d2 = 0
      else:
        d2 = 0
      done = np.abs(d3) < 0.25

      reward = -(d1 + d2 + d3)

    elif 'rope' in self.task:
      cylinder_loc = self._env.sim.data.site_xpos[self.cylinder_index]
      reward = -np.linalg.norm(cylinder_loc - self.goal)
      done = np.abs(reward) < 0.1

    elif 'open_microwave' in self.task:
      end_effector = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['end_effector']]
      microwave_pos = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['microwave_door']]
      microwave_pos_top = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['microwave_door_top']]
      microwave_pos_bottom = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['microwave_door_bottom']]

      # Always give a reward for having the end-effector near the shelf
      # since multiple z-positions are valid we'll compute each dimension separately
      x_dist = abs(end_effector[0] - microwave_pos[0])
      y_dist = abs(end_effector[1] - microwave_pos[1])
      if end_effector[2] > microwave_pos_top[2]:
        z_dist = end_effector[2] - microwave_pos_top[2]
      elif end_effector[2] < microwave_pos_bottom[2]:
        z_dist = microwave_pos_bottom[2] - end_effector[2]
      else:
        z_dist = 0
      dist_to_handle = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
      reach_rew = -dist_to_handle

      # Also have a reward for moving the cabinet
      dist_to_goal = np.abs(microwave_pos[1] - 0.28)
      move_rew = -dist_to_goal
      reward = reach_rew + move_rew

      done = dist_to_goal < 0.05

    elif 'open_cabinet' in self.task:
      end_effector = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['end_effector']]
      cabinet_pos = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door']]
      cabinet_pos_top = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door_top']]
      cabinet_pos_bottom = self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door_bottom']]


      # Always give a reward for having the end-effector near the shelf
      # since multiple z-positions are valid we'll compute each dimension separately
      x_dist = abs(end_effector[0] - cabinet_pos[0])
      y_dist = abs(end_effector[1] - cabinet_pos[1])
      if end_effector[2] > cabinet_pos_top[2]:
        z_dist = end_effector[2] - cabinet_pos_top[2]
      elif end_effector[2] < cabinet_pos_bottom[2]:
        z_dist = cabinet_pos_bottom[2] - end_effector[2]
      else:
        z_dist = 0
      dist_to_handle = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
      reach_rew = -dist_to_handle

      # Also have a reward for moving the cabinet
      dist_to_goal = np.abs(cabinet_pos[0] - 0.18)
      move_rew = -dist_to_goal
      reward = reach_rew + move_rew

      done = dist_to_goal < 0.05

    else:
      raise NotImplementedError

    self.timesteps += 1
    done = done or (self.timesteps >= self._max_episode_steps)
    return reward, done

  def get_sim(self):
    return self._env.sim

  def set_workspace_bounds(self, bounds):
    if bounds == 'no_restrictions':
      x_low = y_low = z_low = -float('inf')
      x_high = y_high = z_high = float('inf')
    elif bounds == 'full_workspace':
      x_low = -1.5  # Around the microwave
      x_high = 1.  # Around the sink
      y_low = -0.1  # Right in front of the robot's pedestal
      y_high = 2  # Past back burner
      z_low = 1.5  # Tabletop
      z_high = 5  # Cabinet height
    elif bounds == 'stove_area':
      x_low = -0.5  # Left edge of stove
      x_high = 0.5  # Right edge of stove
      y_low = -0.1  # Right in front of the robot's pedestal
      y_high = 1.0  # Back burner
      z_low = 1.5  # Tabletop
      z_high = 2.  # Around top of kettle
    elif bounds == 'front_stove_area':  # For use with sliding
      x_low = -0.5  # Left edge of stove
      x_high = 0.5  # Right edge of stove
      y_low = -0.1  # Right in front of the robot's pedestal
      y_high = 0.4  # Mid-front burner
      z_low = 1.5  # Tabletop
      z_high = 2.  # Around top of kettle
    else:
      raise NotImplementedError("No other bounds types")

    self.end_effector_bound_low = [x_low, y_low, z_low]
    self.end_effector_bound_high = [x_high, y_high, z_high]


  @property
  def action_space(self):
    if self.control_version == 'dmc_ik':
      act_shape = 4 if self.use_gripper else 3  # 1 for fingers, 3 for end effector position
      return gym.spaces.Box(np.array([-100.0] * act_shape), np.array([100.0] * act_shape))
    elif self.control_version == 'mocap_ik':
      act_shape = 4 if self.use_gripper else 3  # 1 for fingers, 3 for end effector position
      return gym.spaces.Box(np.array([-1.0] * act_shape), np.array([1.0] * act_shape))
    else:
      return self._env.action_space


  def set_xyz_action(self, action):

    pos_delta = action * self.step_size
    new_mocap_pos = self._env.data.mocap_pos + pos_delta[None]#self._env.sim.data.site_xpos[self.end_effector_index].copy() + pos_delta[None]
    # new_mocap_pos = self._env.data.mocap_pos + pos_delta[None]

    new_mocap_pos[0, :] = np.clip(
      new_mocap_pos[0, :],
      self.end_effector_bound_low,
      self.end_effector_bound_high,
    )


    self._env.data.set_mocap_pos('mocap', new_mocap_pos)
    if 'open_microwave' in self.task or 'open_cabinet' in self.task:
      self._env.data.set_mocap_quat('mocap', np.array([0.93937271,  0., 0., -0.34289781]))

  def set_gripper(self, action):
    #gripper either open or close
    cur_ac = self._env.data.qpos[self.arm_njnts] #current gripper position, either 0 or 0.85

    if action < 0:
      gripper_ac = 0 #open
    else:
      gripper_ac = 0.85 #close

    sequence = np.linspace(cur_ac, gripper_ac, num=50)
    for step in sequence:
      self._env.data.ctrl[self.arm_njnts] = step
      self._env.sim.step()
    #need to linearly space out control with multiple steps so simulator doesnt break


  def step(self, action):
    update = None
    if self.control_version == 'mocap_ik':
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.set_xyz_action(action[:3])

        if self.use_gripper:
          gripper_ac = action[-1]
          self.set_gripper(gripper_ac)


    elif self.control_version == 'dmc_ik':
      action = np.clip(action, self.action_space.low, self.action_space.high)
      xyz_pos = action[:3] * self.step_size + self._env.sim.data.site_xpos[self.end_effector_index]

      physics = self._env.sim
      # The joints which can be manipulated to move the end-effector to the desired spot.
      joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
      ikresult = qpos_from_site_pose(physics, self.end_effector_name, target_pos=xyz_pos, joint_names=joint_names, tol=1e-10, progress_thresh=10, max_steps=50)
      qpos = ikresult.qpos
      success = ikresult.success

      if success:
        action_dim = len(self._env.data.ctrl)
        qpos_low = self._env.model.jnt_range[:, 0]
        qpos_high = self._env.model.jnt_range[:, 1]
        update = np.clip(qpos[:action_dim], qpos_low[:action_dim], qpos_high[:action_dim])
        if self.use_gripper:
          # TODO: almost certainly not the right way to implement this
          gripper_pos = action[3:]
          update[-len(gripper_pos):] = gripper_pos
          raise NotImplementedError
        else:
          update[self.arm_njnts + 1:] = 0 #no gripper movement
        self._env.data.ctrl[:] = update

    elif self.control_version == 'position':
      update = np.clip(action, self.action_space.low, self.action_space.high)
    else:
      raise ValueError(self.control_version)

    if update is not None:
      self._env.data.ctrl[:] = update
    for _ in range(self.step_repeat):
        try:
            self._env.sim.step()
        except PhysicsError as e:
            success = False
            print("Physics error:", e)

    reward, done = self.get_reward()
    info = {'discount': 1.0}
    state = self.goal
    if self.use_state is not "None":
        robot_state = self.get_state()
        state = np.concatenate([state, robot_state])
    return state, reward, done, info


  def get_state(self):
    state_list = [np.squeeze(self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['end_effector']])]
    if self.use_gripper:
      state_list.append(np.squeeze(self._env.data.qpos[self.arm_njnts]))
    if self.use_state == 'all':
      if 'rope' in self.task:
        state_list.append(np.squeeze(self._env.sim.data.site_xpos[self.cylinder_index]))
      elif 'open_microwave' in self.task:
        state_list.append(np.squeeze(self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['microwave_door']]))
      elif 'open_cabinet' in self.task:
        state_list.append(np.squeeze(self._env.sim.data.site_xpos[self._env.sim.model._site_name2id['cabinet_door']]))
      elif 'kettle' in self.task:
        state_list.append(np.squeeze(self._env.sim.data.body_xpos[XPOS_INDICES['kettle']]))
      else:
        raise NotImplementedError("Unrecognized task" + self.task)
    return np.concatenate(state_list)

  def reset(self):
    state_obs = self._env.reset()
    self.setup_task()

    if 'open_microwave' in self.task or 'open_cabinet' in self.task:
      self._env.data.set_mocap_quat('mocap', np.array([0.93937271, 0., 0., -0.34289781]))
      # Make the end-effector horizontal
      for _ in range(2000):
        self._env.sim.step()

    return state_obs

  def render(self, size=None, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    if size is None:
        size = self._size
    h, w = size
    img = self._env.render(width=w, height=h, mode='rgb_array')
    if self.grayscale:
      img = img[:, :, 0] * 0.2989 + img[:, :, 1] * 0.5870 * img[:, :, 2] + 0.1140
      img = np.expand_dims(img, 2)
    return np.transpose(img, (2, 0, 1))

  @property
  def observation_space(self):
    spaces = {}

    if self.use_state is not "None":
      state_shape = 4 if self.use_gripper else 3  # 2 for fingers, 3 for end effector position
      state_shape = self.goal.shape + state_shape
      if self.use_state == 'all':
        state_shape += 3
      state_space = gym.spaces.Box(np.array([-float('inf')] * state_shape), np.array([-float('inf')] * state_shape))
    else:
      state_space= gym.spaces.Box(np.array([-float('inf')] * self.goal.shape[0]),
                                       np.array([float('inf')] * self.goal.shape[0]))
    return state_space