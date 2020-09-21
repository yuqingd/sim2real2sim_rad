import pybullet as p
import pybullet_data as pd

import os
import gym
from gym import error, spaces, utils
import time
import numpy as np
import math



class XArm6Env(gym.Env):
    def __init__(self, useFixedBase=True, flags=p.URDF_INITIALIZE_SAT_FEATURES, robot_state = False):
        p.connect(p.DIRECT)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])
        self.robot_state = robot_state



        obs_shape = 5 # 2 for fingers, 3 for end effector position
        act_shape = 4 # 1 for fingers, 3 for end effector position
        self.img_shape = (64, 64)
        self.action_space = spaces.Box(np.array([-1] * act_shape), np.array([1] * act_shape))
        self.observation_space = self.get_observation_space(self.img_shape, obs_shape, robot_state)

        urdfRootPath = pd.getDataPath()
        xarm_path = os.path.dirname(os.path.realpath(__file__))

        self.planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        self.xarmUid = p.loadURDF(os.path.join(xarm_path, "xarm_description/urdf/xarm7_with_gripper.urdf"), useFixedBase=True)

        self.num_joints = len(p.getNumJoints(self.xarmUid)) #15
        self.end_effector_id = 8 #gripper fix
        self.finger1_id = 10 #left finger joint
        self.finger2_id = 13 #right finger joint
        self.num_dofs = 7

        self.rest_pos = [0] * self.num_joints #TODO: Update with actual reset positions
        table_pos = [0, 0, -0.625]

        self.tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=table_pos)


        #load in objects if needed

    def get_observation_space(self, img_shape, state_shape, robot_state):
        spaces = {}
        if robot_state:
            spaces['state'] = spaces.Box(np.array([-1] * state_shape), np.array([1] * state_shape))
        spaces['image'] = spaces.Box(0, 255, img_shape + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    def reset(self):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        p.resetSimulation()
        p.setGravity(0,0,-10)

        #Reset xarm joints
        for i in range(self.num_joints):
            p.resetJointState(self.xarmUid, i, self.rest_pos[i])

        obs = self.render()
        if self.robot_state:
            state_robot = p.getLinkState(self.xarmUid, self.end_effector_id)[0]
            state_fingers = (p.getJointState(self.xarmUid, self.finger1_id)[0], p.getJointState(self.xarmUid, self.finger2_id)[0])

            obs = obs + state_robot + state_fingers
        return obs


    def step(self, action): #action
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]
        orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])

        currentPose = p.getLinkState(self.xarmUid, self.end_effector_id)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.xarmUid, self.end_effector_id, newPosition, orientation)

        p.setJointMotorControlArray(self.xarmUid, list(range(self.num_dofs)) + [self.finger1_id, self.finger2_id], p.POSITION_CONTROL,
                                    list(jointPoses) + 2 * [fingers])
        p.stepSimulation()

        state_robot = p.getLinkState(self.xarmUid, self.end_effector_id)[0]
        state_fingers = (
        p.getJointState(self.xarmUid, self.finger1_id)[0], p.getJointState(self.xarmUid, self.finger2_id)[0])

        #TODO: implement task + reward function
        reward = 0
        done = False

        obs = self.render()
        if self.robot_state:
            state_robot = p.getLinkState(self.xarmUid, self.end_effector_id)[0]
            state_fingers = (
            p.getJointState(self.xarmUid, self.finger1_id)[0], p.getJointState(self.xarmUid, self.finger2_id)[0])

            obs = obs + state_robot + state_fingers
        return obs, reward, done, {}

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(self.img_shape[0]) /self.img_shape[1],
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=self.img_shape[0],
                                              height=self.img_shape[1],
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, self.img_shape + (4, ))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()
