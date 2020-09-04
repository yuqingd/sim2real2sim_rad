import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPushWallV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'obj_pos': obs[3:6],
            'goal_pos': obs[9:],
            'unused_info': obs[6:9],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self.desired_pos(o_d), p=10.)
        action['grab_effort'] = self.grab_effort(o_d)

        return action.array

    @staticmethod
    def desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_obj = o_d['obj_pos']

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_obj[:2]) > 0.02:
            return pos_obj + np.array([0., 0., 0.2])
        # Once XY error is low enough, drop end effector down on top of obj
        elif abs(pos_curr[2] - pos_obj[2]) > 0.04:
            return pos_obj + np.array([0., 0., 0.03])
        # Move to the goal
        else:
            #if the wall is between the puck and the goal, go around the wall
            if(-0.1 <= pos_obj[0] <= 0.3 and 0.65 <= pos_obj[1] <= 0.75):
                return pos_curr + np.array([-1, 0, 0])
            elif ((-0.15 < pos_obj[0] < 0.05 or 0.15 < pos_obj[0] < 0.35)
                    and 0.695 <= pos_obj[1] <= 0.755):
                return pos_curr + np.array([0, 1, 0])
            return o_d['goal_pos']

    @staticmethod
    def grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_obj = o_d['obj_pos']

        if np.linalg.norm(pos_curr[:2] - pos_obj[:2]) > 0.02 or \
                          abs(pos_curr[2] - pos_obj[2]) > 0.1:
            return 0.0
        # While end effector is moving down toward the obj, begin closing the grabber
        else:
            return 0.6
