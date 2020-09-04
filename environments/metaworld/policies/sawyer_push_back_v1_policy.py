import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPushBackV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'puck_pos': obs[3:6],
            'goal_pos': obs[9:],
            'unused_info': obs[6:9],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos']

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.05:
            return pos_puck + np.array([0., 0., 0.15])
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck + np.array([0., 0., 0.03])
        # Move to the goal
        else:
            return o_d['goal_pos'] + np.array([.0, .0, .05])

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos']

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.04 or abs(pos_curr[2] - pos_puck[2]) > 0.12:
            return 0.
        # While end effector is moving down toward the puck, begin closing the grabber
        else:
            return 0.6
