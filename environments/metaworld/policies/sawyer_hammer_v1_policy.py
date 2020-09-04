import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerHammerV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'hammer_pos': obs[3:6],
            'unused_info': obs[6:],
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
        pos_puck = o_d['hammer_pos'] + np.array([-.08, .0, -.01])
        pos_goal = np.array([0.24, 0.71, 0.11]) + np.array([-.19, .0, .05])

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.04:
            return pos_puck + np.array([0., 0., 0.1])
        # Once XY error is low enough, drop end effector down on top of hammer
        elif abs(pos_curr[2] - pos_puck[2]) > 0.05 and pos_puck[-1] < 0.03:
            return pos_puck + np.array([0., 0., 0.03])
        # If not at the same X pos as the peg, move over to that plane
        elif np.linalg.norm(pos_curr[[0,2]] - pos_goal[[0,2]]) > 0.02:
            return np.array([pos_goal[0], pos_curr[1], pos_goal[2]])
        # Move to the peg
        else:
            return pos_goal

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['hammer_pos'] + np.array([-.08, .0, -.01])

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.04 or abs(pos_curr[2] - pos_puck[2]) > 0.1:
            return 0.
        # While end effector is moving down toward the hammer, begin closing the grabber
        else:
            return 0.8
