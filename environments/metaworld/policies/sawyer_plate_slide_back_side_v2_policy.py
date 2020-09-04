import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPlateSlideBackSideV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'puck_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_xyz(o_d), p=10.)
        action['grab_effort'] = 1.

        return action.array

    @staticmethod
    def _desired_xyz(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos'] + np.array([.023, .0, .025])

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.01:
            return pos_puck + np.array([.0, .0, .07])
        elif abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck
        else:
            return np.array([pos_curr[0] + .1, .6, pos_curr[2]])
