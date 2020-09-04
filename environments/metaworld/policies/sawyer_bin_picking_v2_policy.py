import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerBinPickingV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'cube_pos': obs[3:6],
            'extra_info': obs[6:]
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_cube = o_d['cube_pos'] + np.array([.0, .0, .03])
        pos_bin = np.array([.12, .7, .02])

        if np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > .02:
            return pos_cube + np.array([.0, .0, .15])
        elif abs(pos_curr[2] - pos_cube[2]) > .01:
            return pos_cube
        elif pos_curr[2] < 0.15:
            return pos_curr + np.array([.0, .0, .1])
        elif np.linalg.norm(pos_curr[:2] - pos_bin[:2]) > .02:
            return np.array([*pos_bin[:2], .18])
        else:
            return pos_bin

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_cube = o_d['cube_pos'] + np.array([.0, .0, .03])

        if (np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.02
            or abs(pos_curr[2] - pos_cube[2]) > 0.1):
            return -1.
        else:
            return .6
