import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerDoorOpenV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'door_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = 1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_door = o_d['door_pos']
        pos_door[0] -= 0.05

        # align end effector's Z axis with door handle's Z axis
        if np.linalg.norm(pos_curr[:2] - pos_door[:2]) > 0.08:
            return pos_door + np.array([0., 0.075, 0.2])
        # drop down on front edge of door handle
        elif abs(pos_curr[2] - pos_door[2]) > 0.04:
            return pos_door + np.array([0., 0.075, 0.])
        # push from front edge toward door handle's centroid
        else:
            return pos_door
