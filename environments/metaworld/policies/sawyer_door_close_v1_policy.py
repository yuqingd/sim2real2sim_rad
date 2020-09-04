import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerDoorCloseV1Policy(Policy):
    
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

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = 1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_door = o_d['door_pos']
        pos_door += np.array([0.13, 0.1, 0.02])

        # if to the right of door handle///
        if pos_curr[0] > pos_door[0]:
            # if below door handle by more than 0.2
            if pos_curr[2] < pos_door[2] + 0.2:
                # rise above door handle by ~0.2
                return np.array([pos_curr[0], pos_curr[1], pos_door[2] + 0.25])
            else:
                # move toward door handle in XY plane
                return np.array([pos_door[0] - 0.02, pos_door[1], pos_curr[2]])
        # put end effector on the outer edge of door handle (still above it)
        elif abs(pos_curr[2] - pos_door[2]) > 0.04:
            return pos_door + np.array([-0.02, 0., 0.])
        # push from outer edge toward door handle's centroid
        else:
            return pos_door
