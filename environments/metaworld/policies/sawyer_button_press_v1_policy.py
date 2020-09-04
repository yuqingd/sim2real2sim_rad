import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move


class SawyerButtonPressV1Policy(Policy):

    @staticmethod
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'button_start_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self.desired_pos(o_d), p=4.)
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_start_pos'] + np.array([0., 0., -0.07])
        
        # align the gripper with the button if the gripper does not have
        # the same x and z position as the button.
        hand_x, hand_y, hand_z = pos_curr
        button_initial_x, button_initial_y, button_initial_z = pos_button
        if not np.all(np.isclose(np.array([hand_x, hand_z]),
                                np.array([button_initial_x, button_initial_z]),
                                atol=0.02)):
            pos_button[1] = pos_curr[1] - .1
            return pos_button
        # if the hand is aligned with the button, push the button in, by
        # increasing the hand's y position
        pos_button[1] += 0.02

        return pos_button
