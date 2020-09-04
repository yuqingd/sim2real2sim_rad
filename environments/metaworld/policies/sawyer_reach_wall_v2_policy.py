import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move


class SawyerReachWallV2Policy(Policy):

    @staticmethod
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

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=5.)
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_hand = o_d['hand_pos']
        pos_goal = o_d['goal_pos']
        # if the hand is going to run into the wall, go up while still moving
        # towards the goal position.
        if(-0.1 <= pos_hand[0] <= 0.3 and
                0.60 <= pos_hand[1] <= 0.80 and
                pos_hand[2] < 0.2):
            return pos_goal + np.array([0., 0., 1.])
        return pos_goal
