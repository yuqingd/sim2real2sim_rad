import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerReachV2Policy(Policy):

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

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=o_d['goal_pos'], p=5.)
        action['grab_effort'] = 0.

        return action.array
