#TODO: Deal with normalization
GOAL_DIM = 30
ARM_DIM = 13
ELEMENT_INDICES_LL = [
    [0,1,2,3,4,5,6,7,8,9,10,11,12], #Arm
    [13,14],#Burners
    [15,16],#Burners
    [17,18],#Burners
    [19,20], #Burners
    [21,22], #lightswitch
    [23], #Slide
    [24,25], #Hinge
    [26], #Microwave
    [27,28,29,30,31,32,33] #Kettle
]

ELEMENT_INDICES_HL = [
    [13,14],#Bottom Burners
    [17,18],#Top Burners
    [21,22], #lightswitch
    [23], #Slide
    [24,25], #Hinge
    [26], #Microwave
    [27,28,29,30,31,32,33] #Kettle
]

# For light task
BONUS_THRESH_LL = 0.3
BONUS_THRESH_HL = 0.3

def elementwise_sparse_reward_ll(goal, obs, next_obs, reward,
                             bonus_thresh=0.2, bonus_weight=10.,
                             height_thresh=-0.16, height_penalty=-5,
                             relative_goal=True, **kwargs):
    """Reward based on negative distance to goal."""
    reward = 0.
    for element_idx in ELEMENT_INDICES_LL:
        distance = np.linalg.norm(next_obs[..., element_idx] - goal[element_idx])
        sparse_reward = np.asarray(distance < BONUS_THRESH_LL)
        reward += sparse_reward
    return reward

def elementwise_sparse_reward_hl(goal, obs, next_obs, reward,
                             bonus_thresh=0.2, bonus_weight=10.,
                             height_thresh=-0.16, height_penalty=-5,
                             relative_goal=True, **kwargs):
    """Reward based on negative distance to goal."""
    reward = 0.
    for element_idx in ELEMENT_INDICES_HL:
        distance = np.linalg.norm(next_obs[..., element_idx] - goal[element_idx])
        sparse_reward = np.asarray(distance < BONUS_THRESH_HL)
        reward += sparse_reward
    return reward


def sparse_reward(goal, obs, next_obs, reward,
                             bonus_thresh=0.2, bonus_weight=10.,
                             height_thresh=-0.16, height_penalty=-5,
                             relative_goal=True, **kwargs):
    """Reward based on negative distance to goal."""
    goal_dim = np.shape(goal)[-1]
    distance = np.linalg.norm(next_obs[..., :goal_dim] - goal)
    sparse_reward = np.asarray(distance < BONUS_THRESH_LL)
    return sparse_reward


def sparse_reward_highlevel(goal, obs, next_obs, reward, bonus_weight=10.,
                             height_thresh=-0.16, height_penalty=-5,
                             relative_goal=True, **kwargs):
    """Reward based on negative distance to goal."""
    distance = np.linalg.norm(next_obs[..., ARM_DIM:GOAL_DIM] - goal[ARM_DIM:GOAL_DIM])
    sparse_reward = np.asarray(distance < BONUS_THRESH_HL)
    return sparse_reward
