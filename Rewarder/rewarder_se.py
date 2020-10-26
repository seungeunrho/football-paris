import numpy as np

def calc_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs['ball']

    ball_position_r = 0.0
    if ball_x < -0.7 and (-0.3 < ball_y and ball_y < 0.3):
        ball_position_r = -2.0
    elif ball_x < -0.2 and (-0.42 < ball_y and ball_y < 0.42):
        ball_position_r = -1.0
    elif ball_x < 0.2 and (-0.42 < ball_y and ball_y < 0.42):
        ball_position_r = 0.0
    elif ball_x > 0.7 and (-0.3 < ball_y and ball_y < 0.3):
        ball_position_r = 2.0
    elif ball_x <=1.0 and (-0.42 < ball_y and ball_y < 0.42) :
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    # ball position 1 * 3000 * 0.02 = 60

#         print("x", ball_x, "y", ball_y, "r", ball_position_r)

    left_yellow = np.sum(obs["left_team_yellow_card"]) -  np.sum(prev_obs["left_team_yellow_card"])
    right_yellow = np.sum(obs["right_team_yellow_card"]) -  np.sum(prev_obs["right_team_yellow_card"])
    yellow_r = right_yellow - left_yellow

    reward = 5.0*rew + ball_position_r * 0.01 + yellow_r

    return reward