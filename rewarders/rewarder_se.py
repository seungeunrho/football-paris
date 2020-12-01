import numpy as np

def calc_reward(rew, prev_obs, obs):
    ball_x, ball_y, ball_z = obs['ball']
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42

    ball_position_r = 0.0
    if   (-END_X <= ball_x    and ball_x < -PENALTY_X)and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
        ball_position_r = -2.0
    elif (-END_X <= ball_x    and ball_x < -MIDDLE_X) and (-END_Y < ball_y     and ball_y < END_Y):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y     and ball_y < END_Y):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x  and ball_x <=END_X)     and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x   and ball_x <=END_X)     and (-END_Y < ball_y     and ball_y < END_Y):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    left_yellow = np.sum(obs["left_team_yellow_card"]) -  np.sum(prev_obs["left_team_yellow_card"])
    right_yellow = np.sum(obs["right_team_yellow_card"]) -  np.sum(prev_obs["right_team_yellow_card"])
    yellow_r = right_yellow - left_yellow

    
    win_reward = 0.0
    if obs['steps_left'] == 0:
        [my_score, opponent_score] = obs['score']
        if my_score > opponent_score:
            win_reward = 1.0
            
    reward = 5.0*win_reward + 5.0*rew + 0.003*ball_position_r + yellow_r
        

    return reward