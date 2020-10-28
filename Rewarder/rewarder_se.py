import numpy as np

# def calc_reward(rew, prev_obs, obs):
#     ball_x, ball_y, ball_z = obs['ball']

#     ball_position_r = 0.0
#     if obs['ball_owned_team'] == 1:
#         if (-1.0<=ball_x and ball_x<-0.7) and (-0.3<ball_y and ball_y<0.3):
#             ball_position_r = -2.0
#         elif (-1.0<=ball_x and ball_x<-0.2) and (-0.42<ball_y and ball_y<0.42):
#             ball_position_r = -1.0    
#     elif obs['ball_owned_team'] == 0:
#         if (0.7<ball_x and ball_x<=1.0) and (-0.3<ball_y and ball_y<0.3):
#             ball_position_r = 2.0
#         elif (0.2<ball_x and ball_x<=1.0) and (-0.42<ball_y and ball_y<0.42) :
#             ball_position_r = 1.0

#     left_yellow = np.sum(obs["left_team_yellow_card"]) -  np.sum(prev_obs["left_team_yellow_card"])
#     right_yellow = np.sum(obs["right_team_yellow_card"]) -  np.sum(prev_obs["right_team_yellow_card"])
#     yellow_r = right_yellow - left_yellow
    
#     win_reward = 0.0
#     if obs['steps_left'] == 0:
#         [my_score, opponent_score] = obs['score']
#         if my_score > opponent_score:
#             win_reward = 1.0

#     reward = 5.0*win_reward + 5.0*rew + 0.005*ball_position_r + yellow_r

#     return reward


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

    left_yellow = np.sum(obs["left_team_yellow_card"]) -  np.sum(prev_obs["left_team_yellow_card"])
    right_yellow = np.sum(obs["right_team_yellow_card"]) -  np.sum(prev_obs["right_team_yellow_card"])
    yellow_r = right_yellow - left_yellow

    
    win_reward = 0.0
    if obs['steps_left'] == 0:
        [my_score, opponent_score] = obs['score']
        if my_score > opponent_score:
            win_reward = 1.0
            
    reward = 5.0*win_reward + 5.0*rew + 0.005*ball_position_r + yellow_r
        

    return reward