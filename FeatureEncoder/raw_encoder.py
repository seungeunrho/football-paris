import numpy as np

class FeatureEncoder:
    def __init__(self):
        self.active = -1
        self.player_pos_x, self.player_pos_y  = 0, 0
        
    def get_feature_dims(self):
        dims = {
            'player':17,
            'ball':18,
            'left_team':7,
            'left_team_closest':7,
            'right_team':7,
            'right_team_closest':7,
        }
        return dims


    def encode(self, obs):
        self.player_num = obs['active']
        player_pos_x, player_pos_y = obs['left_team'][self.player_num]
        player_direction = obs['left_team_direction'][self.player_num]
        player_role = obs['left_team_roles'][self.player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs['left_team_tired_factor'][self.player_num]
        is_dribbling = obs['sticky_actions'][9]
        is_sprinting = obs['sticky_actions'][8]

        player_state = np.concatenate((obs['left_team'][self.player_num], player_direction*100, 
                                       player_role_onehot, [player_tired, is_dribbling, is_sprinting]))

        ball_x, ball_y, ball_z = obs['ball']
        ball_x_relative = ball_x - self.player_pos_x
        ball_y_relative = ball_y - self.player_pos_y
        ball_z_relative = ball_z - 0.0
        ball_direction = obs['ball_direction']
        ball_speed = np.linalg.norm(ball_direction)
        ball_owned = 0.0 
        if obs['ball_owned_team'] == -1:
          ball_owned = 0.0
        else:
          ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs['ball_owned_team'] == 0:
          ball_owned_by_us = 1.0
        elif obs['ball_owned_team'] == 1:
          ball_owned_by_us = 0.0
        else:
          ball_owned_by_us = 0.0
        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y) 
        ball_state = np.concatenate((obs['ball'], 
                                     np.array(ball_which_zone),
                                     np.array([ball_x_relative, ball_y_relative, ball_z_relative]),
                                     ball_direction*5,
                                     np.array([ball_speed*5, ball_owned, ball_owned_by_us])))
    
        obs_left_team = np.delete(obs['left_team'], self.player_num, axis=0)
        obs_left_team_direction = np.delete(obs['left_team_direction'], self.player_num, axis=0)
#         left_team_relative = obs_left_team - obs['left_team'][self.player_num]
        left_team_relative = obs_left_team
#         left_team_distance = np.linalg.norm(left_team_relative, axis=1, keepdims=True)
        left_team_distance = np.linalg.norm(left_team_relative - obs['left_team'][self.player_num], axis=1, keepdims=True)
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_inner_product = np.sum(left_team_relative*obs_left_team_direction, axis=1, keepdims=True)
        left_team_cos = left_team_inner_product/(left_team_distance*(left_team_speed+1e-8))
        left_team_state = np.concatenate((left_team_relative*2, obs_left_team_direction*100, left_team_speed*100, \
                                          left_team_distance*2, left_team_cos), axis=1)
        left_closest_idx = np.argmin(left_team_distance)
        left_closest_state = left_team_state[left_closest_idx]
        
        
        obs_right_team = np.delete(obs['right_team'], self.player_num, axis=0)
        obs_right_team_direction = np.delete(obs['right_team_direction'], self.player_num, axis=0)
#         right_team_relative = obs_right_team - obs['left_team'][self.player_num]
        right_team_relative = obs_right_team
#         right_team_distance = np.linalg.norm(right_team_relative, axis=1, keepdims=True)
        right_team_distance = np.linalg.norm(right_team_relative - obs['left_team'][self.player_num], axis=1, keepdims=True)
        right_team_speed = np.linalg.norm(obs_right_team_direction, axis=1, keepdims=True)
        right_team_inner_product = np.sum(right_team_relative*obs_right_team_direction, axis=1, keepdims=True)
        right_team_cos = right_team_inner_product/(right_team_distance*(right_team_speed+1e-8))
        right_team_state = np.concatenate((right_team_relative*2, obs_right_team_direction*100, right_team_speed*100, \
                                           right_team_distance*2, right_team_cos), axis=1)
        right_closest_idx = np.argmin(right_team_distance)
        right_closest_state = right_team_state[right_closest_idx]
        
        

        state_dict = {"player": player_state,
                      "ball": ball_state,
                      "left_team" : left_team_state,
                      "left_closest" : left_closest_state,
                      "right_team" : right_team_state,
                      "right_closest" : right_closest_state}

        return state_dict
    
    def calc_additional_reward(self, prev_obs, obs):
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
        
        reward = ball_position_r * 0.01 + yellow_r
        
        return reward
    
        
    def _encode_ball_which_zone(self, ball_x, ball_y):
        if ball_x < -0.7 and (-0.3 < ball_y and ball_y < 0.3):
            return [1.0,0,0,0,0,0]
        elif ball_x < -0.2 and (-0.42 < ball_y and ball_y < 0.42):
            return [0,1.0,0,0,0,0]
        elif ball_x < 0.2 and (-0.42 < ball_y and ball_y < 0.42):
            return [0,0,1.0,0,0,0]
        elif ball_x > 0.7 and (-0.3 < ball_y and ball_y < 0.3):
            return [0,0,0,1.0,0,0]
        elif ball_x <=1.0 and (-0.42 < ball_y and ball_y < 0.42) :
            return [0,0,0,0,1.0,0]
        else:
            return [0,0,0,0,0,1.0]
        

    def _encode_role_onehot(self, role_num):
        result = [0,0,0,0,0,0,0,0,0,0]
        result[role_num] = 1.0
        return np.array(result)