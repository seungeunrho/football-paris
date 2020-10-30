
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
import numpy as np


class PPO(nn.Module):
    def __init__(self, arg_dict, device=None):
        super(PPO, self).__init__()
        self.device=None
        if device:
            self.device = device

        self.fc_player = nn.Linear(arg_dict["feature_dims"]["player"],64)  
        self.fc_ball = nn.Linear(arg_dict["feature_dims"]["ball"],64)
        self.fc_left = nn.Linear(arg_dict["feature_dims"]["left_team"],32)
        self.fc_right  = nn.Linear(arg_dict["feature_dims"]["right_team"],32)
        self.fc_left_closest = nn.Linear(arg_dict["feature_dims"]["left_team_closest"],32)
        self.fc_right_closest = nn.Linear(arg_dict["feature_dims"]["right_team_closest"],32)
        
        self.conv1d_left = nn.Conv1d(32, 32, 1, stride=1)
        self.conv1d_right = nn.Conv1d(32, 32, 1, stride=1)
        self.fc_left2 = nn.Linear(32*10,96)
        self.fc_right2 = nn.Linear(32*10,96)
        self.fc_cat = nn.Linear(96+96+64+64+32+32,arg_dict["lstm_size"])
        
        self.norm_player = nn.LayerNorm(64)
        self.norm_ball = nn.LayerNorm(64)
        self.norm_left = nn.LayerNorm(32)
        self.norm_left2 = nn.LayerNorm(96)
        self.norm_left_closest = nn.LayerNorm(32)
        self.norm_right = nn.LayerNorm(32)
        self.norm_right2 = nn.LayerNorm(96)
        self.norm_right_closest = nn.LayerNorm(32)
        self.norm_cat = nn.LayerNorm(arg_dict["lstm_size"])
        
        self.lstm  = nn.LSTM(arg_dict["lstm_size"],arg_dict["lstm_size"])

        self.fc_pi_a1 = nn.Linear(arg_dict["lstm_size"], 128)
        self.fc_pi_a2 = nn.Linear(128, 12)
        self.norm_pi_a1 = nn.LayerNorm(128)
        
        self.fc_pi_m1 = nn.Linear(arg_dict["lstm_size"], 128)
        self.fc_pi_m2 = nn.Linear(128, 8)
        self.norm_pi_m1 = nn.LayerNorm(128)

        self.fc_v1 = nn.Linear(arg_dict["lstm_size"], 128)
        self.norm_v1 = nn.LayerNorm(128)
        self.fc_v2 = nn.Linear(128, 1,  bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1,None))
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["learning_rate"])

        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        self.lmbda = arg_dict["lmbda"]
        self.eps_clip = 0.1
        self.entropy_coef = arg_dict["entropy_coef"]
        
    def forward(self, state_dict):
        player_state = state_dict["player"]          
        ball_state = state_dict["ball"]              
        left_team_state = state_dict["left_team"]
        left_closest_state = state_dict["left_closest"]
        right_team_state = state_dict["right_team"]  
        right_closest_state = state_dict["right_closest"]
        avail = state_dict["avail"]
        
        player_embed = self.norm_player(self.fc_player(player_state))
        ball_embed = self.norm_ball(self.fc_ball(ball_state))
        left_team_embed = self.norm_left(self.fc_left(left_team_state))  # horizon, batch, n, dim
        left_closest_embed = self.norm_left_closest(self.fc_left_closest(left_closest_state))
        right_team_embed = self.norm_right(self.fc_right(right_team_state))
        right_closest_embed = self.norm_right_closest(self.fc_right_closest(right_closest_state))
        
        [horizon, batch_size, n_player, dim] = left_team_embed.size()
        left_team_embed = left_team_embed.view(horizon*batch_size, n_player, dim).permute(0,2,1)         # horizon * batch, dim1, n
        left_team_embed = F.relu(self.conv1d_left(left_team_embed)).permute(0,2,1)                       # horizon * batch, n, dim2
        left_team_embed = left_team_embed.reshape(horizon*batch_size, -1).view(horizon,batch_size,-1)    # horizon, batch, n * dim2
        left_team_embed = F.relu(self.norm_left2(self.fc_left2(left_team_embed)))
        
        right_team_embed = right_team_embed.view(horizon*batch_size, n_player, dim).permute(0,2,1)  # horizon * batch, dim1, n
        right_team_embed = F.relu(self.conv1d_right(right_team_embed)).permute(0,2,1)  # horizon * batch, n * dim2
        right_team_embed = right_team_embed.reshape(horizon*batch_size, -1).view(horizon,batch_size,-1)
        right_team_embed = F.relu(self.norm_right2(self.fc_right2(right_team_embed)))
        
        cat = torch.cat([player_embed, ball_embed, left_team_embed, right_team_embed, left_closest_embed, right_closest_embed], 2)
        cat = F.relu(self.norm_cat(self.fc_cat(cat)))
        h_in = state_dict["hidden"]
        out, h_out = self.lstm(cat, h_in)
        
        a_out = F.relu(self.norm_pi_a1(self.fc_pi_a1(out)))
        a_out = self.fc_pi_a2(a_out)
        logit = a_out + (avail-1)*1e7
        prob = F.softmax(logit, dim=2)
        
        prob_m = F.relu(self.norm_pi_m1(self.fc_pi_m1(out)))
        prob_m = self.fc_pi_m2(prob_m)
        prob_m = F.softmax(prob_m, dim=2)

        v = F.relu(self.norm_v1(self.fc_v1(out)))
        v = self.fc_v2(v)

        return prob, prob_m, v, h_out

    def make_batch(self, data):
        # data = [tr1, tr2, ..., tr10] * batch_size
        s_player_batch, s_ball_batch, s_left_batch, s_left_closest_batch, s_right_batch, s_right_closest_batch, avail_batch =  [],[],[],[],[],[],[]
        s_player_prime_batch, s_ball_prime_batch, s_left_prime_batch, s_left_closest_prime_batch, \
                                                  s_right_prime_batch, s_right_closest_prime_batch, avail_prime_batch =  [],[],[],[],[],[],[]
        h1_in_batch, h2_in_batch, h1_out_batch, h2_out_batch = [], [], [], []
        a_batch, m_batch, r_batch, prob_batch, done_batch, need_move_batch = [], [], [], [], [], []
        
        for rollout in data:
            s_player_lst, s_ball_lst, s_left_lst, s_left_closest_lst, s_right_lst, s_right_closest_lst, avail_lst =  [], [], [], [], [], [], []
            s_player_prime_lst, s_ball_prime_lst, s_left_prime_lst, s_left_closest_prime_lst, \
                                                  s_right_prime_lst, s_right_closest_prime_lst, avail_prime_lst =  [], [], [], [], [], [], []
            h1_in_lst, h2_in_lst, h1_out_lst, h2_out_lst = [], [], [], []
            a_lst, m_lst, r_lst, prob_lst, done_lst, need_move_lst = [], [], [], [], [], []
            
            for transition in rollout:
                s, a, m, r, s_prime, prob, done, need_move = transition

                s_player_lst.append(s["player"])
                s_ball_lst.append(s["ball"])
                s_left_lst.append(s["left_team"])
                s_left_closest_lst.append(s["left_closest"])
                s_right_lst.append(s["right_team"])
                s_right_closest_lst.append(s["right_closest"])
                avail_lst.append(s["avail"])
                h1_in, h2_in = s["hidden"]
                h1_in_lst.append(h1_in)
                h2_in_lst.append(h2_in)
                
                s_player_prime_lst.append(s_prime["player"])
                s_ball_prime_lst.append(s_prime["ball"])
                s_left_prime_lst.append(s_prime["left_team"])
                s_left_closest_prime_lst.append(s_prime["left_closest"])
                s_right_prime_lst.append(s_prime["right_team"])
                s_right_closest_prime_lst.append(s_prime["right_closest"])
                avail_prime_lst.append(s_prime["avail"])
                h1_out, h2_out = s_prime["hidden"]
                h1_out_lst.append(h1_out)
                h2_out_lst.append(h2_out)

                a_lst.append([a])
                m_lst.append([m])
                r_lst.append([r])
                prob_lst.append([prob])
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                need_move_lst.append([need_move]),
                
            s_player_batch.append(s_player_lst)
            s_ball_batch.append(s_ball_lst)
            s_left_batch.append(s_left_lst)
            s_left_closest_batch.append(s_left_closest_lst)
            s_right_batch.append(s_right_lst)
            s_right_closest_batch.append(s_right_closest_lst)
            avail_batch.append(avail_lst)
            h1_in_batch.append(h1_in_lst[0])
            h2_in_batch.append(h2_in_lst[0])
            
            s_player_prime_batch.append(s_player_prime_lst)
            s_ball_prime_batch.append(s_ball_prime_lst)
            s_left_prime_batch.append(s_left_prime_lst)
            s_left_closest_prime_batch.append(s_left_closest_prime_lst)
            s_right_prime_batch.append(s_right_prime_lst)
            s_right_closest_prime_batch.append(s_right_closest_prime_lst)
            avail_prime_batch.append(avail_prime_lst)
            h1_out_batch.append(h1_out_lst[0])
            h2_out_batch.append(h2_out_lst[0])

            a_batch.append(a_lst)
            m_batch.append(m_lst)
            r_batch.append(r_lst)
            prob_batch.append(prob_lst)
            done_batch.append(done_lst)
            need_move_batch.append(need_move_lst)
        

        s = {
          "player": torch.tensor(s_player_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "ball": torch.tensor(s_ball_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "left_team": torch.tensor(s_left_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "left_closest": torch.tensor(s_left_closest_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "right_team": torch.tensor(s_right_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "right_closest": torch.tensor(s_right_closest_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "avail": torch.tensor(avail_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "hidden" : (torch.tensor(h1_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2), 
                      torch.tensor(h2_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2))
        }

        s_prime = {
          "player": torch.tensor(s_player_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "ball": torch.tensor(s_ball_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "left_team": torch.tensor(s_left_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "left_closest": torch.tensor(s_left_closest_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "right_team": torch.tensor(s_right_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "right_closest": torch.tensor(s_right_closest_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "avail": torch.tensor(avail_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "hidden" : (torch.tensor(h1_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2), 
                      torch.tensor(h2_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2))
        }

        a,m,r,done_mask,prob,need_move = torch.tensor(a_batch, device=self.device).permute(1,0,2), \
                                         torch.tensor(m_batch, device=self.device).permute(1,0,2), \
                                         torch.tensor(r_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                         torch.tensor(done_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                         torch.tensor(prob_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                         torch.tensor(need_move_batch, dtype=torch.float, device=self.device).permute(1,0,2)

        return s, a, m, r, s_prime, done_mask, prob, need_move
    

    def train_net(self, data):
        tot_loss_lst = []
        pi_loss_lst = []
        entropy_lst = []
        v_loss_lst = []
        for i in range(self.K_epoch):
            for mini_batch in data:
                s, a, m, r, s_prime, done_mask, prob, need_move = mini_batch
                pi, pi_move, v, _ = self.forward(s)
                pi_prime, pi_m_prime, v_prime, _ = self.forward(s_prime)

                td_target = r + self.gamma * v_prime * done_mask
                delta = td_target - v                           # [horizon * batch_size * 1]
                delta = delta.detach().cpu().numpy()

                advantage_lst = []
                advantage = np.array([0])
                for delta_t in delta[::-1]:
                    advantage = self.gamma * self.lmbda * advantage + delta_t           
                    advantage_lst.append(advantage)
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float, device=self.device)

                pi_a = pi.gather(2,a)
                pi_m = pi_move.gather(2,m)
                pi_am = pi_a - pi_a*need_move*(1-pi_m)
                ratio = torch.exp(torch.log(pi_am) - torch.log(prob))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                entropy = -torch.sum(pi*torch.log(pi+ 1e-7), dim=2, keepdim=True)

                surr_loss = -torch.min(surr1, surr2)
                v_loss = F.smooth_l1_loss(v, td_target.detach())
                entropy_loss = -1*self.entropy_coef*entropy
                loss = surr_loss + v_loss + entropy_loss
#                 loss = surr_loss + v_loss
                loss = loss.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                tot_loss_lst.append(loss.item())
                pi_loss_lst.append(surr_loss.mean().item())
                v_loss_lst.append(v_loss.item())
                entropy_lst.append(entropy.mean().item())
                
        return np.mean(tot_loss_lst), np.mean(pi_loss_lst), np.mean(v_loss_lst), np.mean(entropy_lst) 




class FeatureEncoder:
    def __init__(self):
        self.active = -1
        self.player_pos_x, self.player_pos_y  = 0, 0
        
    def get_feature_dims(self):
        dims = {
            'player':27,
            'ball':18,
            'left_team':7,
            'left_team_closest':7,
            'right_team':7,
            'right_team_closest':7,
        }
        return dims

    def encode(self, obs):
        avail = self._get_avail(obs)
        player_num = obs['active']
        
        player_pos_x, player_pos_y = obs['left_team'][player_num]
        player_direction = np.array(obs['left_team_direction'][player_num])*100
        player_role = obs['left_team_roles'][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs['left_team_tired_factor'][player_num]
        is_dribbling = obs['sticky_actions'][9]
        is_sprinting = obs['sticky_actions'][8]

        
        player_state = np.concatenate((avail[2:], obs['left_team'][player_num], player_direction, 
                                       player_role_onehot, [player_tired, is_dribbling, is_sprinting]))

        ball_x, ball_y, ball_z = obs['ball']
        ball_x_relative = ball_x - self.player_pos_x
        ball_y_relative = ball_y - self.player_pos_y
        ball_z_relative = ball_z - 0.0
        ball_direction = np.array(obs['ball_direction'])*5
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
                                     ball_direction,
                                     np.array([ball_speed*5, ball_owned, ball_owned_by_us])))
    
        obs_left_team = np.delete(obs['left_team'], player_num, axis=0)
        obs_left_team_direction = np.delete(obs['left_team_direction'], player_num, axis=0)
#         left_team_relative = obs_left_team - obs['left_team'][player_num]
        left_team_relative = obs_left_team
#         left_team_distance = np.linalg.norm(left_team_relative, axis=1, keepdims=True)
        left_team_distance = np.linalg.norm(left_team_relative - obs['left_team'][player_num], axis=1, keepdims=True)
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_inner_product = np.sum(left_team_relative*obs_left_team_direction, axis=1, keepdims=True)
        left_team_cos = left_team_inner_product/(left_team_distance*(left_team_speed+1e-8))
        left_team_state = np.concatenate((left_team_relative*2, obs_left_team_direction*100, left_team_speed*100, \
                                          left_team_distance*2, left_team_cos), axis=1)
        left_closest_idx = np.argmin(left_team_distance)
        left_closest_state = left_team_state[left_closest_idx]
        
        
        obs_right_team = np.delete(obs['right_team'], player_num, axis=0)
        obs_right_team_direction = np.delete(obs['right_team_direction'], player_num, axis=0)
#         right_team_relative = obs_right_team - obs['left_team'][player_num]
        right_team_relative = obs_right_team
#         right_team_distance = np.linalg.norm(right_team_relative, axis=1, keepdims=True)
        right_team_distance = np.linalg.norm(right_team_relative - obs['left_team'][player_num], axis=1, keepdims=True)
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
                      "right_closest" : right_closest_state,
                      "avail" : avail}

        return state_dict
    
    def _get_avail(self, obs):
        avail = [1,1,1,1,1,1,1,1,1,1,1,1]
        NO_OP, MOVE, LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SPRINT, RELEASE_MOVE, \
                                                      RELEASE_SPRINT, SLIDE, DRIBBLE, RELEASE_DRIBBLE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

        # When opponents owning ball ...
        if obs['ball_owned_team'] == 1: # opponents owning ball
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS], avail[SHOT], avail[DRIBBLE] = 0, 0, 0, 0, 0
        elif obs['ball_owned_team'] == -1: # GR ball 
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS], avail[SHOT], avail[DRIBBLE] = 0, 0, 0, 0, 0
        else:
            avail[SLIDE] = 0
            
        # Dealing with sticky actions
        sticky_actions = obs['sticky_actions']
        if sticky_actions[8] == 1:  # sprinting
            avail[SPRINT] = 0
        else:
            avail[RELEASE_SPRINT] = 0
            
        if sticky_actions[9] == 1:  # dribbling
            avail[DRIBBLE], avail[SLIDE] = 0, 0
        else:
            avail[RELEASE_DRIBBLE] = 0
            
        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0
            
        
        # if too far, no shot
        ball_x, ball_y, _ = obs['ball']
        if ball_x < 0.6:
            avail[SHOT] = 0
            
        if obs['ball_owned_team'] == 0:  # our team 
            if obs['game_mode'] == 2:  # GoalKick
                avail[SPRINT], avail[DRIBBLE] = 0, 0
            elif obs['game_mode'] == 3:  # FreeKick
                avail[DRIBBLE] = 0
            elif obs['game_mode'] == 4:  # Corner
                avail[SHOT], avail[SPRINT], avail[DRIBBLE] = 0, 0, 0
            elif obs['game_mode'] == 5:  #ThrowIn
                avail[SHOT], avail[SPRINT], avail[DRIBBLE] = 0, 0, 0
            elif obs['game_mode'] == 6:  # Penalty
                avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS], avail[DRIBBLE] = 0, 0, 0, 0
            
        return np.array(avail)
        
    def _encode_ball_which_zone(self, ball_x, ball_y):
        if (-1.0<=ball_x and ball_x<-0.7) and (-0.3<ball_y and ball_y<0.3):
            return [1.0,0,0,0,0,0]
        elif (-1.0<=ball_x and ball_x<-0.2) and (-0.42<ball_y and ball_y<0.42):
            return [0,1.0,0,0,0,0]
        elif (0.7<ball_x and ball_x<=1.0) and (-0.3<ball_y and ball_y<0.3):
            return [0,0,1.0,0,0,0]
        elif (0.2<ball_x and ball_x<=1.0) and (-0.42<ball_y and ball_y<0.42) :
            return [0,0,0,1.0,0,0]
        elif (-0.2<=ball_x and ball_x<=0.2) and (-0.42<ball_y and ball_y<0.42) :
            return [0,0,0,0,1.0,0]
        else:
            return [0,0,0,0,0,1.0]
        

    def _encode_role_onehot(self, role_num):
        result = [0,0,0,0,0,0,0,0,0,0]
        result[role_num] = 1.0
        return np.array(result)

    
    


fe = FeatureEncoder()

arg_dict = {
    "lstm_size" : 196,
    "learning_rate" : 0.0002,
    "gamma" : 0.992,
    "lmbda" : 0.96,
    "entropy_coef" : 0.0,
    "trained_model_dir" : "/kaggle_simulations/agent/model_3202560.tar",
    "k_epoch" : 3,

}
arg_dict["feature_dims"] = fe.get_feature_dims()
model = PPO(arg_dict)
cpu_device = torch.device('cpu')
checkpoint = torch.load(arg_dict["trained_model_dir"], map_location=cpu_device)
model.load_state_dict(checkpoint['model_state_dict'])

h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
         torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
steps = 0

def state_to_tensor(state_dict, h_in):
    player_state = torch.from_numpy(state_dict["player"]).float().unsqueeze(0).unsqueeze(0)
    ball_state = torch.from_numpy(state_dict["ball"]).float().unsqueeze(0).unsqueeze(0)
    left_team_state = torch.from_numpy(state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0)
    left_closest_state = torch.from_numpy(state_dict["left_closest"]).float().unsqueeze(0).unsqueeze(0)
    right_team_state = torch.from_numpy(state_dict["right_team"]).float().unsqueeze(0).unsqueeze(0)
    right_closest_state = torch.from_numpy(state_dict["right_closest"]).float().unsqueeze(0).unsqueeze(0)
    avail = torch.from_numpy(state_dict["avail"]).float().unsqueeze(0).unsqueeze(0)

    state_dict_tensor = {
      "player" : player_state,
      "ball" : ball_state,
      "left_team" : left_team_state,
      "left_closest" : left_closest_state,
      "right_team" : right_team_state,
      "right_closest" : right_closest_state,
      "avail" : avail,
      "hidden" : h_in
    }
    return state_dict_tensor

def agent(obs):
    global model
    global fe
    global h_out
    global steps
    
    steps +=1
    
    if steps%100==0:
        print(steps)
    obs = obs['players_raw'][0]
    h_in = h_out
    state_dict = fe.encode(obs)
    state_dict_tensor = state_to_tensor(state_dict, h_in)
    a_prob, m_prob, _, h_out = model(state_dict_tensor)
    a = Categorical(a_prob).sample().item()
    if a==0:
        real_action = a
    elif a==1:
        m = Categorical(m_prob).sample().item()
        real_action = m + 1
    else:
        real_action = a + 7

    return [real_action]