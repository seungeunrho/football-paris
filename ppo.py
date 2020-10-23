import time
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self, lstm_size, device=None):
        super(PPO, self).__init__()
        if device:
            self.device = device

        self.fc_player = nn.Linear(17,64)
        self.fc_ball = nn.Linear(18,64)
        self.fc_left = nn.Linear(5,64)
        self.fc_right  = nn.Linear(5,64)
        self.fc_cat = nn.Linear(256,lstm_size)
        self.lstm  = nn.LSTM(lstm_size,lstm_size)

        self.fc_pi1 = nn.Linear(lstm_size, 128)
        self.fc_pi2 = nn.Linear(128, 19)

        self.fc_v1 = nn.Linear(lstm_size, 128)
        self.fc_v2 = nn.Linear(128, 1,  bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1,None))
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)

        self.gamma = 0.98
        self.K_epoch = 2
        self.lmbda = 0.95
        self.eps_clip = 0.1
        
    def forward(self, state_dict):
        player_state = state_dict["player"]          
        ball_state = state_dict["ball"]              
        left_team_state = state_dict["left_team"]    
        right_team_state = state_dict["right_team"]  

        player_embed = F.relu(self.fc_player(player_state))
        ball_embed = F.relu(self.fc_ball(ball_state))
        left_team_embed = F.relu(self.fc_left(left_team_state))
        right_team_embed = F.relu(self.fc_right(right_team_state))
        
        left_team_embed = self.pool(left_team_embed).squeeze(2)
        right_team_embed = self.pool(right_team_embed).squeeze(2)

        cat = torch.cat([player_embed, ball_embed, left_team_embed,  right_team_embed], 2)
        cat = F.relu(self.fc_cat(cat))
        h_in = state_dict["hidden"]
        out, h_out = self.lstm(cat, h_in)
        
        prob = F.relu(self.fc_pi1(out))
        prob = F.relu(self.fc_pi2(prob))
        prob = F.softmax(prob, dim=2)

        v = F.relu(self.fc_v1(out))
        v = self.fc_v2(v)

        return prob, v, h_out

    def make_batch(self, data):
        # data = [tr1, tr2, ..., tr10] * batch_size
        s_player_batch, s_ball_batch, s_left_batch, s_right_batch =  [], [], [], []
        s_player_prime_batch, s_ball_prime_batch, s_left_prime_batch, s_right_prime_batch =  [], [], [], []
        h1_in_batch, h2_in_batch, h1_out_batch, h2_out_batch = [], [], [], []
        a_batch, r_batch, prob_a_batch, done_batch = [], [], [], []
        
        for rollout in data:
            s_player_lst, s_ball_lst, s_left_lst, s_right_lst =  [], [], [], []
            s_player_prime_lst, s_ball_prime_lst, s_left_prime_lst, s_right_prime_lst =  [], [], [], []
            h1_in_lst, h2_in_lst, h1_out_lst, h2_out_lst = [], [], [], []
            a_lst, r_lst, prob_a_lst, done_lst = [], [], [], []
            
            for transition in rollout:
                s, a, r, s_prime, prob_a, done = transition

                s_player_lst.append(s["player"])
                s_ball_lst.append(s["ball"])
                s_left_lst.append(s["left_team"])
                s_right_lst.append(s["right_team"])
                h1_in, h2_in = s["hidden"]
                h1_in_lst.append(h1_in)
                h2_in_lst.append(h2_in)
                
                s_player_prime_lst.append(s_prime["player"])
                s_ball_prime_lst.append(s_prime["ball"])
                s_left_prime_lst.append(s_prime["left_team"])
                s_right_prime_lst.append(s_prime["right_team"])
                h1_out, h2_out = s_prime["hidden"]
                h1_out_lst.append(h1_out)
                h2_out_lst.append(h2_out)

                a_lst.append([a])
                r_lst.append([r])
                prob_a_lst.append([prob_a])
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                
            s_player_batch.append(s_player_lst)
            s_ball_batch.append(s_ball_lst)
            s_left_batch.append(s_left_lst)
            s_right_batch.append(s_right_lst)
            h1_in_batch.append(h1_in_lst[0])
            h2_in_batch.append(h2_in_lst[0])
            
            s_player_prime_batch.append(s_player_prime_lst)
            s_ball_prime_batch.append(s_ball_prime_lst)
            s_left_prime_batch.append(s_left_prime_lst)
            s_right_prime_batch.append(s_right_prime_lst)
            h1_out_batch.append(h1_out_lst[0])
            h2_out_batch.append(h2_out_lst[0])

            a_batch.append(a_lst)
            r_batch.append(r_lst)
            prob_a_batch.append(prob_a_lst)
            done_batch.append(done_lst)
            
        s = {
          "player": torch.tensor(s_player_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "ball": torch.tensor(s_ball_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "left_team": torch.tensor(s_left_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "right_team": torch.tensor(s_right_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "hidden" : (torch.tensor(h1_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2), 
                      torch.tensor(h2_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2))
        }

        s_prime = {
          "player": torch.tensor(s_player_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "ball": torch.tensor(s_ball_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
          "left_team": torch.tensor(s_left_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "right_team": torch.tensor(s_right_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2,3),
          "hidden" : (torch.tensor(h1_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2), 
                      torch.tensor(h2_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1,0,2))
        }

        a,r,done_mask, prob_a = torch.tensor(a_batch, device=self.device).permute(1,0,2), \
                                torch.tensor(r_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                torch.tensor(done_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                torch.tensor(prob_a_batch, dtype=torch.float, device=self.device).permute(1,0,2)
        
        
        return s, a, r, s_prime, done_mask, prob_a
    

    def train_net(self, data):
        for i in range(self.K_epoch):
            for mini_batch in data:
                s, a, r, s_prime, done_mask, prob_a = mini_batch
                pi, v, _ = self.forward(s)
                pi_prime, v_prime, _ = self.forward(s_prime)

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
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v, td_target.detach())

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()