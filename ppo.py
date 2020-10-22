import time
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self, device=None):
        super(PPO, self).__init__()
        if device:
            self.device = device

        self.fc_player = nn.Linear(17,64)
        self.fc_ball = nn.Linear(12,64)
        self.fc_left = nn.Linear(4,64)
        self.fc_right  = nn.Linear(4,64)

        self.fc_pi1 = nn.Linear(256, 128)
        self.fc_pi2 = nn.Linear(128, 19)

        self.fc_v1 = nn.Linear(256, 128)
        self.fc_v2 = nn.Linear(128, 1,  bias=False)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.max_pool_batch = nn.AdaptiveMaxPool2d((1,None))
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

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

        left_team_embed = left_team_embed.permute(0,2,1)
        left_team_embed = self.max_pool(left_team_embed).squeeze(2)
        right_team_embed = right_team_embed.permute(0,2,1)
        right_team_embed = self.max_pool(right_team_embed).squeeze(2)

        cat = torch.cat([player_embed, ball_embed, left_team_embed,  right_team_embed], 1)
        prob = F.relu(self.fc_pi1(cat))
        prob = F.relu(self.fc_pi2(prob))
        prob = F.softmax(prob, dim=1)

        v = F.relu(self.fc_v1(cat))
        v = self.fc_v2(v)

        return prob, v
    
    def forward_batch(self, state_dict):
        player_state = state_dict["player"]          # [batch_size, rollout_len, dim]
        ball_state = state_dict["ball"]              # [batch_size, rollout_len, dim]
        left_team_state = state_dict["left_team"]    # [batch_size, rollout_len, n_player, dim]
        right_team_state = state_dict["right_team"]  # [batch_size, rollout_len, n_player, dim]

        player_embed = F.relu(self.fc_player(player_state))
        ball_embed = F.relu(self.fc_ball(ball_state))
        left_team_embed = F.relu(self.fc_left(left_team_state))
        right_team_embed = F.relu(self.fc_right(right_team_state))

        left_team_embed = self.max_pool_batch(left_team_embed).squeeze(2)
        right_team_embed = self.max_pool_batch(right_team_embed).squeeze(2)

        cat = torch.cat([player_embed, ball_embed, left_team_embed,  right_team_embed], 2)
        prob = F.relu(self.fc_pi1(cat))
        prob = F.relu(self.fc_pi2(prob))
        prob = F.softmax(prob, dim=2)

        v = F.relu(self.fc_v1(cat))
        v = self.fc_v2(v)

        return prob, v

    def make_batch(self, data):
        # data = [tr1, tr2, ..., tr10] * batch_size
        s_player_batch, s_ball_batch, s_left_batch, s_right_batch =  [], [], [], []
        s_player_prime_batch, s_ball_prime_batch, s_left_prime_batch, s_right_prime_batch =  [], [], [], []
        a_batch, r_batch, prob_a_batch, done_batch = [], [], [], []
        
        for rollout in data:
            s_player_lst, s_ball_lst, s_left_lst, s_right_lst =  [], [], [], []
            s_player_prime_lst, s_ball_prime_lst, s_left_prime_lst, s_right_prime_lst =  [], [], [], []
            a_lst, r_lst, prob_a_lst, done_lst = [], [], [], []
            
            for transition in rollout:
                s, a, r, s_prime, prob_a, done = transition

                s_player_lst.append(s["player"])
                s_ball_lst.append(s["ball"])
                s_left_lst.append(s["left_team"])
                s_right_lst.append(s["right_team"])
                s_player_prime_lst.append(s_prime["player"])
                s_ball_prime_lst.append(s_prime["ball"])
                s_left_prime_lst.append(s_prime["left_team"])
                s_right_prime_lst.append(s_prime["right_team"])

                a_lst.append([a])
                r_lst.append([r])
                prob_a_lst.append([prob_a])
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                
            s_player_batch.append(s_player_lst)
            s_ball_batch.append(s_ball_lst)
            s_left_batch.append(s_left_lst)
            s_right_batch.append(s_right_lst)
            s_player_prime_batch.append(s_player_prime_lst)
            s_ball_prime_batch.append(s_ball_prime_lst)
            s_left_prime_batch.append(s_left_prime_lst)
            s_right_prime_batch.append(s_right_prime_lst)

            a_batch.append(a_lst)
            r_batch.append(r_lst)
            prob_a_batch.append(prob_a_lst)
            done_batch.append(done_lst)
                
        s = {
          "player": torch.tensor(s_player_batch, dtype=torch.float, device=self.device),
          "ball": torch.tensor(s_ball_batch, dtype=torch.float, device=self.device),
          "left_team": torch.tensor(s_left_batch, dtype=torch.float, device=self.device),
          "right_team": torch.tensor(s_right_batch, dtype=torch.float, device=self.device),
        }

        s_prime = {
          "player": torch.tensor(s_player_prime_batch, dtype=torch.float, device=self.device),
          "ball": torch.tensor(s_ball_prime_batch, dtype=torch.float, device=self.device),
          "left_team": torch.tensor(s_left_prime_batch, dtype=torch.float, device=self.device),
          "right_team": torch.tensor(s_right_prime_batch, dtype=torch.float, device=self.device),
        }

        a,r,done_mask, prob_a = torch.tensor(a_batch, device=self.device), torch.tensor(r_batch, device=self.device), \
                                torch.tensor(done_batch, dtype=torch.float, device=self.device), torch.tensor(prob_a_batch, device=self.device)
        
        
        return s, a, r, s_prime, done_mask, prob_a
    

    def train_net(self, data):
        for i in range(self.K_epoch):
            for mini_batch in data:
                s, a, r, s_prime, done_mask, prob_a = mini_batch
                pi, v = self.forward_batch(s)
                pi_prime, v_prime = self.forward_batch(s_prime)

                td_target = r + self.gamma * v_prime * done_mask
                delta = td_target - v                           # [batch_size * horizon * 1]
                delta = delta.detach().cpu().numpy()

                advantage_batch = []
                for delta_row in delta:
                    advantage_lst = []
                    advantage = 0.0
                    for delta_t in delta_row[::-1]:
                      advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                      advantage_lst.append([advantage])
                    advantage_lst.reverse()
                    advantage_batch.append(advantage_lst)
                advantage = torch.tensor(advantage_batch, dtype=torch.float, device=self.device)

                pi_a = pi.gather(2,a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v, td_target.detach())

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()