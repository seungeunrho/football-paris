import time
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class DPA(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DPA, self).__init__()
        self.fc_q = nn.Linear(dim_in, dim_out)
        self.fc_k = nn.Linear(dim_in, dim_out)
        self.fc_v = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        q = self.fc_q(x['q'])
        k = self.fc_q(x['k'])
        v = self.fc_q(x['v'])

        A = torch.softmax(torch.matmul(q, torch.transpose(k, -2, -1)) / k.shape[-1]**0.5, dim=-1)
        h = torch.matmul(A, v)

        return h

class TrXLI(nn.Module):
    def __init__(self, dim, dim_hidden, n_head):
        super(TrXLI, self).__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_k = nn.LayerNorm(dim)
        self.ln_v = nn.LayerNorm(dim)

        self.heads = nn.ModuleList([DPA(dim, dim_hidden//n_head) for _ in range(n_head)])

        self.fc_mhdpa = nn.Linear(dim_hidden, dim)
        
        self.fc1_mlp = nn.Linear(dim, dim)
        self.ln1_mlp = nn.LayerNorm(dim)

    def forward(self, x):
        x_norm = {
            'q':self.ln_q(x['q']),
            'k':self.ln_k(x['k']),
            'v':self.ln_v(x['v']),
        }

        hs = [head(x_norm) for head in self.heads]
        h = torch.cat(hs, dim=-1)
        h = self.fc_mhdpa(h) + x['q']
        
        h_mlp = F.relu(self.fc1_mlp(self.ln1_mlp(h)))
        h_mlp = h_mlp + h

        return h_mlp

class Transformer(nn.Module):
    def __init__(self, dim, dim_hidden, n_head):
        super(Transformer, self).__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_k = nn.LayerNorm(dim)
        self.ln_v = nn.LayerNorm(dim)

        self.heads = nn.ModuleList([DPA(dim, dim_hidden//n_head) for _ in range(n_head)])

        self.fc_mhdpa = nn.Linear(dim_hidden, dim)
        
        self.fc1_mlp = nn.Linear(dim, dim)
        self.ln1_mlp = nn.LayerNorm(dim)

    def forward(self, x):
        x_norm = {
            'q':self.ln_q(x['q']),
            'k':self.ln_k(x['k']),
            'v':self.ln_v(x['v']),
        }

        hs = [head(x_norm) for head in self.heads]
        h = torch.cat(hs, dim=-1)
        h = self.fc_mhdpa(h) + x['q']
        
        h_mlp = F.relu(self.fc1_mlp(self.ln1_mlp(h)))
        h_mlp = h_mlp + h

        return h_mlp

class PPO(nn.Module):
    def __init__(self, arg_dict, device=None):
        super(PPO, self).__init__()
        if device:
            self.device = device

        self.fc_player = nn.Linear(arg_dict["feature_dims"]["player"],64)  
        self.fc_ball = nn.Linear(arg_dict["feature_dims"]["ball"],64)
        self.fc_left = nn.Linear(arg_dict["feature_dims"]["left_team"],64)
        self.fc_right  = nn.Linear(arg_dict["feature_dims"]["right_team"],64)
        self.norm_player = nn.LayerNorm(64)
        self.norm_ball = nn.LayerNorm(64)
        self.norm_left = nn.LayerNorm(64)
        self.norm_right = nn.LayerNorm(64)

        # self.trxli = TrXLI(dim=64, dim_hidden=128, n_head=8)
        self.trxli = Transformer(dim=64, dim_hidden=128, n_head=8)
        self.lstm  = nn.LSTM(3 * 64, arg_dict['lstm_size'])

        self.fc_pi_a1 = nn.Linear(arg_dict['lstm_size'], 128)
        self.fc_pi_a2 = nn.Linear(128, 12)
        self.norm_pi_a1 = nn.LayerNorm(128)
        
        self.fc_pi_m1 = nn.Linear(arg_dict['lstm_size'], 128)
        self.fc_pi_m2 = nn.Linear(128, 8)
        self.norm_pi_m1 = nn.LayerNorm(128)

        self.fc_v1 = nn.Linear(arg_dict['lstm_size'], 128)
        self.norm_v1 = nn.LayerNorm(128)
        self.fc_v2 = nn.Linear(128, 1,  bias=False)
        
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict['learning_rate'])

        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        self.lmbda = arg_dict["lmbda"]
        self.eps_clip = 0.2
        self.entropy_coef = arg_dict["entropy_coef"]
        
    def forward(self, state_dict):
        player_state = state_dict["player"]          
        ball_state = state_dict["ball"]              
        left_team_state = state_dict["left_team"]
        right_team_state = state_dict["right_team"]  
        avail = state_dict["avail"]
        
        player_embed = self.norm_player(self.fc_player(player_state))
        ball_embed = self.norm_ball(self.fc_ball(ball_state))
        left_team_embed = self.norm_left(self.fc_left(left_team_state))
        right_team_embed = self.norm_right(self.fc_right(right_team_state))
        
        embed_cat = torch.cat([left_team_embed, right_team_embed], dim=2)
        h_trxli = {'q':player_embed.unsqueeze(-2), 'k':embed_cat, 'v':embed_cat}
        h_trxli = self.trxli(h_trxli).squeeze(-2)

        cat = torch.cat([player_embed, ball_embed, h_trxli], dim=-1)

        h_in = state_dict["hidden"]
        out, h_out = self.lstm(cat, h_in)
        
        a_out = self.norm_pi_a1(F.relu(self.fc_pi_a1(out)))
        a_out = self.fc_pi_a2(a_out)
        logit = a_out + (avail-1)*1e8
        prob = F.softmax(logit, dim=2)
        
        prob_m = self.norm_pi_m1(F.relu(self.fc_pi_m1(out)))
        prob_m = self.fc_pi_m2(prob_m)
        prob_m = F.softmax(prob_m, dim=2)

        v = self.norm_v1(F.relu(self.fc_v1(out)))
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
                pi, pi_m, v, _ = self.forward(s)
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
                pi_m_chosen = pi_m.gather(2,m)
                pi_am = pi_a * (1 - need_move) + pi_a*pi_m_chosen * need_move
                ratio = torch.exp(torch.log(pi_am) - torch.log(prob))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                entropy = -torch.sum(pi*torch.log(pi+ 1e-8), dim=2, keepdim=True) - need_move * pi_a * torch.sum(pi_m * torch.log(pi_m + 1e-8), dim=2, keepdim=True)

                surr_loss = -torch.min(surr1, surr2)
                v_loss = F.smooth_l1_loss(v, td_target.detach())
                entropy_loss = -self.entropy_coef * entropy
                loss = surr_loss + v_loss + entropy_loss
                #loss = surr_loss + v_loss
                loss = loss.mean()
#                 print(i,loss)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                tot_loss_lst.append(loss.item())
                pi_loss_lst.append(surr_loss.mean().item())
                v_loss_lst.append(v_loss.item())
                entropy_lst.append(entropy.mean().item())
                
        return np.mean(tot_loss_lst), np.mean(pi_loss_lst), np.mean(v_loss_lst), np.mean(entropy_lst) 
                