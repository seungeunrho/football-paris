import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
import numpy as np


class Algo():
    def __init__(self, arg_dict, device=None):
        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        self.lmbda = arg_dict["lmbda"]
        self.eps_clip = arg_dict["eps_clip"]
        self.entropy_coef = arg_dict["entropy_coef"]
        self.grad_clip = arg_dict["grad_clip"]

    def train(self, model, data):
        tot_loss_lst = []
        pi_loss_lst = []
        entropy_lst = []
        move_entropy_lst = []
        v_loss_lst = []

        # to calculate fixed advantages before update
        data_with_adv = []
        for mini_batch in data:
            s, a, m, r, s_prime, done_mask, prob, need_move = mini_batch
            with torch.no_grad():
                pi, pi_move, v, _ = model(s)
                pi_prime, pi_m_prime, v_prime, _ = model(s_prime)

            td_target = r + self.gamma * v_prime * done_mask
            delta = td_target - v                           # [horizon * batch_size * 1]
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = np.array([0])
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t           
                advantage_lst.append(advantage)
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float, device=model.device)

            data_with_adv.append((s, a, m, r, s_prime, done_mask, prob, need_move, td_target, advantage))

        for i in range(self.K_epoch):
            for mini_batch in data_with_adv:
                s, a, m, r, s_prime, done_mask, prob, need_move, td_target, advantage = mini_batch
                pi, pi_move, v, _ = model(s)
                pi_prime, pi_m_prime, v_prime, _ = model(s_prime)

                pi_a = pi.gather(2,a)
                pi_m = pi_move.gather(2,m)
                pi_am = pi_a*(1-need_move + need_move*pi_m)
                ratio = torch.exp(torch.log(pi_am) - torch.log(prob))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                entropy = -torch.log(pi_am)
                move_entropy = -need_move*torch.log(pi_m)

                surr_loss = -torch.min(surr1, surr2)
                v_loss = F.smooth_l1_loss(v, td_target.detach())
                entropy_loss = -1*self.entropy_coef*entropy
                loss = surr_loss + v_loss + entropy_loss.mean()
                loss = loss.mean()

                model.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                model.optimizer.step()

                tot_loss_lst.append(loss.item())
                pi_loss_lst.append(surr_loss.mean().item())
                v_loss_lst.append(v_loss.item())
                entropy_lst.append(entropy.mean().item())
                n_need_move = torch.sum(need_move).item()
                if n_need_move == 0:
                    move_entropy_lst.append(0)
                else:
                    move_entropy_lst.append((torch.sum(move_entropy)/n_need_move).item())
        return np.mean(tot_loss_lst), np.mean(pi_loss_lst), np.mean(v_loss_lst), np.mean(entropy_lst), np.mean(move_entropy_lst)
