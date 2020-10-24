import gfootball.env as football_env
import time, pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from tensorboardX import SummaryWriter

from FeatureEncoder import *
from ppo import *
from datetime import datetime, timedelta


def learner(center_model, queue, signal_queue, summary_queue, arg_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PPO(arg_dict["lstm_size"], arg_dict["k_epoch"], device)
    model.load_state_dict(center_model.state_dict())
    model.to(device)
    cur_time = datetime.now() + timedelta(hours = 9)
    log_dir = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S")
    writer = SummaryWriter(logdir=log_dir)
    optimization_step = 0
    n_game = 0
    
    print("learner start")
    
    while True:
        if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
            if optimization_step % 500 == 0:
                torch.save(model.state_dict(), log_dir+"/model_"+str(optimization_step)+".pt")
            signal_queue.put(1)
            data = []
            for j in range(arg_dict["buffer_size"]):
                mini_batch_np = []
                for i in range(arg_dict["batch_size"]):
                    rollout = queue.get()
                    mini_batch_np.append(rollout)
                mini_batch = model.make_batch(mini_batch_np)
                data.append(mini_batch)
            model.train_net(data)
            center_model.load_state_dict(model.state_dict())
            
            optimization_step += arg_dict["batch_size"]*arg_dict["buffer_size"]*arg_dict["k_epoch"]
            
            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                print(" data remaining. queue size : ", queue.qsize())
            
            
            while summary_queue.qsize() > arg_dict["summary_game_window"]:
                win, score, tot_reward, game_len = [], [], [], []
                
                
                for i in range(arg_dict["summary_game_window"]):
                    game_data = summary_queue.get()
                    n_game += 1
                    a,b,c,d = game_data
                    win.append(a)
                    score.append(b)
                    tot_reward.append(c)
                    game_len.append(d)
                writer.add_scalar('game/win', float(np.mean(win)), n_game)
                writer.add_scalar('game/score', float(np.mean(score)), n_game)
                writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
                writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
                writer.add_scalar('train/win', float(optimization_step), n_game)
                
            _ = signal_queue.get()   
                    
            
        else:
            time.sleep(0.1)