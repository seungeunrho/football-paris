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



def learner(center_model, queue, signal_queue, summary_queue, arg_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PPO(arg_dict["lstm_size"], arg_dict["k_epoch"], device)
    model.load_state_dict(center_model.state_dict())
    model.to(device)
    log_dir = arg_dict["log_dir"]
    writer = SummaryWriter(logdir=log_dir)
    optimization_step = 0
    n_game = 0
    last_saved_step = 0
    loss_lst, entropy_lst = [], []
    save_interval = 1000
    
    print("learner start")
    
    while True:
        if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
            if optimization_step >= last_saved_step + save_interval:
                torch.save(model.state_dict(), log_dir+"/model_"+str(optimization_step)+".pt")
                last_saved_step = optimization_step
            signal_queue.put(1)
            data = []
            for j in range(arg_dict["buffer_size"]):
                mini_batch_np = []
                for i in range(arg_dict["batch_size"]):
                    rollout = queue.get()
                    mini_batch_np.append(rollout)
                mini_batch = model.make_batch(mini_batch_np)
                data.append(mini_batch)
            loss, entropy = model.train_net(data)
            loss_lst.append(loss)
            entropy_lst.append(entropy)
            center_model.load_state_dict(model.state_dict())
            print("step :", optimization_step, "loss", loss, "data_q", queue.qsize(), "summary_q", summary_queue.qsize() )
            
            optimization_step += arg_dict["batch_size"]*arg_dict["buffer_size"]*arg_dict["k_epoch"]
            
            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                print(" data remaining. queue size : ", queue.qsize())
            
            
            if summary_queue.qsize() > arg_dict["summary_game_window"]:
                win, score, tot_reward, game_len = [], [], [], []
                loop_t, forward_t, wait_t = [], [], []
                
                for i in range(arg_dict["summary_game_window"]):
                    game_data = summary_queue.get()
                    n_game += 1
                    a,b,c,d,t1,t2,t3 = game_data
                    win.append(a)
                    score.append(b)
                    tot_reward.append(c)
                    game_len.append(d)
                    loop_t.append(t1)
                    forward_t.append(t2)
                    wait_t.append(t3)
                writer.add_scalar('game/win_rate', float(np.mean(win)), n_game)
                writer.add_scalar('game/score', float(np.mean(score)), n_game)
                writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
                writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
                writer.add_scalar('train/step', float(optimization_step), n_game)
                writer.add_scalar('time/loop', float(np.mean(loop_t)), n_game)
                writer.add_scalar('time/forward', float(np.mean(forward_t)), n_game)
                writer.add_scalar('time/wait', float(np.mean(wait_t)), n_game)
                writer.add_scalar('train/loss', np.mean(loss_lst), n_game)
                writer.add_scalar('train/entropy', np.mean(loss_lst), n_game)
                loss_lst, entropy_lst = [], []
                
            _ = signal_queue.get()             
            
        else:
            time.sleep(0.1)