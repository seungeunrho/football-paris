import gfootball.env as football_env
import time, pprint, importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from tensorboardX import SummaryWriter


def write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, optimization_step):
    win, score, tot_reward, game_len = [], [], [], []
    loop_t, forward_t, wait_t = [], [], []

    for i in range(arg_dict["summary_game_window"]):
        game_data = summary_queue.get()
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
    writer.add_scalar('train/pi_loss', np.mean(pi_loss_lst), n_game)
    writer.add_scalar('train/v_loss', np.mean(v_loss_lst), n_game)
    writer.add_scalar('train/entropy', np.mean(entropy_lst), n_game)

def save_model(model, arg_dict, optimization_step, last_saved_step):
    if optimization_step >= last_saved_step + arg_dict["model_save_interval"]:
        torch.save(model.state_dict(), arg_dict["log_dir"]+"/model_"+str(optimization_step)+".pt")
        return optimization_step
    else:
        return last_saved_step
        
def get_data(queue, arg_dict, model):
    data = []
    for i in range(arg_dict["buffer_size"]):
        mini_batch_np = []
        for j in range(arg_dict["batch_size"]):
            rollout = queue.get()
            mini_batch_np.append(rollout)
        mini_batch = model.make_batch(mini_batch_np)
        data.append(mini_batch)
    return data

def learner(center_model, queue, signal_queue, summary_queue, arg_dict):
    print("Learner process started")
    imported_model = importlib.import_module("Model." + arg_dict["model"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = imported_model.PPO(arg_dict, device)
    model.load_state_dict(center_model.state_dict())
    model.to(device)
    
    writer = SummaryWriter(logdir=arg_dict["log_dir"])
    optimization_step = 0
    n_game = 0
    last_saved_step = 0
    loss_lst, pi_loss_lst, v_loss_lst, entropy_lst = [], [], [], []
    
    while True:
        if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
            last_saved_step = save_model(model, arg_dict, optimization_step, last_saved_step)
            
            signal_queue.put(1)
            data = get_data(queue, arg_dict, model)
            loss, pi_loss, v_loss, entropy = model.train_net(data)
            optimization_step += arg_dict["batch_size"]*arg_dict["buffer_size"]*arg_dict["k_epoch"]
            
            print("step :", optimization_step, "loss", loss, "data_q", queue.qsize(), "summary_q", summary_queue.qsize())
            loss_lst.append(loss)
            pi_loss_lst.append(pi_loss)
            v_loss_lst.append(v_loss)
            entropy_lst.append(entropy)
            center_model.load_state_dict(model.state_dict())
            
            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                print("warning. data remaining. queue size : ", queue.qsize())
            
            if summary_queue.qsize() > arg_dict["summary_game_window"]:
                write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, optimization_step)
                loss_lst, pi_loss_lst, v_loss_lst, entropy_lst = [], [], [], []
                n_game += arg_dict["summary_game_window"]
                
            _ = signal_queue.get()             
            
        else:
            time.sleep(0.1)
            
