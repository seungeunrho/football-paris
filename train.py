import gfootball.env as football_env
import time, pprint, json, os, importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from tensorboardX import SummaryWriter

from actor import *
from learner import *
from datetime import datetime, timedelta



def save_args(arg_dict):
    os.makedirs(arg_dict["log_dir"])
    args_info = json.dumps(arg_dict, indent=4)
    f = open(arg_dict["log_dir"]+"/args.json","w")
    f.write(args_info)
    f.close()
    
    
    
def main(arg_dict):
    cur_time = datetime.now() + timedelta(hours = 9)
    arg_dict["log_dir"] = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S")
    save_args(arg_dict)

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    pp = pprint.PrettyPrinter(indent=4)
    torch.set_num_threads(1)
    
    fe = importlib.import_module("FeatureEncoder." + arg_dict["encoder"])
    fe = fe.FeatureEncoder()
    arg_dict["feature_dims"] = fe.get_feature_dims()
    
    model = importlib.import_module("Model." + arg_dict["model"])
    cpu_device = torch.device('cpu')
    center_model = model.PPO(arg_dict)
    if arg_dict["trained_model_path"]:
        checkpoint = torch.load(arg_dict["trained_model_path"], map_location=cpu_device)
        optimization_step = checkpoint['optimization_step']
        center_model.load_state_dict(checkpoint['model_state_dict'])
        center_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        arg_dict["optimization_step"] = optimization_step
        print("Trained model", arg_dict["trained_model_path"] ,"suffessfully loaded") 
    else:
        model_dict = {
            'optimization_step': 0,
            'model_state_dict': center_model.state_dict(),
            'optimizer_state_dict': center_model.optimizer.state_dict(),
        }
        path = arg_dict["log_dir"]+"/model_0.tar"
        torch.save(model_dict, path)
        
        
    center_model.share_memory()
    data_queue = mp.Queue()
    signal_queue = mp.Queue()
    summary_queue = mp.Queue()
    processes = []
    
    p = mp.Process(target=learner, args=(center_model, data_queue, signal_queue, summary_queue, arg_dict))
    p.start()
    processes.append(p)
    for rank in range(arg_dict["num_processes"]):
        if arg_dict["env"] == "11_vs_11_kaggle":
            p = mp.Process(target=actor_self, args=(rank, center_model, data_queue, signal_queue, summary_queue, arg_dict))
        else:
            p = mp.Process(target=actor, args=(rank, center_model, data_queue, signal_queue, summary_queue, arg_dict))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    

if __name__ == '__main__':
    # envs
    #11_vs_11_easy_stochastic  : vs. easy_rule
    #11_vs_11_stochastic  : vs. medium_rule
    #11_vs_11_kaggle  : vs. self-play

    # hyperparameters
    arg_dict = {
        "env": "11_vs_11_kaggle",
        "num_processes": 9,
        "batch_size": 32,   
        "buffer_size": 5,
        "rollout_len": 30,
        "lstm_size" : 196,
        "k_epoch" : 3,
        "summary_game_window" : 10,
        "model_save_interval" : 100000,
        "learning_rate" : 0.0001,
        "gamma" : 0.993,
        "lmbda" : 0.96,
        "entropy_coef" : 0.00003,
#         "trained_model_path" : "logs/[10-28]03.56.37/model_800640.pt",   # default : None
        "trained_model_path" : None,
        "print_mode" : False,
        "latest_ratio" : 0.5,
        
        "encoder" : "encoder_raw",
        "rewarder" : "rewarder_se",
        "model" : "ppo_conv1d"
    }

    
    main(arg_dict)
    
    
        
        
