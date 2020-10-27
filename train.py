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

#11_vs_11_easy_stochastic
#academy_empty_goal_close 300 epi done
#academy_empty_goal 450 epi done


def save_args(arg_dict):
    os.mkdir(arg_dict["log_dir"])
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
    center_model = model.PPO(arg_dict)
    center_model.share_memory()
    data_queue = mp.Queue()
    signal_queue = mp.Queue()
    summary_queue = mp.Queue()
    processes = []
    
    p = mp.Process(target=learner, args=(center_model, data_queue, signal_queue, summary_queue, arg_dict))
    p.start()
    processes.append(p)
    for rank in range(arg_dict["num_processes"]):
        p = mp.Process(target=actor, args=(rank, center_model, data_queue, signal_queue, summary_queue, arg_dict))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    

if __name__ == '__main__':
    # hyperparameters
    arg_dict = {
        "env": "11_vs_11_easy_stochastic",
        "num_processes": 9,
        "batch_size": 16,   
        "buffer_size": 5,
        "rollout_len": 20,
        "lstm_size" : 196,
        "k_epoch" : 3,
        "summary_game_window" : 5,
        "model_save_interval" : 100000,
        
        "encoder" : "raw_encoder",
        "rewarder" : "rewarder_se",
        "model" : "ppo"
    }
    
    main(arg_dict)
    
    
        
        
