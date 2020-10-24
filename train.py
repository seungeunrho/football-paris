import gfootball.env as football_env
import time, pprint, json, os
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
from actor import *
from learner import *
from datetime import datetime, timedelta

#11_vs_11_easy_stochastic
#academy_empty_goal_close 300 epi done
#academy_empty_goal 450 epi done

if __name__ == '__main__':
    # hyperparameters
    arg_dict = {
        "env": "11_vs_11_easy_stochastic",
        "num_processes": 8,
        "batch_size": 16,   
        "buffer_size": 6,
        "rollout_len": 10,
        "lstm_size" : 196,
        "k_epoch" : 3,
        "summary_game_window" : 5
    }
    
    cur_time = datetime.now() + timedelta(hours = 9)
    arg_dict["log_dir"] = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S")
    os.mkdir(arg_dict["log_dir"])
    json = json.dumps(arg_dict, indent=4)
    f = open(arg_dict["log_dir"]+"/args.json","w")
    f.write(json)
    f.close()

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    pp = pprint.PrettyPrinter(indent=4)
    torch.set_num_threads(1)
    
    center_model = PPO(arg_dict["lstm_size"], arg_dict["k_epoch"])
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
        
        
