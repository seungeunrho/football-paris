import gfootball.env as football_env
import time
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from FeatureEncoder import *
from ppo import *

import torch.multiprocessing as mp 

def actor(actor_num, center_model, data_queue, signal_queue, arg_dict):
    print("actor {} started".format(actor_num))
    #11_vs_11_easy_stochastic
    #academy_empty_goal_close 300 epi done
    #academy_empty_goal 450 epi done
    model = PPO(arg_dict["lstm_size"])
    model.load_state_dict(center_model.state_dict())
    env = football_env.create_environment(env_name="11_vs_11_easy_stochastic", representation="raw", stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    fe = FeatureEncoder()
    
    n_epi = 0
    score = 0
    rollout = []
    
    while True:
        env.reset()
        done = False
        steps = 0
#         score = 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        while not done:
            t1 = time.time()
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
            else:
                model.load_state_dict(center_model.state_dict())
                
            obs = env.observation()
            state_dict = fe.encode(obs[0])
            player_state = torch.from_numpy(state_dict["player"]).float().unsqueeze(0).unsqueeze(0)
            ball_state = torch.from_numpy(state_dict["ball"]).float().unsqueeze(0).unsqueeze(0)
            left_team_state = torch.from_numpy(state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0)
            right_team_state = torch.from_numpy(state_dict["right_team"]).float().unsqueeze(0).unsqueeze(0)
            
            h_in = h_out

            state_dict_tensor = {
              "player" : player_state,
              "ball" : ball_state,
              "left_team" : left_team_state,
              "right_team" : right_team_state,
              "hidden" : h_in
            }
            
            with torch.no_grad():
                prob, _, h_out = model(state_dict_tensor)
            m = Categorical(prob)
            a = m.sample().item()

            prev_obs = obs
            obs, rew, done, info = env.step(a)
            additional_r = fe.calc_additional_reward(prev_obs[0], obs[0])
            fin_r = rew*3.0 + additional_r
            
            state_prime_dict = fe.encode(obs[0])
            
            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())

            transition = (state_dict, a, fin_r, state_prime_dict, prob[0][0][a].item(), done)
            rollout.append(transition)

            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                
            state_dict = state_prime_dict

            steps += 1
            score += rew

            if done:
                if n_epi % 2 == 0 and actor_num == 0:
                    print("%d, Done, Step %d Reward: %f" % (n_epi, steps, score))
                    score = 0   


def learner(center_model, queue, signal_queue, arg_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PPO(arg_dict["lstm_size"], device)
    model.load_state_dict(center_model.state_dict())
    model.to(device)
    
    print("learner start")
    
    while True:
        if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
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
            
            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                print(" data remaining. queue size : ", queue.qsize())
            signal_queue.get()
            
        else:
            time.sleep(0.1)
    

if __name__ == '__main__':
    # hyperparameters
    arg_dict = {
        "num_processes": 6,
        "batch_size": 16,   
        "buffer_size": 5,
        "rollout_len": 10,
        "lstm_size" : 196
    }
    
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    pp = pprint.PrettyPrinter(indent=4)
    torch.set_num_threads(1)
    
    center_model = PPO(arg_dict["lstm_size"])
    center_model.share_memory()
    data_queue = mp.Queue()
    signal_queue = mp.Queue()
    processes = []
    
    p = mp.Process(target=learner, args=(center_model, data_queue, signal_queue, arg_dict))
    p.start()
    processes.append(p)
    for rank in range(arg_dict["num_processes"]):
        p = mp.Process(target=actor, args=(rank, center_model, data_queue, signal_queue, arg_dict))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
        
