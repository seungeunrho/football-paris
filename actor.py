import gfootball.env as football_env
import time, pprint, importlib, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from os import listdir
from os.path import isfile, join
import numpy as np

from datetime import datetime, timedelta


def state_to_tensor(state_dict, h_in):
    player_state = torch.from_numpy(state_dict["player"]).float().unsqueeze(0).unsqueeze(0)
    ball_state = torch.from_numpy(state_dict["ball"]).float().unsqueeze(0).unsqueeze(0)
    left_team_state = torch.from_numpy(state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0)
    left_closest_state = torch.from_numpy(state_dict["left_closest"]).float().unsqueeze(0).unsqueeze(0)
    right_team_state = torch.from_numpy(state_dict["right_team"]).float().unsqueeze(0).unsqueeze(0)
    right_closest_state = torch.from_numpy(state_dict["right_closest"]).float().unsqueeze(0).unsqueeze(0)
    avail = torch.from_numpy(state_dict["avail"]).float().unsqueeze(0).unsqueeze(0)

    state_dict_tensor = {
      "player" : player_state,
      "ball" : ball_state,
      "left_team" : left_team_state,
      "left_closest" : left_closest_state,
      "right_team" : right_team_state,
      "right_closest" : right_closest_state,
      "avail" : avail,
      "hidden" : h_in
    }
    return state_dict_tensor


def get_action(a_prob, m_prob):
    
    a = Categorical(a_prob).sample().item()    
    m, need_m = 0, 0
    prob_selected_a = a_prob[0][0][a].item()
    prob_selected_m = 0
    if a==0:
        real_action = a
        prob = prob_selected_a
    elif a==1:
        m = Categorical(m_prob).sample().item()
        need_m = 1
        real_action = m + 1
        prob_selected_m = m_prob[0][0][m].item()
        prob = prob_selected_a* prob_selected_m
    else:
        real_action = a + 7
        prob = prob_selected_a

    assert prob != 0, 'prob 0 ERROR!!!! a : {}, m:{}  {}, {}'.format(a,m,prob_selected_a,prob_selected_m)
    
    return real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m

def actor(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    
    fe = fe_module.FeatureEncoder()
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())
    
    env = football_env.create_environment(env_name=arg_dict["env"], representation="raw", stacked=False, logdir='/tmp/football', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    n_epi = 0
    rollout = []
    while True: # episode loop
        env.reset()   
        done = False
        steps, score, tot_reward, win = 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        obs = env.observation()
        
        while not done:  # step loop
            init_t = time.time()
            
            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t
            
            h_in = h_out
            state_dict = fe.encode(obs[0])
            state_dict_tensor = state_to_tensor(state_dict, h_in)
            
            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = model(state_dict_tensor)
            forward_t += time.time()-t1 
            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)

            prev_obs = obs
            obs, rew, done, info = env.step(real_action)
            fin_r = rewarder.calc_reward(rew, prev_obs[0], obs[0])
            state_prime_dict = fe.encode(obs[0])
            
            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())

            steps += 1
            score += rew
            tot_reward += fin_r
            
            if arg_dict['print_mode']:
                print_status(steps,a,m,prob_selected_a,prob_selected_m,prev_obs,obs,fin_r,tot_reward)
            loop_t += time.time()-init_t
            
            if done:
                if score > 0:
                    win = 1
                print("score",score,"total reward",tot_reward)
                summary_data = (win, score, tot_reward, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)

def select_opponent(arg_dict):
    onlyfiles_lst = [f for f in listdir(arg_dict["log_dir"]) if isfile(join(arg_dict["log_dir"], f))]
    model_num_lst = []
    for file_name in onlyfiles_lst:
        if file_name[:6] == "model_":
            model_num = file_name[6:]
            model_num = model_num[:-4]
            model_num_lst.append(int(model_num))
    model_num_lst.sort()
            
    coin = random.random()
    if coin<arg_dict["latest_ratio"]:
        if len(model_num_lst) > arg_dict["latest_n_model"]:
            opp_model_num = random.randint(len(model_num_lst)-arg_dict["latest_n_model"],len(model_num_lst)-1)
        else:
            opp_model_num = len(model_num_lst)-1
    else:
        opp_model_num = random.randint(0,len(model_num_lst)-1)
        
    model_name = "/model_"+str(model_num_lst[opp_model_num])+".tar"
    opp_model_path = arg_dict["log_dir"] + model_name
    return opp_model_num, opp_model_path
                
                
def actor_self(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    print("Actor process {} started".format(actor_num))
    cpu_device = torch.device('cpu')
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    
    fe = fe_module.FeatureEncoder()
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())
    opp_model = imported_model.Model(arg_dict)
    
    env = football_env.create_environment(env_name=arg_dict["env"], number_of_right_players_agent_controls=1, representation="raw", \
                                          stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, \
                                          render=False)

    n_epi = 0
    rollout = []
    while True: # episode loop
        opp_model_num, opp_model_path = select_opponent(arg_dict)
        checkpoint = torch.load(opp_model_path, map_location=cpu_device)
        opp_model.load_state_dict(checkpoint['model_state_dict'])
        print("Current Opponent model Num:{}, Path:{} successfully loaded".format(opp_model_num, opp_model_path))
        del checkpoint
        
        env.reset()   
        done = False
        steps, score, tot_reward, win = 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        opp_h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                     torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        [obs, opp_obs] = env.observation()
        
        while not done:  # step loop
            init_t = time.time()
            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t
            
            h_in = h_out
            opp_h_in = opp_h_out
            state_dict = fe.encode(obs)
            state_dict_tensor = state_to_tensor(state_dict, h_in)
            opp_state_dict = fe.encode(opp_obs)
            opp_state_dict_tensor = state_to_tensor(opp_state_dict, opp_h_in)
            
            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = model(state_dict_tensor)
                opp_a_prob, opp_m_prob, _, opp_h_out = opp_model(opp_state_dict_tensor)
            forward_t += time.time()-t1 
            
            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
            opp_real_action, _, _, _, _, _, _ = get_action(opp_a_prob, opp_m_prob)

            prev_obs = obs
            [obs, opp_obs], [rew, _], done, info = env.step([real_action, opp_real_action])
            fin_r = rewarder.calc_reward(rew, prev_obs, obs)
            state_prime_dict = fe.encode(obs)

            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())

            steps += 1
            score += rew
            tot_reward += fin_r
            
            if arg_dict['print_mode']:
                print_status(steps,a,m,prob_selected_a,prob_selected_m,prev_obs,obs,fin_r,tot_reward)
            
            loop_t += time.time()-init_t

            if done:
                if score > 0:
                    win = 1
                print("score {}, total reward {:.2f}, opp num:{}, opp:{} ".format(score,tot_reward,opp_model_num, opp_model_path))
                summary_data = (win, score, tot_reward, steps, str(opp_model_num), loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)                

