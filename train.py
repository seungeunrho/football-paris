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
from actor import *
from learner import *
from datetime import datetime, timedelta




# def actor(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
#     print("actor {} started".format(actor_num))
#     #11_vs_11_easy_stochastic
#     #academy_empty_goal_close 300 epi done
#     #academy_empty_goal 450 epi done
#     model = PPO(arg_dict["lstm_size"], arg_dict["k_epoch"])
#     model.load_state_dict(center_model.state_dict())
#     fe = FeatureEncoder()
    
#     env = football_env.create_environment(env_name="11_vs_11_stochastic", representation="raw", stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, render=False)
#     n_epi = 0
#     rollout = []
#     while True:
#         env.reset()   
#         done = False
#         score = 0
#         win = 0
#         steps = 0
#         tot_reward = 0
#         n_epi += 1
#         h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
#                  torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        
#         while not done:
#             t1 = time.time()
#             while signal_queue.qsize() > 0:
#                 time.sleep(0.02)
#             else:
#                 model.load_state_dict(center_model.state_dict())
                
#             obs = env.observation()
#             state_dict = fe.encode(obs[0])
#             player_state = torch.from_numpy(state_dict["player"]).float().unsqueeze(0).unsqueeze(0)
#             ball_state = torch.from_numpy(state_dict["ball"]).float().unsqueeze(0).unsqueeze(0)
#             left_team_state = torch.from_numpy(state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0)
#             right_team_state = torch.from_numpy(state_dict["right_team"]).float().unsqueeze(0).unsqueeze(0)
            
#             h_in = h_out

#             state_dict_tensor = {
#               "player" : player_state,
#               "ball" : ball_state,
#               "left_team" : left_team_state,
#               "right_team" : right_team_state,
#               "hidden" : h_in
#             }
            
#             with torch.no_grad():
#                 prob, _, h_out = model(state_dict_tensor)
#             m = Categorical(prob)
#             a = m.sample().item()

#             prev_obs = obs
#             obs, rew, done, info = env.step(a)
#             additional_r = fe.calc_additional_reward(prev_obs[0], obs[0])
#             fin_r = rew*3.0 + additional_r
            
#             state_prime_dict = fe.encode(obs[0])
            
#             (h1_in, h2_in) = h_in
#             (h1_out, h2_out) = h_out
#             state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
#             state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())

#             transition = (state_dict, a, fin_r, state_prime_dict, prob[0][0][a].item(), done)
#             rollout.append(transition)

#             if len(rollout) == arg_dict["rollout_len"]:
#                 data_queue.put(rollout)
#                 rollout = []
                
#             state_dict = state_prime_dict

#             steps += 1
#             score += rew
#             tot_reward += fin_r

#             if done:
#                 if score > 0:
#                     win = 1
#                 summary_data = (win, score, tot_reward, steps)
#                 summary_queue.put(summary_data)
# #                 if n_epi % 4 == 0 and actor_num == 0:
# #                     print("%d, Done, Step %d win: %d, score: %d" % (n_epi, steps, win, score))
# #                     score = 0
# #                     win = 0


# def learner(center_model, queue, signal_queue, summary_queue, arg_dict):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = PPO(arg_dict["lstm_size"], arg_dict["k_epoch"], device)
#     model.load_state_dict(center_model.state_dict())
#     model.to(device)
#     cur_time = datetime.now() + timedelta(hours = 9)
#     log_dir = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S")
#     writer = SummaryWriter(logdir=log_dir)
#     optimization_step = 0
#     n_game = 0
    
#     print("learner start")
    
#     while True:
#         if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
#             if optimization_step % 500 == 0:
#                 torch.save(model.state_dict(), log_dir+"/model_"+str(optimization_step)+".pt")
#             signal_queue.put(1)
#             data = []
#             for j in range(arg_dict["buffer_size"]):
#                 mini_batch_np = []
#                 for i in range(arg_dict["batch_size"]):
#                     rollout = queue.get()
#                     mini_batch_np.append(rollout)
#                 mini_batch = model.make_batch(mini_batch_np)
#                 data.append(mini_batch)
#             model.train_net(data)
#             center_model.load_state_dict(model.state_dict())
            
#             optimization_step += arg_dict["batch_size"]*arg_dict["buffer_size"]*arg_dict["k_epoch"]
            
#             if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
#                 print(" data remaining. queue size : ", queue.qsize())
            
            
#             while summary_queue.qsize() > arg_dict["summary_game_window"]:
#                 win, score, tot_reward, game_len = [], [], [], []
                
                
#                 for i in range(arg_dict["summary_game_window"]):
#                     game_data = summary_queue.get()
#                     n_game += 1
#                     a,b,c,d = game_data
#                     win.append(a)
#                     score.append(b)
#                     tot_reward.append(c)
#                     game_len.append(d)
#                 writer.add_scalar('game/win', float(np.mean(win)), n_game)
#                 writer.add_scalar('game/score', float(np.mean(score)), n_game)
#                 writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
#                 writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
#                 writer.add_scalar('train/win', float(optimization_step), n_game)
                
#             _ = signal_queue.get()   
                    
            
#         else:
#             time.sleep(0.1)
    

if __name__ == '__main__':
    # hyperparameters
    arg_dict = {
        "num_processes": 8,
        "batch_size": 16,   
        "buffer_size": 6,
        "rollout_len": 10,
        "lstm_size" : 196,
        "k_epoch" : 3,
        "summary_game_window" : 5,
    }
    
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
        
        
