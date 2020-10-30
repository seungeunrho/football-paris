import numpy as np

def action_to_readable(a, m):
    if a==0:
        return "NO_OP "
    elif a==1:
        if m == 0:   
            return "\u2190\u2190  "  #left
        elif m == 1:  
            return "\u2191\u2190  "  #top_left
        elif m == 2:  
            return "\u2191\u2191  "  #top
        elif m == 3:  
            return "\u2191\u2190  "  #top_right
        elif m == 4:  
            return "\u2192\u2192  "  #right
        elif m == 5:  
            return "\u2193\u2190  "  #bot-right
        elif m == 6:  
            return "\u2193\u2193  "  #bot
        elif m == 7:  
            return "\u2193\u2190  "  #bot-left

    elif a==2:
        return "PASS_L"
    elif a==3:
        return "PASS_H"
    elif a==4:
        return "PASS_S"
    elif a==5:
        return "SHOT  "
    elif a==6:
        return "SPRINT"
    elif a==7:
        return "STOP_M"
    elif a==8:
        return "STOP_S"
    elif a==9:
        return "SLIDE "
    elif a==10:
        return "DRIBLE"
    elif a==11:
        return "STOP_D"

def mode_to_readable(game_mode):
    if game_mode==1:
        return 'KickOff'
    elif game_mode==2:
        return 'GoalKick'
    elif game_mode==3:
        return 'FreeKick'
    elif game_mode==4:
        return 'Corner'
    elif game_mode==5:
        return 'ThrowIn'
    elif game_mode==6:
        return 'Penalty'

def print_status(steps,a,m,a_prob,m_prob,obs,next_obs,fin_r,tot_reward):
    readable_a=action_to_readable(a,m)
    active_player = obs[0]['active']
    me_x, me_y = obs[0]['left_team'][active_player]
    ball_x, ball_y, _ = obs[0]['ball']
    [score_us, score_opponent] = obs[0]['score']
    
    
    left_yellow = np.sum(next_obs[0]["left_team_yellow_card"]) -  np.sum(obs[0]["left_team_yellow_card"])
    right_yellow = np.sum(next_obs[0]["right_team_yellow_card"]) -  np.sum(obs[0]["right_team_yellow_card"])
    yellow_r = right_yellow - left_yellow
    if yellow_r > 0:
        yellow_result = "OPP YELLOW"
    elif yellow_r == 0:
        yellow_result = ""
    else:
        yellow_result = "ME YELLOW"
        
    
    sticky_sprnt = obs[0]['sticky_actions'][8]
    sticky_dribble = obs[0]['sticky_actions'][9]
    
    if sticky_dribble==1:
        is_dribbling = "DRB"
    else:
        is_dribbling = "   "
        
    if sticky_sprnt == 1:
        is_sprnt = "RUN "
    else:
        is_sprnt = "WALK"
        
    if obs[0]['ball_owned_team'] == 1:
        owned_team = "OP"
    elif obs[0]['ball_owned_team'] == 0:
        owned_team = "ME"
    elif obs[0]['ball_owned_team'] == -1:
        owned_team = "GR"
    else:
        owned_team = "!!"
    game_mode = obs[0]['game_mode']
    tired = obs[0]['left_team_tired_factor'][active_player]
    result = "{} {}:{} {}({:2.2f}%), {} ball:{:.2f},{:.2f} me({}{}{:.2f}):{:.2f},{:.2f} r:{}/{:.2f}".format(steps,score_us,score_opponent, \
                               readable_a,a_prob*100,owned_team,ball_x,ball_y,active_player,is_sprnt,tired,me_x,me_y,fin_r,tot_reward)
    if game_mode != 0:
        result = result + mode_to_readable(game_mode)
    if yellow_r != 0:
        result = result + yellow_result
    print(result)