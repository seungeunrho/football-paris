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

def print_status(steps,a,m,a_prob,m_prob,obs,fin_r,tot_reward):
    readable_a=action_to_readable(a,m)
    ball_x, ball_y, _ = obs[0]['ball']
    [score_us, score_opponent] = obs[0]['score']
    if obs[0]['ball_owned_team'] == 1:
        owned_team = "OP"
    elif obs[0]['ball_owned_team'] == 0:
        owned_team = "ME"
    elif obs[0]['ball_owned_team'] == -1:
        owned_team = "GR"
    else:
        owned_team = "!!"
    game_mode = obs[0]['game_mode']
    result = "{} {}:{} {}({:2.2f}%), {} ball:{:.2f},{:.2f} r:{}/{:.2f}".format(steps,score_us,score_opponent, \
                                                           readable_a,a_prob*100,owned_team,ball_x,ball_y,fin_r,tot_reward)
    if game_mode != 0:
        result = result + mode_to_readable(game_mode)
    print(result)