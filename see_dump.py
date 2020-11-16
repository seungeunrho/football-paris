import pickle
import numpy as np

np.set_printoptions(precision=4)


def open_dump(path):
    res = None
    with open(path, 'rb') as f_dump:
        res = pickle.load(f_dump)
    return res

if __name__ == '__main__':
    args = {
            'num_actor': 0,
            'dir_log': './logs/[11-15]16.15.33/dumps'
            }
    num_actor = args['num_actor']
    path_dump = args['dir_log'] + f'/actor_{num_actor}.dump'

    # state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m
    dump = open_dump(path_dump)


    #print(dir(dump[0]))
    #print(len(dump[0]))
    #print(dump[-1][-3])
    print(dump[-1][-4])
    target = dump[-1][-4]
    for key in target.keys():
        obj = target[key]
        if not isinstance(obj, np.ndarray):
            print(f"{key}: {type(obj)}")
        else:
            exist_big = np.sum(obj > 1e+2)
            #print(f"{key}: {exist_big}")
            print(f"{key}: {obj}")



