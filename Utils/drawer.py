import os
import visdom

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

class Drawer():
    def __init__(self, server, idx=0, port=8097):
        self.name_env = f'football-paris_{idx}'
        self.viz = visdom.Visdom(port=port, server=server)
        self.viz.delete_env(self.name_env)
        
        self.fig = plt.figure(figsize=(12,6), dpi=5)
        self.fontsize = 0.5
        #self.ax = plt.axes(projection ="3d")

    def draw(self, obs):

        # 2d plot
        plt.cla()
        plt.xlim(-1.1, 1.1)
        plt.ylim(-0.5, 0.5)

        # overall line
        plt.plot([0, 0], 
                 [-0.42, 0.42], 
                 linewidth=0.5,
                 color='black')

        plt.plot([-1, -1], 
                 [-0.42, 0.42],
                 linewidth=0.5,
                 color='black')
        plt.plot([-1, 1], 
                 [0.42, 0.42],
                 linewidth=0.5,
                 color='black')
        plt.plot([1, 1], 
                 [0.42, -0.42],
                 linewidth=0.5,
                 color='black')
        plt.plot([1, -1], 
                 [-0.42, -0.42],
                 linewidth=0.5,
                 color='black')

        # left team penalty line
        plt.plot([-1, -0.78], 
                 [0.27, 0.27], 
                 linewidth=0.5,
                 color='black')
        plt.plot([-1, -0.78], 
                 [-0.27, -0.27], 
                 linewidth=0.5,
                 color='black')
        plt.plot([-0.78, -0.78], 
                 [-0.27, 0.27], 
                 linewidth=0.5,
                 color='black')

        # right team penalty line
        plt.plot([0.78, 1], 
                 [0.27, 0.27], 
                 linewidth=0.5,
                 color='black')
        plt.plot([0.78, 1.], 
                 [-0.27, -0.27], 
                 linewidth=0.5, 
                 color='black')
        plt.plot([0.78, 0.78], 
                 [-0.27, 0.27], 
                 linewidth=0.5, 
                 color='black')

        plt.scatter(obs['left_team'][:,0], 
                    obs['left_team'][:,1], 
                    color='red')
        for i in range(obs['left_team'].shape[0]):
            plt.text(obs['left_team'][i,0], 
                     obs['left_team'][i,1],
                     str(i))

        plt.scatter(obs['right_team'][:,0], 
                    obs['right_team'][:,1], 
                    color='blue')
        for i in range(obs['right_team'].shape[0]):
            plt.text(obs['right_team'][i,0], 
                     obs['right_team'][i,1],
                     str(i))


        plt.scatter([obs['ball'][0]],
                    [obs['ball'][1]],
                    color='orange' )

        self.viz.matplot(plt, env=self.name_env, win=0)