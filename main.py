import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
'''
parts of the code are taken and modified from
https://github.com/zhm-real/PathPlanning
'''

class Env:
    # TODO: generate env based on seed
    def __init__(self, seed):
        np.random.seed(seed)
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 30],
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = []
        for _ in range(np.random.randint(2,5)):
            a = np.random.randint(14,32)
            b = np.random.randint(7,22)
            c = np.random.randint(2,10)
            d = np.random.randint(2,12)
            obs_rectangle += [[a,b,c,d]]

        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = []
        for _ in range(np.random.randint(4,6)):
            a = np.random.randint(7,46)
            b = np.random.randint(7,23)
            c = np.random.randint(2,3)
            obs_cir += [[a,b,c]]

        return obs_cir

def plot(env):
    fig, ax = plt.subplots()

    for (ox, oy, w, h) in env.obs_boundary:
        ax.add_patch(
            patches.Rectangle(
                (ox, oy), w, h,
                edgecolor='black',
                facecolor='black',
                fill=True
            )
        )

    for (ox, oy, w, h) in env.obs_rectangle:
        ax.add_patch(
            patches.Rectangle(
                (ox, oy), w, h,
                edgecolor='black',
                facecolor='gray',
                fill=True
            )
        )

    for (ox, oy, r) in env.obs_circle:
        ax.add_patch(
            patches.Circle(
                (ox, oy), r,
                edgecolor='black',
                facecolor='gray',
                fill=True
            )
        )

    plt.axis("equal")
    plt.show()

def gen_data():
    _seed = np.random.randint(0,1000)
    print(_seed)
    env = Env(_seed)

    plot(env)



def parse_me():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--arg')

    args = parser.parse_args()
    
    return args

def main():
    args = parse_me()
    
    # Generate data
    gen_data()

    # TODO: set start and end points
    #


    # Learning 
    # TODO: Use RL to generate from image and start and end path weight map

    # TODO: RRT-weighted evaluation here as reward function


    # Evaluation

    # TODO: RRT-weighted evaluation here as reward function, 
    #       but as evaluation

if __name__ == '__main__':
    main()