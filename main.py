import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
'''
parts of the code are taken and modified from
https://github.com/zhm-real/PathPlanning
'''

class Env:
    def __init__(self):
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
        obs_rectangle = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]

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
    env = Env()

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

    # Learning 

    # Evaluation

if __name__ == '__main__':
    main()