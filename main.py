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


def plot(env, start, goal, render_me):
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
                facecolor='black',
                fill=True
            )
        )

    for (ox, oy, r) in env.obs_circle:
        ax.add_patch(
            patches.Circle(
                (ox, oy), r,
                edgecolor='black',
                facecolor='black',
                fill=True
            )
        )

    ax.add_patch(
        patches.Circle(
            (start[0], start[1]), 1,
            edgecolor='blue',
            facecolor='blue',
            fill=True
        ))
    ax.add_patch(
        patches.Circle(
            (goal[0], goal[1]), 1,
            edgecolor='red',
            facecolor='red',
            fill=True
        ))
    plt.axis("equal")
    ax.set_axis_off()
    fig.savefig('map.png')#, bbox_inches='tight', pad_inches=0)
    if render_me:
        plt.show()
    

def gen_data(start, goal, seed=None, render_me=True):
    # get random seed
    if seed is None:
        _seed = np.random.randint(0,1000)
    else:
        _seed = seed

    # create env
    env = Env(_seed)

    print(_seed)
    plot(env, start, goal, render_me)

    return env 


def parse_me():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--arg')

    args = parser.parse_args()
    
    return args

def main():
    args = parse_me()

    # Set start and end position
    start = (3, 3)  # Starting node
    goal = (49, 27)  # Goal node

    # Generate data
    env = gen_data(start, goal, 420, False)

    # Learning
    # read map.py
    import torchvision.transforms as transforms
    from PIL import Image
    image = Image.open('map.png').convert('RGB')

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    t_image = transform(image).unsqueeze(0)
    # t_image = torch.tensor(image).float()

    # TODO: Use RL to generate from image and start and end path weight map
    from RL_model import Model
    model = Model()
    weight = model(t_image)

    # TODO: RRT-weighted evaluation here as reward function


    # Evaluation

    # TODO: RRT-weighted evaluation here as reward function, 
    #       but as evaluation

if __name__ == '__main__':
    main()