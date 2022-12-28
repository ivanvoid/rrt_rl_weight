import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
from PIL import Image
'''
parts of the code are taken and modified from
https://github.com/zhm-real/PathPlanning
nvidia_srl
'''
from rrtstar import RRTStar
from RL_model import Model
    

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
    

def gen_data(start, goal, seed=None, render_me=False):
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
    
    parser.add_argument('--train_epochs', default=10)

    args = parser.parse_args()
    
    return args

def main():
    args = parse_me()

    # Generate data
    N_IMAGES = 3
    # from generate import generate
    # n_seeds = np.arange(3) + 420
    # generate(0, 3, n_seeds)

    # Learning
    model = Model()
    opt = torch.optim.Adam(model.parameters(), 1e-4)

    def get_image(image_id):
        # read map.py
        name = None
        if image_id < 10:
            name = f'p0000{image_id}'
        elif image_id < 100:
            name = f'p000{image_id}'
        
        image = Image.open(f'data/img/{name}.png').convert('RGB')
        image = np.array(image)
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        t_image = transform(image).unsqueeze(0)
        # print(t_image)


        return image, t_image, name

    # Train loop
    for epoch in range(args.train_epochs):
        for im_id in range(N_IMAGES):
            image, t_image, im_name = get_image(im_id)
            # TODO: Use RL to generate from image and start and end path weight map
        
            weight_mu_std, value = model(t_image)
            
            weight_mu = weight_mu_std[:,0]
            weight_std = weight_mu_std[:,1]**2
            # print(min(weight_mu), max(weight_mu))
            
            # w_img = weight_distribution.detach()[0,0]
            # plt.figure();plt.imshow(weight_mu.detach()[0]);plt.colorbar();plt.title("weight_mu");plt.show()
            # plt.imshow(weight_std.detach()[0]);plt.colorbar();plt.title("weight_std");plt.show()
            # plt.imshow(image);plt.imshow(weight_mu.detach()[0],alpha=0.5);plt.colorbar();plt.title("image and weight_mu");plt.show()
            # plt.imshow(image);plt.imshow(weight_std.detach()[0],alpha=0.5);plt.colorbar();plt.title("image and weight_std");plt.show()

            # print(weight_distribution, value)

            # ASSUME THAT WEIGHT IMAGE IS CORRECT
            # log_std = torch.nn.Parameter(torch.ones_like(w_img))
            # std   = log_std.exp().expand_as(w_img)
            distribution = torch.distributions.Normal(weight_mu[0], weight_std[0])
            # weight = torch.distributions.Normal(w_img, std)

            weight = distribution.sample()
            weight = torch.sigmoid(weight)
            # plt.figure();plt.imshow(weight.detach());plt.colorbar();plt.title("weight");plt.show()

            log_prob = distribution.log_prob(weight).mean()
            entropy  = distribution.entropy().mean()

            # print('log_prob: ', log_prob)

            # TODO: RRT-weighted evaluation here as reward function

            # Compute weight map
            cond = np.all(np.array(image) == np.array([0,0,0]), 2)
            masked_weight = weight
            masked_weight[cond] = torch.tensor([0.]).repeat(masked_weight[cond].size()[0])
            
            # plt.imshow(masked_weight);plt.show()
            
            # put all values into small bakets 
            p_weights = (masked_weight*100).round()


            plt.figure();plt.imshow(p_weights, 'bwr');plt.colorbar()
            plt.savefig(f'data/results/{im_name}_weight_epoch{epoch}.png')
            
            # bin all probabilities
            unique_bins = torch.unique(p_weights)
            probs_coords = {}
            for ub in unique_bins:
                if ub != 0: # if probability not 0
                    coords = np.where(p_weights == ub)
                    probs_coords[ub.item()] = [[tuple(c) for c in coords]] 
            
            all_rewards = []
            for i in range(100):
                rrt = RRTStar()
                rrt.run_time_seconds = 0.1  # configure the time for each run
                rrt.load_environment(0)
                # Run with probability dictionary
                rrt.set_probability_map_from_dict(probs_coords)
                solution_cost, first_solution_at_iteration = rrt.run()  # run and get reward
                # print(f"Solution Cost: {solution_cost}")
                # print(f"First Solution: {first_solution_at_iteration}")
                if first_solution_at_iteration != -1:
                    all_rewards += [solution_cost + first_solution_at_iteration]
            if len(all_rewards) > 0:
                mean_reward = np.sum(all_rewards)
                print('[',len(all_rewards),'] Mean Reward at this iteration: ', mean_reward)
            else:
                mean_reward = 0

            # print(probs_coords)

            # Then use this new sampler to select points at random with predetermin 
            # probability. And select coordinates at uniform from them.

            returns = mean_reward # many iterations of rewards

            # TODO: UPDATE
            advantage = returns - value
            actor_loss = -(log_prob * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            if loss < 0:
                print('loss < 0')

            print('ACTER LOSS: {:.4f}, CRITIC LOSS: {:.4f}, RL LOSS: {:4f}'.format(
                actor_loss, critic_loss, loss
            ))

            loss.backward()
            opt.step()

    # Evaluation

    # TODO: RRT-weighted evaluation here as reward function, 
    #       but as evaluation

if __name__ == '__main__':
    main()