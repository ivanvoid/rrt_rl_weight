import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import time
from tqdm import tqdm 

from RL_model import Model
from rrtstar import RRTStar

N_IMAGES = 100

###############################################################
# WEIGHTED RRT STAR
###############################################################

# generate test images
from generate import generate
# n_seeds = np.arange(N_IMAGES) + 420
# generate(0, N_IMAGES, n_seeds)

model = Model()
model.load_state_dict(torch.load('models/model_epoch_1.pth'))

def get_image(image_id):
    # read map.py
    name = None
    if image_id < 10:
        name = f'p0000{image_id}'
    elif image_id < 100:
        name = f'p000{image_id}'
    elif image_id < 999:
        name = f'p00{image_id}'
    
    image = Image.open(f'data/img/{name}.png').convert('RGB')
    image = np.array(image)
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    t_image = transform(image).unsqueeze(0)
    return image, t_image, name

costs = []
times = []
n_iters = []
for im_id in tqdm(range(N_IMAGES)):
    image, t_image, im_name = get_image(im_id)
    with torch.no_grad():
        weight_mu_std, value = model(t_image)
    weight_mu = weight_mu_std[:,0]
    weight_std = weight_mu_std[:,1]**2
    distribution = torch.distributions.Normal(weight_mu[0], weight_std[0])

    
    weight = distribution.sample()
    for i in range(99):
        weight += distribution.sample()
    weight /= 100
    weight = torch.sigmoid(weight)

    cond = np.all(np.array(image) == np.array([0,0,0]), 2)
    masked_weight = weight
    masked_weight[cond] = torch.tensor([0.]).repeat(masked_weight[cond].size()[0])
    
    p_weights = (masked_weight*100).round()

    # plt.figure();plt.imshow(p_weights, 'bwr',vmin=0, vmax=100);plt.colorbar()

    # bin all probabilities
    unique_bins = torch.unique(p_weights)
    probs_coords = {}
    for ub in unique_bins:
        if ub != 0: # if probability not 0
            coords = np.where(p_weights == ub)
            probs_coords[ub.item()] = [[tuple(c) for c in coords]] 
    

    st = time.time()

    for iter in range(1000):
        rrt = RRTStar()
        rrt.run_time_seconds = 0.1  # configure the time for each run
        rrt.load_environment(im_id)
        # Run with probability dictionary
        rrt.set_probability_map_from_dict(probs_coords)
        solution_cost, first_solution_at_iteration = rrt.run()
        if first_solution_at_iteration != -1:
            break

    et = time.time()
    elapsed_time = et - st
    
    times += [elapsed_time]
    costs += [solution_cost]
    n_iters += [iter]

print('Times: ',np.mean(times))
print('Costs: ', np.mean(costs))
print('N_iters: ', np.mean(n_iters))

exit()
###############################################################
# JUST RRT STAR
###############################################################
costs = []
times = []
for im_id in tqdm(range(N_IMAGES)):
# plt.figure();plt.imshow(p_weights, 'bwr',vmin=0, vmax=100);plt.colorbar()
    prob_map = np.full((512, 512), 1 / (512 ** 2))

    st = time.time()
    
    for iter in range(1000):
        rrt = RRTStar()
        rrt.run_time_seconds = 0.1  # configure the time for each run
        rrt.load_environment(im_id)
        # Run with probability dictionary
        rrt.set_probability_map_from_dict(probs_coords)
        solution_cost, first_solution_at_iteration = rrt.run()
        if first_solution_at_iteration != -1:
            break

    et = time.time()
    elapsed_time = et - st
    
    times += [elapsed_time]
    costs += [solution_cost]
    n_iters += [iter]

print('Times: ',np.mean(times))
print('Costs: ', np.mean(costs))
print('N_iters: ', np.mean(n_iters))