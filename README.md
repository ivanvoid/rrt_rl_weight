# rrt_rl_weight
padm final project

![Probability Map Learning for RRT](https://github.com/ivanvoid/rrt_rl_weight/blob/main/figs/Probability%20Map%20Learning%20for%20RRT.png)

[Slides](https://docs.google.com/presentation/d/1C-S6DbEJoRkHchx5VedGsQ736_kZjxWA/edit?usp=sharing&ouid=107258873336279381104&rtpof=true&sd=true)

## Data Generation
In the generate.py program make configurations by changing the capitalized global variables.
You can configure the number of generated data instances, number of obstacles, etc.
The Program will generate 512x512 images as well as a corresponding json data file which will be used by RRT*.

## RRT*
See the main() method of rrtstar.py for an example.
