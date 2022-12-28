import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

class Node:
    def __init__(self, v, r):
        self.value = v
        self.root = r

class RRT:
    def __init__(self, state_space, goal_color, eps=0.1, step=1, max_iter=50000):
        # Exploration vs explotation ratio
        self.state_space = state_space
        self.eps = eps
        self.step = step
        self.max_iter = max_iter
        self.background_color = np.array([255,255,255])
        self.goal_color = goal_color

        self.is_goal_clear_flag = False

    def _sample(self):
        x = np.random.uniform(self.state_space[0,0], self.state_space[0,1])
        y = np.random.uniform(self.state_space[1,0], self.state_space[1,1])
        # round(x)
        return np.array([x,y]) 

    def _distance(self, a,b):
        part1 = (a[0] - b[0])**2
        part2 = (a[1] - b[1])**2
        return np.sqrt(part1 + part2)
    
    def _check_me(self, coords, image):
        # is vertex overlap
        color = image[coords[0],coords[1]]
        cond = np.all(color == self.background_color)
        return cond 

    def draw_line(self, start, finish):
        # can draw a line
        # all colors on this line are background_color?
        # y = mx + b     
        
        dx = finish[0] - start[0]
        dy = finish[1] - start[1]
        if dx == 0: 
            dx = -1

        slope = dy / dx # m
        intercept = start[1] - slope * start[0] # b

        if start[0] < finish[0]:
            coordinates = []
            for x in range(start[0], finish[0], 1):
                y = slope * x + intercept
                coordinates += [[x,y]]

        else:
            coordinates = []
            for x in range(finish[0], start[0], 1):
                y = slope * x + intercept
                coordinates += [[x,y]]
        
        coordinates = np.array(coordinates).astype(int)
        return coordinates

    def _check_edge_vertex_collison(self, start, finish, image):
        # this is out of bounds?
        if (finish[0] < 0) | (finish[1] < 0):
            return False
        if (finish[0] >= image.shape[0]) | (finish[1] >= image.shape[1]):
            return False

        # is vertex overlap
        color = image[finish[0],finish[1]]
        cond_vertex = np.all(color == self.background_color)
        if not cond_vertex:
            return cond_vertex # False

        # is edge overlap?
        coordinates = self.draw_line(start, finish)
        
        # coordinates = np.unique(coordinates, axis=0)
        edge_cond = []
        for xy in coordinates:
            # xy here fliped for some reason
            xy = np.array([xy[1], xy[0]])
            cond = self._check_me(xy, image)
            edge_cond.append(cond)
        cond_edge = np.all(edge_cond) & (len(edge_cond) > 1)#
        # plt.figure();plt.imshow(image);plt.plot(coordinates[:,0], coordinates[:,1]);plt.show()

        cond = cond_vertex & cond_edge
        return cond 
        
    def are_we_close(self, my_xy, goal_xy, distance=50):
        # if we close to the goal coordinates
        D = self._distance(my_xy, goal_xy)
        cond = D <= distance
        return cond 

    def compute_path(self, node_tree, xy_goal_hit):
        # add last coordinate that we check in another function 
        # that we can clearly see
        path = [xy_goal_hit]

        # list of (x,y) of how to go from start to finish
        # we are going from end to start
        last_node = node_tree[-1]

        path += [last_node.value]

        # I know it's not efficient 
        while not np.all(last_node.root == None):
            for i in range(len(node_tree)):
                node = node_tree[i]

                # if node is parent of a last node
                if np.all(last_node.root == node.value):
                    
                    last_node = node

                    path += [last_node.value]
        
        print('Close the window!')
        return np.array(path)

    def is_goal_visible(self, sample, goal, image):
        # this is just a modifyed _check_edge_vertex_collison() function
        # but here we wanna make sure that we intersect color of our goal
        # with no obsticles.
        coordinates = self.draw_line(sample, goal)

        # wall types 
        # 0 is background
        # 1 is a goal
        # 2 is a wall/obstacle

        wall_types = np.zeros(coordinates.size)
        xy_goal_hits = np.zeros((coordinates.size,2))
        for i, xy in enumerate(coordinates):
            # xy here fliped for some reason
            xy = np.array([xy[1], xy[0]])

            # cond = self._check_me(xy, image)
            color = image[xy[0],xy[1]]
            is_bg = np.all(color == self.background_color)
            is_goal = np.all(color == self.goal_color)

            if is_bg:
                wall_types[i] = 0
            elif is_goal:
                wall_types[i] = 1
                xy_goal_hits[i] = xy
            else:
                wall_types[i] = 2

        # if we can reach goal and have no obstacles on the way
        # then goal is visible
        is_goal_visible = False
        for i in range(coordinates.size):
            if wall_types[i] == 0:
                pass
            if wall_types[i] == 1:
                is_goal_visible = True
                xy = xy_goal_hits[i]
                return is_goal_visible, xy 
            if wall_types[i] == 2:
                return is_goal_visible, None # False, None

    def run(self, image, start, goal, obj_map):
        # start = [415, 346]
        path = [start]
        goal = goal.astype(int)
        node_tree = [Node(start, None)]

        for i in tqdm(range(self.max_iter)):

            # Sample 
            if np.random.uniform(0,1) < self.eps:
                sample = goal.astype(int)
            else:
                sample = self._sample().astype(int)
            
            cond_gen_existed = np.any(np.array(sample) == np.array(path))
            if cond_gen_existed:
                # generate another one
                continue

            if self._check_me(sample, image):                
                if len(path) >= 2:
                    X = np.array(path)
                    kdt = KDTree(X, leaf_size=30, metric='euclidean')
                    idx = kdt.query([sample], k=1, return_distance=False)[0]
                    neighbor = X[idx][0]
                else:
                    neighbor = path[-1]

                # STEER
                D = self._distance(neighbor, sample)

                new_x = neighbor[0] + (self.step/D) * (sample[0] - neighbor[0]) 
                new_y = neighbor[1] + (self.step/D) * (sample[1] - neighbor[1]) 
                new_sample = np.array([new_x, new_y]).astype(int)
                new_sample[0] = np.clip(new_sample[0], 
                    self.state_space[0,0], self.state_space[0,1])
                new_sample[1] = np.clip(new_sample[1], 
                    self.state_space[1,0], self.state_space[1,1])

                if self._check_edge_vertex_collison(
                        neighbor, new_sample, image):
                    path.append(new_sample)
                    node_tree += [Node(new_sample, neighbor)]

                    # after we add new sample check if we closer to our goal
                    if self.are_we_close(new_sample, goal):

                        goal_visible, xy_goal_hit = self.is_goal_visible(
                            new_sample, goal, image)
                        if goal_visible:
                            final_path = self.compute_path(
                                node_tree, xy_goal_hit)
                            break

        return final_path, path
        # plt.figure();plt.imshow(image);plt.scatter(np.array(path)[:,0], np.array(path)[:,1]);plt.show()
        # plt.figure();plt.imshow(image);plt.scatter(final_path[:,0], final_path[:,1]);plt.show()