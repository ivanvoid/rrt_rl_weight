import json
import numpy as np
import time


def binary_search(value, lst):
    low = 0
    high = len(lst)

    while low != high:
        mid = int((low + high) / 2)
        if lst[mid] <= value:
            low = mid + 1
        else:
            high = mid

    return low

def k_nearest_neighbor(x, tree, k):
    if len(tree) == 1:
        return [0]
    difference = tree[:, 2:4] - x
    dist = np.linalg.norm(difference, axis=1)
    if k == 1:  # easy case optimization
        return [np.argmin(dist)]
    idx = np.argpartition(dist, k) if k < len(tree) else np.arange(k)

    # idx = idx[:k].tolist()
    # idx.sort(key=lambda e: dist[e])
    # return np.array(idx)

    return idx


def steer(x1, x2, scale):
    return x1 + scale * (x2 - x1)


def distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def create_solution_path(tree, goal_index):
    path = [tree[goal_index, 2:4].tolist()]
    parent = int(tree[goal_index, 1])

    while parent > 0:
        path.append(tree[parent, 2:4].tolist())
        parent = int(tree[parent, 1])
    path.append(tree[0, 2:4].tolist())

    path.reverse()
    return path


class RRTStar:
    def __init__(self):
        self.goal_bias = 10  #  every nth iteration the goal is sampled
        self.space_size = 512.0
        self.start = None  # set by the json configuration [x, y]
        self.goal = None  # set by the json configuration [x, y]
        self.obstacles = None  # set by the json configuration [x, y, distance threshold]
        self.rng = np.random.default_rng()
        self.range = 32.0  # max distance between two points
        self.interpolation_steps = 4  # granularity of collision check
        self.k_rrg = 1.5 * np.e  # see RRT* paper
        self.range_threshold_factor = 1.001
        self.run_time_seconds = 0.5  # how long a single rrt run will take
        self.prob_map = None  # call set_probability_map() method
        self.cumulative_prob_x = None # call set_probability_map() method

    def set_probability_map(self, prob_map):
        self.prob_map = prob_map
        marginal_prob_x = np.sum(prob_map, axis=0)
        self.cumulative_prob_x = np.zeros(len(marginal_prob_x))
        self.cumulative_prob_x[0] = marginal_prob_x[0]
        for i in range(1, len(marginal_prob_x)):
            self.cumulative_prob_x[i] = self.cumulative_prob_x[i - 1] + marginal_prob_x[i]


    def load_environment(self, index):
        with open(f"data/json/p{index:05d}.json", "r") as infile:
            dct = json.load(infile)

        self.start = np.array(dct['start'][0])
        self.goal = np.array(dct['goal'][0])
        agent_radius = dct['start'][1]
        obs = dct['obstacles']
        ob = obs[0]
        x = np.array([ob[0][0], ob[0][1], ob[1] + agent_radius])
        self.obstacles = x[None, :]
        for ob in obs[1:]:
            x = np.array([ob[0][0], ob[0][1], ob[1] + agent_radius])
            self.obstacles = np.vstack((self.obstacles, x))

    # point collision check
    def is_collision_free(self, point):
        difference = self.obstacles[:, :2] - point
        dist = np.linalg.norm(difference, axis=1)
        margin = dist - self.obstacles[:, 2]

        return np.all(margin > 0)

    def get_cumulative_probability(self, index):
        p_y = self.prob_map[:, index]
        conditional_probability_y = p_y / np.sum(p_y)
        cumulative_prob_y = np.zeros(len(conditional_probability_y))
        cumulative_prob_y[0] = conditional_probability_y[0]
        for i in range(1, len(conditional_probability_y)):
            cumulative_prob_y[i] = cumulative_prob_y[i - 1] + conditional_probability_y[i]

        return cumulative_prob_y

    # legacy function
    def sample_random(self):
        while True:
            p = self.rng.uniform(0, self.space_size, 2)
            if self.is_collision_free(p):
                return p

    # sample according to the given probability map
    def sample_with_probability_map(self):
        while True:
            # sample x
            val = self.rng.random()
            index = binary_search(val, self.cumulative_prob_x)
            left_bound = self.space_size / len(self.cumulative_prob_x) * index
            right_bound = self.space_size / len(self.cumulative_prob_x) * (index + 1)
            x = self.rng.uniform(left_bound, right_bound)

            # construct conditional probability for y
            cumulative_prob_y = self.get_cumulative_probability(index)

            # sample y
            val = self.rng.random()
            index = binary_search(val, cumulative_prob_y)
            index = -1 * (index - len(cumulative_prob_y) + 1)  # y coordinates start at the bottom
            left_bound = self.space_size / len(self.cumulative_prob_x) * index
            right_bound = self.space_size / len(self.cumulative_prob_x) * (index + 1)
            y = self.rng.uniform(left_bound, right_bound)

            p = np.array([x, y])
            if self.is_collision_free(p):
                return p

    def new_state(self, x_rand, x_near):
        dist = distance(x_rand, x_near)
        if dist <= self.range:
            return x_rand

        scale = self.range / dist
        x_new = steer(x_near, x_rand, scale)

        return x_new

    # line collision check
    def path_is_free(self, x1, x2, n_steps):
        if n_steps == 0:
            return True
        if np.allclose(x1, x2):
            return True

        mid = steer(x1, x2, 0.5)
        if self.is_collision_free(mid):
            return self.path_is_free(x1, mid, n_steps - 1) and self.path_is_free(mid, x2, n_steps - 1)

        return False

    def connect(self, tree, p_new, nearest_index, k_nearest_idx):
        index_min = nearest_index
        cost_min = tree[nearest_index, 4] + distance(tree[index_min, 2:4], p_new)

        for i in k_nearest_idx:
            dist = distance(tree[i, 2:4], p_new)
            if dist > self.range_threshold_factor * self.range:
                continue
            cost = tree[i, 4] + dist
            if cost < cost_min and self.path_is_free(tree[i, 2:4], p_new, self.interpolation_steps):
                cost_min = cost
                index_min = i

        x_new = np.array([len(tree), index_min, p_new[0], p_new[1], cost_min])

        return x_new

    # goal is resampled
    def connect_goal(self, tree, x_goal, k_nearest_idx):
        cost_min = x_goal[4]
        index_min = x_goal[0]
        for i in k_nearest_idx:
            if np.allclose(x_goal, tree[i]):
                continue
            dist = distance(tree[i, 2:4], x_goal[2:4])
            if dist > self.range_threshold_factor * self.range:
                continue
            cost = tree[i, 4] + dist
            if cost < cost_min and self.path_is_free(tree[i, 2:4], x_goal[2:4], self.interpolation_steps):
                cost_min = cost
                index_min = i

        if cost_min < x_goal[4]:
            goal_index = int(x_goal[0])
            tree[goal_index, 1] = index_min
            tree[goal_index, 4] = cost_min

        return tree


    def rewire(self, tree, x_new, k_nearest_idx):
        for i in k_nearest_idx:
            dist = distance(x_new[2:4], tree[i, 2:4])
            if dist > self.range_threshold_factor * self.range:
                continue
            cost = x_new[4] + dist
            if cost < tree[i, 4] and self.path_is_free(x_new[2:4], tree[i, 2:4], self.interpolation_steps):
                tree[i, 1] = x_new[0]  # change parent index
                tree[i, 4] = cost  # change cost

        return tree


    def run_single_iteration(self, tree, iteration, goal_index):
        solution_cost = float('inf')
        if iteration % self.goal_bias == 0:
            p_rand = self.goal
        else:
            #p_rand = self.sample_random()
            p_rand = self.sample_with_probability_map()
        nearest_index = k_nearest_neighbor(p_rand, tree, 1)[0]

        p_near = tree[nearest_index, 2:4]
        p_new = self.new_state(p_rand, p_near)
        is_goal = np.allclose(self.goal, p_new)
        is_free = self.path_is_free(p_near, p_new, self.interpolation_steps)
        if is_free:
            k = min(len(tree), np.ceil(self.k_rrg * np.log(len(tree))).astype(int))
            k_nearest_idx = k_nearest_neighbor(p_new, tree, k)
            if is_goal and goal_index >= 0:
                tree = self.connect_goal(tree, tree[goal_index], k_nearest_idx)
                x_new = tree[goal_index]
            else:
                x_new = self.connect(tree, p_new, nearest_index, k_nearest_idx)
                tree = np.vstack((tree, x_new))

            if is_goal:
                solution_cost = x_new[4]
            else:
                tree = self.rewire(tree, x_new, k_nearest_idx)

        return tree, solution_cost

    # return the path length of the best solution, the iteration where the first solution was found,
    #   if a solution was found
    # return float('inf), -1 otherwise
    def run(self):
        start_time = time.time()
        # tree: self index, parent index, x, y, cost
        start_entry = np.array([0.0, 0.0, self.start[0], self.start[1], 0.0])
        tree = start_entry[None, :]
        goal_index = -1
        iteration = 0
        solution_cost = float('inf')
        first_solution_at_iteration = -1
        while time.time() - start_time < self.run_time_seconds:
            tree, it_solution_cost = self.run_single_iteration(tree, iteration, goal_index)
            solution_cost = min(solution_cost, it_solution_cost)
            if it_solution_cost < float('inf') and goal_index < 0:
                goal_index = len(tree) - 1
                first_solution_at_iteration = iteration
            iteration += 1

        # print(f"Solution Cost: {solution_cost}")
        # print(f"Total Iterations: {iteration}")
        # print(f"First Solution: {first_solution_at_iteration}")
        #
        # path = create_solution_path(tree, goal_index)
        # print(path)

        return solution_cost, first_solution_at_iteration


def main():
    # example probability maps
    prob_map1 = np.full((512, 512), 1 / (512 ** 2))
    prob_map2 = np.random.default_rng().random(size=(512, 512))
    prob_map2 /= np.sum(prob_map2)

    rrt = RRTStar()
    rrt.run_time_seconds = 0.2  # configure the time for each run
    rrt.load_environment(1)  # load environment by index

    # Run with probability map
    print("\nProbability Map 1")
    rrt.set_probability_map(prob_map1)  # set probability map
    solution_cost, first_solution_at_iteration = rrt.run()  # run and get reward
    print(f"Solution Cost: {solution_cost}")
    print(f"First Solution: {first_solution_at_iteration}")

    # Run with different probability map
    print("\nProbability Map 2")
    rrt.set_probability_map(prob_map2)  # set probability map
    solution_cost, first_solution_at_iteration = rrt.run()  # run and get reward
    print(f"Solution Cost: {solution_cost}")
    print(f"First Solution: {first_solution_at_iteration}")

if __name__ == '__main__':
    main()  # example for correct usage