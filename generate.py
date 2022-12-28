import numpy as np
import json

from visualize import visualize

# You can make all configurations at the Capitalized global variables.
# The program will generate x (= N_GENERATED_INSTANCES) images and corresponding json files into the /data folder.
# The image will have 512x512 pixels.

START_INDEX = 0
N_GENERATED_INSTANCES = 3
SEED = None  # set to any integer, if you want a fixed seed

MAX_TRIES = 1000
SPACE_SIZE = 512.0
DIMENSIONS = 2
START_X_MAX = 32.0
END_X_MIN = 480.0
AGENT_RADIUS = 4.0
N_OBSTACLES = 100
RADIUS_MIN = 3.0
RADIUS_MAX = 6.0


def export_to_json(spheres, index):
    d = {
        "start": [spheres[0][0].tolist(), spheres[0][1]],
        "goal": [spheres[1][0].tolist(), spheres[1][1]],
        "obstacles": [[s[0].tolist(), s[1]] for s in spheres[2:]]
    }

    with open(f"data/json/p{index:05d}.json", "w") as outfile:
        outfile.write(json.dumps(d, indent=4))

    return d


def sample_collision_free(rng, spheres, radius):
    n_try = 0
    while n_try < MAX_TRIES:
        n_try += 1
        pos = rng.uniform(radius, SPACE_SIZE - radius, size=DIMENSIONS)
        if has_collision(spheres, pos, radius):
            continue
        return pos
    # print("Couldn't sample new Point without collision!")
    return None

def sample_with_x_bias(rng, spheres, radius, x_min, x_max):
    n_try = 0
    while n_try < MAX_TRIES:
        n_try += 1
        x = rng.uniform(radius + x_min, x_max - radius)
        y = rng.uniform(radius, SPACE_SIZE - radius)
        pos = np.array([x, y])
        if has_collision(spheres, pos, radius):
            continue
        return pos
    # print("Couldn't sample new Point without collision!")
    return None


def has_collision(spheres, pos, radius):
    for o in spheres:
        dist = np.linalg.norm(o[0] - pos)
        if dist < radius + o[1]:
            return True
    return False


def generate_dataset(index, seed):
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed=seed)
    spheres = []
    start = sample_with_x_bias(rng, spheres, AGENT_RADIUS, 0.0, START_X_MAX)
    #start = sample_collision_free(rng, spheres, AGENT_RADIUS, start_bias=START_BIAS)
    spheres.append((start, AGENT_RADIUS))
    end = sample_with_x_bias(rng, spheres, AGENT_RADIUS, END_X_MIN, SPACE_SIZE)
    #end = sample_collision_free(rng, spheres, AGENT_RADIUS, end_bias=END_BIAS)
    spheres.append((end, AGENT_RADIUS))

    for i in range(N_OBSTACLES):
        # print('obstacle ' + str(i))
        r = rng.uniform(RADIUS_MIN, RADIUS_MAX)
        pos = sample_collision_free(rng, spheres, r)
        if pos is None:
            return False

        spheres.append((pos, r))

    data = export_to_json(spheres, index)
    return True, data


def main():
    index = START_INDEX
    while index < START_INDEX + N_GENERATED_INSTANCES:
        success, data = generate_dataset(index, SEED)
        if success:
            visualize(index, data, save_fig=True)
            index += 1
            print(f"{index - START_INDEX}: complete.")


def generate(start_index, N, n_seeds):
    index = start_index
    n_generated = N
    while index < start_index + n_generated:
        seed = n_seeds[index]
        success, data = generate_dataset(index, seed)
        if success:
            visualize(index, data, save_fig=True)
            index += 1
            print(f"{index - start_index}: complete.")


if __name__ == '__main__':
    main()
