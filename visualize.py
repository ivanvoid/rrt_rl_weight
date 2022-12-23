import json

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

SPACE_SIZE = 512.0
START_COLOR = 'blue'
GOAL_COLOR = 'red'
OBSTACLE_COLOR = 'black'


def open_json(index):
    with open(f"data/json/p{index:05d}.json", 'r') as openfile:
        data = json.load(openfile)
    return data

def save(fig, index):
    plt.axis('off')
    fig.savefig(f"data/img/p{index:05d}.png", bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)


# Example if you want to visualize the RRT* solution path
def draw_path(ax, path):
    path = [[20.199602082091726, 16.05798364004058], [42.930465142298075, 38.58147947301789],
            [56.27292920716192, 67.66719164717338], [88.27144533326174, 67.97535674534258],
            [119.90288122660779, 63.13260305295437], [150.18835635292524, 52.79867851417399],
            [181.77497953769165, 57.92559140436205], [205.62208027828436, 79.26372127246489],
            [237.2571596330183, 74.44482573306091], [254.25107184397612, 101.5595255795696],
            [269.9451077299198, 129.44674913464425], [290.3184630564646, 154.12318308970242],
            [320.5392118284316, 164.64489129995323], [348.8394761790121, 179.58125758118985],
            [355.44019153547606, 210.89308532699448], [366.2414583981194, 241.01504798154212],
            [381.16384649990295, 269.3226853492164], [385.22386755395513, 301.06408149117357],
            [387.42851933555613, 332.98804600878765], [405.83819515208256, 359.1621522271555],
            [424.4975130665103, 372.66897280244405], [435.3818064214018, 402.36862679054343],
            [452.72232408644464, 415.0204257727008], [467.6927751306001, 438.1363902442602],
            [479.4790734938021, 451.059427990294], [493.17926252808655, 468.90145757838275],
            [496.3884094408318, 487.90265534682845]]

    for i in range(len(path) - 1):
        xs = [path[i][0], path[i + 1][0]]
        ys = [path[i][1], path[i + 1][1]]
        ax.plot(xs, ys, color='r')

    return ax

def draw_base_plot(ax, data):
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(0, SPACE_SIZE), ylim=(0, SPACE_SIZE))

    start = Circle(data['start'][0], data['start'][1], color=START_COLOR, alpha=1.0, zorder=2)
    ax.add_patch(start)

    # plot goal
    goal = Circle(data['goal'][0], data['goal'][1], color=GOAL_COLOR, alpha=1.0, zorder=0)
    ax.add_patch(goal)

    for o in data['obstacles']:
        c = Circle(o[0], o[1], color=OBSTACLE_COLOR, alpha=1.0)
        ax.add_patch(c)

    return ax


def visualize(index, data=None, save_fig=False):
    if data is None:
        data = open_json(index)

    fig, ax = plt.subplots(figsize=(6.65, 6.65), dpi=100)
    ax = draw_base_plot(ax, data)

    if save_fig:
        save(fig, index)
    else:
        plt.show()



def main():
    visualize(0, save_fig=False)


if __name__ == '__main__':
    main()
