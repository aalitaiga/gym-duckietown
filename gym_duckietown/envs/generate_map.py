from __future__ import print_function

import numpy as np



def generate_random_tile_map(n_center_points=5, size=10):
    map_ = np.ones([size, size]) * 0.5
    rows = np.random.choice(range(size), n_center_points, replace=False)
    cols = np.random.choice(range(size), n_center_points, replace=False)

    map_[rows, cols] = np.random.randint(0, 2, size=n_center_points)


def basic_map(size=10):
    map_ = np.zeros([size+1, size+1])
    map_[0,:] = 1
    map_[-1,:] = 1
    map_[:,0] = 1
    map_[:,-1] = 1

    rows = np.random.choice(range(1,size+1), 5)
    map_[rows, 1] = 1

    rows = np.random.choice(range(1,size+1), 6)
    map_[rows, 2] = 1
    map_[(map_[:,1] == 0) & (map_[:,2] == 1), 2] = 0

    rows = np.random.choice(range(1,size+1), 6)
    map_[rows, 4] = 1
    map_[(map_[:,3] == 0) & (map_[:,4] == 1), 4] = 0

    rows = np.random.choice(range(1,size+1), 5)
    map_[rows, 5] = 1

    rows = np.random.choice(range(1,size+1), 3)
    map_[rows, 6] = 1

    rows = np.random.choice(range(1,size+1), 5)
    map_[rows, 7] = 1
    map_[(map_[:,6] == 0) & (map_[:,7] == 1), 7] = 0

    rows = np.random.choice(range(1,size+1), 5)
    map_[rows, 9] = 1

    # import ipdb; ipdb.set_trace()
    return map_

def cleaner_map(size=10):
    map_ = np.zeros([size+1, size+1])
    map_[0,:] = 1
    map_[-1,:] = 1
    map_[:,0] = 1
    map_[:,-1] = 1

    previous = -1
    rows_to_modify = range(1,size)
    del rows_to_modify[size//3]
    # TODO: more interesting maps if commented but occassionaly non feasable
    # let's call them "noise" for now...@
    # if size // 3 != size // 2:
    #     del rows_to_modify[size//2]
    del rows_to_modify[-2]

    n_wall_row = [size // 2] * size

    for i, j in zip(rows_to_modify, n_wall_row):
        rows = np.random.choice(range(1,size+1), j)
        map_[rows, i] = 1

        if i == previous + 1:
            map_[(map_[:,previous] == 0) & (map_[:,i] == 1), i] = 0

        previous = i
    return [convert(row) for row in add_sprite(map_)]

def add_sprite(map):
    index = np.where(map == 0)
    choice = np.random.randint(len(index[0]))
    map[index[0][choice], index[1][choice]] = 2.0
    return map

def convert(row):
    l = []
    for num in row:
        if num == 1.0:
            l.append('#')
        elif num == 0.0:
            l.append(' ')
        elif num == 2.0:
            l.append('P')
    return ''.join(l)

if __name__ == '__main__':
    map = cleaner_map()
    for row in map:
        print(row)