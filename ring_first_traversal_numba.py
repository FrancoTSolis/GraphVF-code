import os
import random
import numpy as np
from numba import njit, jit
from numba.typed import List
from numba.types import int32, unicode_type, ListType

config_Floyd_INF = 100 # No edge, infinite distance.

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

@njit
def set_seed(value):
    random.seed(1234)
    np.random.seed(value)

set_seed(1234)
seed_torch(1234)

@njit
def DepthFirstTraversal(n_atoms, dist_mat, order_lst, cur_idx, last_idx):
    # No ring detected. The traversal is on a tree.
    order_lst.append(cur_idx)
    vis_lst = np.arange(n_atoms)
    np.random.shuffle(vis_lst)
    for idx in vis_lst:
        if (idx != cur_idx) and (idx != last_idx) and (dist_mat[cur_idx][idx] < config_Floyd_INF):
            DepthFirstTraversal(n_atoms, dist_mat, order_lst, idx, cur_idx)


@njit
def IterFloyd(n_atoms, dist_mat, start_atom):
    # Find the closest ring to the starting atom with Floyd.
    # If multiple rings satisfy the condition, select one of the smallest.

    # Firstly, perform normal Floyd update.
    min_dist_mat = dist_mat.copy()
    for idx2 in range(n_atoms):
        for idx0 in range(n_atoms):
            for idx1 in range(n_atoms):
                if min_dist_mat[idx0, idx2] + min_dist_mat[idx2, idx1] < min_dist_mat[idx0, idx1]:
                    min_dist_mat[idx0, idx1] = min_dist_mat[idx0, idx2] + min_dist_mat[idx2, idx1]

    # Modify Floyd to find the closest ring.
    start_dist = min_dist_mat[start_atom]
    start_dist_mat = np.zeros(dist_mat.shape, dtype=int32)
    for idx0 in range(n_atoms):
        for idx1 in range(n_atoms):
            if idx0 != idx1:
                if dist_mat[idx0, idx1] < config_Floyd_INF:
                    start_dist_mat[idx0, idx1] = min(start_dist[idx0], start_dist[idx1])
                else:
                    start_dist_mat[idx0, idx1] = config_Floyd_INF
    cur_dist_mat = dist_mat.copy()
    
    last_visit = [[[-2] for idx0 in range(n_atoms)] for idx1 in range(n_atoms)]
    min_ring_dist = config_Floyd_INF
    min_ring_len = config_Floyd_INF
    for idx2 in range(n_atoms):
        for idx0 in range(idx2):
            for idx1 in range(idx0 + 1, idx2):
                if (cur_dist_mat[idx0, idx1] + dist_mat[idx0, idx2] + dist_mat[idx1, idx2] < config_Floyd_INF) and \
                   ((min(start_dist_mat[idx0, idx1], start_dist[idx2]) < min_ring_dist) or \
                   ((min(start_dist_mat[idx0, idx1], start_dist[idx2]) == min_ring_dist) and \
                    (cur_dist_mat[idx0, idx1] + dist_mat[idx0, idx2] + dist_mat[idx1, idx2] < min_ring_len))) and \
                   (len(set([idx0, idx1, idx2]).intersection(set(last_visit[idx0][idx1][1:]))) == 0):
                    # A ring is detected.
                    min_ring_dist = min(start_dist_mat[idx0, idx1], start_dist[idx2])
                    min_ring_len = cur_dist_mat[idx0, idx1] + dist_mat[idx0, idx2] + dist_mat[idx1, idx2]
                    min_ring_lst = last_visit[idx0][idx1][1:] + [idx1, idx2, idx0]
                    if random.randint(0, 1) == 0:
                        min_ring_lst.reverse()
                    
        for idx0 in range(n_atoms):
            for idx1 in range(n_atoms):
                if (idx0 != idx1) and (idx0 != idx2) and (idx1 != idx2) and \
                   (cur_dist_mat[idx0, idx2] + cur_dist_mat[idx2, idx1] < config_Floyd_INF) and \
                   ((min(min(start_dist_mat[idx0, idx2], start_dist_mat[idx2, idx1]), start_dist[idx2]) < start_dist_mat[idx0, idx1]) or \
                   ((min(min(start_dist_mat[idx0, idx2], start_dist_mat[idx2, idx1]), start_dist[idx2]) == start_dist_mat[idx0, idx1]) and \
                    (cur_dist_mat[idx0, idx2] + cur_dist_mat[idx2, idx1] < cur_dist_mat[idx0, idx1]))) and \
                   (len(set([idx0, idx1, idx2]).intersection(set(last_visit[idx0][idx2][1:] + last_visit[idx2][idx1][1:]))) == 0) and \
                   (len(set(last_visit[idx0][idx2][1:]).intersection(set(last_visit[idx2][idx1][1:]))) == 0):
                    # Modified Floyd update.
                    start_dist_mat[idx0, idx1] = min(min(start_dist_mat[idx0, idx2], start_dist_mat[idx2, idx1]), start_dist[idx2])
                    cur_dist_mat[idx0, idx1] = cur_dist_mat[idx0, idx2] + cur_dist_mat[idx2, idx1]
                    last_visit[idx0][idx1][1:] = last_visit[idx0][idx2][1:] + [idx2] + last_visit[idx2][idx1][1:]

    # No ring detected. End interation.
    if min_ring_dist == config_Floyd_INF:
        depth_first_order = [-2, ]
        DepthFirstTraversal(n_atoms, dist_mat, depth_first_order, start_atom, -1)
        return depth_first_order[1:]

    # Otherwise, replace the ring with a new node indexed 0 and continue iteration. So the distance matrix is to be modified.
    tot_new_node = 1
    new_node_map = np.zeros((n_atoms, ), dtype=int32)
    rev_node_map = [-1]
    for idx in range(n_atoms):
        if idx in min_ring_lst:
            new_node_map[idx] = 0
        else:
            new_node_map[idx] = tot_new_node
            rev_node_map.append(idx)
            tot_new_node = tot_new_node + 1
    new_dist_mat = np.ones((tot_new_node, tot_new_node), dtype=int32) * config_Floyd_INF
    for idx0 in range(n_atoms):
        for idx1 in range(n_atoms):
            new_idx0 = new_node_map[idx0]
            new_idx1 = new_node_map[idx1]
            new_dist_mat[new_idx0, new_idx1] = min(new_dist_mat[new_idx0, new_idx1], dist_mat[idx0, idx1])
    ring_first_order = IterFloyd(tot_new_node, new_dist_mat, int(new_node_map[start_atom]))

    # Finally, combine the order of the ring and the remaining part.
    for pos, idx in enumerate(min_ring_lst):
        if min_ring_dist == start_dist[idx]:
            min_ring_lst = min_ring_lst[pos:] + min_ring_lst[:pos]
            break
    combined_ring_first_order = [-3, ]
    for idx in ring_first_order:
        if idx != 0:
            combined_ring_first_order.append(rev_node_map[idx])
        else:
            combined_ring_first_order = combined_ring_first_order + min_ring_lst
    return combined_ring_first_order[1:]

lig_n_atoms = 36
lig_edge_lst = [
    (0, 1, 2),
    (1, 2, 1),
    (2, 3, 1),
    (3, 4, 1),
    (4, 5, 1),
    (1, 5, 1),
    (5, 6, 2),
    (6, 7, 1),
    (7, 8, 2),
    (8, 9, 1),
    (4, 9, 2),
    (9, 10, 1),
    (10, 11, 1),
    (8, 12, 1),
    (12, 13, 1),
    (12, 14, 1),
    (14, 15, 1),
    (15, 16, 1),
    (16, 17, 1),
    (17, 18, 1),
    (18, 19, 1),
    (19, 20, 1),
    (15, 20, 1),
    (18, 21, 1),
    (21, 22, 1),
    (22, 23, 1),
    (23, 24, 1),
    (18, 24, 1),
    (24, 25, 2),
    (23, 26, 1),
    (26, 27, 1),
    (27, 28, 2),
    (28, 29, 1),
    (29, 30, 2),
    (29, 31, 1),
    (31, 32, 1),
    (26, 32, 2),
    (31, 33, 1),
    (33, 34, 1),
    (33, 35, 1),
]

lig_con_mat = np.zeros((lig_n_atoms, lig_n_atoms), dtype=int) # 默认是float，会影响程序行为（有样例）
for x, y, z in lig_edge_lst:
    lig_con_mat[x][y] = z
    lig_con_mat[y][x] = z
print(lig_con_mat)

# prepare for Floyd
@njit
def con2dist(lig_con_mat): 
    lig_dist_mat = np.zeros((lig_n_atoms, lig_n_atoms), dtype=int32)
    for idx0 in range(lig_n_atoms):
        for idx1 in range(lig_n_atoms):
            if idx0 != idx1:
                if lig_con_mat[idx0, idx1] > 0:
                    lig_dist_mat[idx0, idx1] = 1
                else:
                    lig_dist_mat[idx0, idx1] = config_Floyd_INF # no edge
    return lig_dist_mat

lig_dist_mat = con2dist(lig_con_mat)

# ring-first traversal
# start_atom = 15
start_atom = random.randint(0, lig_n_atoms - 1)
standard_ring_first_order = IterFloyd(lig_n_atoms, lig_dist_mat, start_atom)
print(standard_ring_first_order)

# focal list & neighbourhood priority
focal_order = [-1]
ring_first_order = [start_atom]
ring_edges = []

vis = [0 for _ in range(lig_n_atoms)]
vis[start_atom] = 1
remaining_edges = lig_con_mat.copy()
for idx in standard_ring_first_order:
    for new_idx in range(lig_n_atoms):
        if remaining_edges[idx][new_idx] != 0:
            if vis[new_idx] == 1:
                ring_edges.append((idx, new_idx))
            else:
                vis[new_idx] = 1
                focal_order.append(idx)
                ring_first_order.append(new_idx)
            remaining_edges[idx][new_idx] = 0
            remaining_edges[new_idx][idx] = 0
print(focal_order)
print(ring_first_order)
print(ring_edges)
