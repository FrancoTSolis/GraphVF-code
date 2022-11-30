import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import pickle as pkl

import networkx as nx
from networkx.algorithms import tree
from math import pi
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from Bio.PDB import PDBParser
import random
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from numba.types import int32
from numba import njit
from fast_jtnn import MolTree, Vocab, tensorize_moltree

# lig_elem_range = [
#        #B, C, N, O, F, Mg, Al, Si, P,  S, Cl, Sc, V, Fe, Cu, Zn, As, Se, Br, Y, Mo, Ru, Rh, Sb, I, W, Au
#         5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 21, 23, 26, 29, 30, 33, 34, 35, 39, 42, 44, 45, 51, 53, 74, 79
#     ]
# rec_elem_range = [
#        #C, N, O, Na, Mg,  P,  S, Cl,  K, Ca, Mn, Co, Cu, Zn, Se, Cd, I, Cs, Hg
#         6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 25, 27, 29, 30, 34, 48, 53, 55, 80
#     ]



atomic_num_to_type = {5:0, 6:1, 7:2, 8:3, 9:4, 12:5, 13:6, 14:7, 15:8, 16:9, 17:10, 21:11, 23:12, 26:13, 29:14, 30:15, 33:16, 34:17, 35:18, 39:19, 42:20, 44:21, 45:22, 51:23, 53:24, 74:25, 79:26}

atomic_element_to_type = {'C':27, 'N':28, 'O':29, 'NA':30, 'MG':31, 'P':32, 'S':33, 'CL':34, 'K':35, 'CA':36, 'MN':37, 'CO':38, 'CU':39, 'ZN':40, 'SE':41, 'CD':42, 'I':43, 'CS':44, 'HG':45}

# covert an adjacency matrix of bonds to an adjacency table
def con2edge(lig_con_mat, rec_n_atoms, lig_n_atoms):
    edge_lst = []
    for idx0 in range(lig_n_atoms):
        for idx1 in range(lig_n_atoms):
            if idx0 != idx1:
                if lig_con_mat[idx0, idx1] > 0:
                    tot = 0
                    for cur in range(lig_n_atoms):
                        if cur > max(idx0, idx1):
                            edge_lst.append((tot + idx0, tot + idx1, lig_con_mat[idx0, idx1]))
                        tot = tot + cur + rec_n_atoms
    return sorted(edge_lst)


'''
Ring-first traversal is applied during training.
For a 2D ligand molecular graph, a ring-first traversal is a sequential order of generating nodes in the graph. It is used for the teacher-forcing training.
To be specific, a ring-first traversal visits each node once in the graph. It tries to keep that rings in the graph are visited consecutively.
The ring-first traversal of trees are identical to the depth-first traversal, for there is no rings in trees.
For graphs that contain rings, we iteratively apply Floyd algorithm to find a closest ring of minimum length and collapse them into single nodes, until there is no ring in the graph.
Then we initialize ring-first traversal as depth-first traversal, and restore the rings and insert them to the ring-first traversal one by one in the reverse order of collapsions.
'''

config_Floyd_INF = 1000 # No edge, infinite distance.

# prepare for Floyd
@njit
def con2dist(lig_con_mat, lig_n_atoms): 
    lig_dist_mat = np.zeros((lig_n_atoms, lig_n_atoms), dtype=int32)
    for idx0 in range(lig_n_atoms):
        for idx1 in range(lig_n_atoms):
            if idx0 != idx1:
                if lig_con_mat[idx0, idx1] > 0:
                    lig_dist_mat[idx0, idx1] = 1
                else:
                    lig_dist_mat[idx0, idx1] = config_Floyd_INF # no edge
    return lig_dist_mat

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

def collate_mols(mol_dicts, moltree_vocab):
    data_batch = {}

    for key in ['atom_type', 'position', 'rec_mask', 'cannot_contact', 'new_atom_type', 'new_dist', 'new_angle', 'new_torsion', 'cannot_focus', 'can_cyclize']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)

    data_batch['bond_type'] = mol_dicts[0]['bond_type']
    idx_bias = mol_dicts[0]['atom_type'].shape[0]
    for mol_dict in mol_dicts[1:]:
        data_batch['bond_type'] = torch.cat((data_batch['bond_type'], torch.cat((mol_dict['bond_type'][:, :2] + idx_bias, mol_dict['bond_type'][:, 2:]), dim=1)), dim=0)
        idx_bias = idx_bias + mol_dict['atom_type'].shape[0]

    lig_bond_repeats = torch.tensor([int(mol_dicts[i]['can_cyclize'].shape[0]) for i in range(len(mol_dicts))])
    data_lig_bond_stepid = torch.repeat_interleave(torch.arange(len(mol_dicts)), repeats=lig_bond_repeats)
    data_batch['lig_bond_stepid'] = data_lig_bond_stepid

    num_steps_list = torch.tensor([0]+[len(mol_dicts[i]['new_atom_type']) for i in range(len(mol_dicts)-1)])
    batch_idx_offsets = torch.cumsum(num_steps_list, dim=0)
    repeats = torch.tensor([len(mol_dict['batch']) for mol_dict in mol_dicts])
    lig_repeats = torch.tensor([len(mol_dicts[i]['new_atom_type']) for i in range(len(mol_dicts))])
    data_lig_stepid = torch.repeat_interleave(torch.arange(len(mol_dicts)), repeats=lig_repeats)
    data_lig_moltree = np.array([mol_dict['lig_moltree_raw'] for mol_dict in mol_dicts], dtype=MolTree) 
    lig_moltree = tensorize_moltree(tree_batch=data_lig_moltree, vocab=moltree_vocab)
    data_batch['lig_stepid'] = data_lig_stepid      # [total_steps, ]
    data_batch['lig_moltree'] = lig_moltree   # [total_steps, ?]

    batch_idx_repeated_offsets = torch.repeat_interleave(batch_idx_offsets, repeats)
    batch_offseted = torch.cat([mol_dict['batch'] for mol_dict in mol_dicts], dim=0) + batch_idx_repeated_offsets
    data_batch['batch'] = batch_offseted

    num_atoms_list = torch.tensor([0]+[len(mol_dicts[i]['atom_type']) for i in range(len(mol_dicts)-1)])
    atom_idx_offsets = torch.cumsum(num_atoms_list, dim=0)
    for key in ['focus', 'c1_focus', 'c2_c1_focus', 'contact_y_or_n']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        atom_idx_repeated_offsets = torch.repeat_interleave(atom_idx_offsets, repeats)
        if key == 'contact_y_or_n':
            atom_offseted = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0) + atom_idx_repeated_offsets
        else:
            atom_offseted = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0) + atom_idx_repeated_offsets[:,None]
        data_batch[key] = atom_offseted

    return data_batch


class CrossDocked2020_SBDD(Dataset):
    def __init__(self, data_root='../data/crossdocked_pocket10', data_file='../data/crossdocked_pocket10/train_pocket10.types', \
        moltree_file='../data/pocket10_moltree.pkl', smi_file='../data/pocket10_smi.txt', vocab_file='../data/pocket10_vocab.txt', \
        atomic_num_to_type = atomic_num_to_type, atomic_element_to_type = atomic_element_to_type, binding_site_range=15.0):
        super().__init__()
        data_cols = [
            'low_rmsd',
            'true_aff',
            'xtal_rmsd',
            'rec_src',
            'lig_src',
            'vina_aff'
        ]
        self.data_lines = pd.read_csv(
            data_file, sep=' ', names=data_cols, index_col=False
        )
        self.data_root = data_root
        
        self.atomic_num_to_type = atomic_num_to_type
        self.atomic_element_to_type = atomic_element_to_type
        self.bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3}
        self.binding_site_range = binding_site_range
        self.pdb_parser = PDBParser()

        # tensor of pre-processed moltree objects
        with open(moltree_file, 'rb') as f:
            self.moltree_data = pkl.load(f)
        assert self.moltree_data.shape[0] == len(self.data_lines)

        self.moltree_vocab = [x.strip("\r\n ") for x in open(vocab_file)] 
        self.moltree_vocab = Vocab(self.moltree_vocab)

        

    def __len__(self):
        return len(self.data_lines)
    
    
    def read_rec_mol(self, mol_src):
        '''
        mol_src: the path of a .pdb file
        return: biopython <Structure>
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            structure = self.pdb_parser.get_structure('', os.path.join(self.data_root, mol_src))
        return structure
    
    
    def read_lig_mol(self, mol_src):
        '''
        mol_src: the path of a .sdf file
        return: rdkit.Chem.rdmolfiles.SDMolSupplier
        '''
        supp = Chem.SDMolSupplier()
        sdf_file = os.path.join(self.data_root, mol_src)
        supp.SetData(open(sdf_file).read(), removeHs=False, sanitize=False)
        return supp
       
    

    def get_rec_mol(self, mol_src):
        return self.read_rec_mol(mol_src)

    def get_lig_mol(self, mol_src):
        return self.read_lig_mol(mol_src)


    def __getitem__(self, index):
        '''
        Note that H atoms are not considered in both lig and rec.
        '''
        example = self.data_lines.iloc[index]
        rec_structure = self.get_rec_mol(example.rec_src)
        lig_supplier = self.get_lig_mol(example.lig_src.rsplit('.', 1)[0]) # Read .sdf file instead of .sdf.gz file; Why? (https://github.com/rdkit/rdkit/issues/1938)

        rec_atom_type = [self.atomic_element_to_type[atom.element] for atom in rec_structure.get_atoms() if atom.element!='H']

        try:
            assert len([atom.coord for atom in rec_structure.get_atoms() if atom.element!='H'])
        except:
            print(example.rec_src, example.lig_src)
            print('Void PDB file. We can ignore it!')
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index)
        rec_position = np.stack([atom.coord for atom in rec_structure.get_atoms() if atom.element!='H'], axis=0)
        rec_atom_type = torch.tensor(rec_atom_type) #[rec_n_atoms]
        rec_position = torch.tensor(rec_position)  #[rec_n_atoms, 3]
        
        del rec_structure

        lig_moltree_raw = self.moltree_data[index]

        lig_mol = Chem.rdmolops.RemoveAllHs(lig_supplier[0], sanitize=False)
        lig_n_atoms = lig_mol.GetNumAtoms()
        lig_pos = lig_supplier.GetItemText(0).split('\n')[4:4+lig_n_atoms]
        lig_position = np.array([[float(x) for x in line.split()[:3]] for line in lig_pos], dtype=np.float32)
        lig_atom_type = np.array([self.atomic_num_to_type[atom.GetAtomicNum()] for atom in lig_mol.GetAtoms()])
        lig_con_mat = np.zeros([lig_n_atoms, lig_n_atoms], dtype=int)
        for bond in lig_mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = self.bond_to_type[bond.GetBondType()]
            lig_con_mat[start, end] = bond_type
            lig_con_mat[end, start] = bond_type
        lig_atom_type = torch.tensor(lig_atom_type) #[lig_n_atoms]
        lig_position = torch.tensor(lig_position)  #[lig_n_atoms, 3]
        lig_atom_bond_valency = torch.tensor(np.sum(lig_con_mat, axis=1)) #[lig_n_atoms]
        del lig_supplier


        # Get binding site
        lig_center = torch.mean(lig_position, dim=0)
        rec_atom_dist_to_lig_center = torch.sqrt(torch.sum(torch.square(rec_position - lig_center), dim=-1))
        selected_mask = rec_atom_dist_to_lig_center <= self.binding_site_range
        try:
            assert torch.sum(selected_mask) >= 3 # Ensure that there are at least 3 selected atoms in rec
        except:
            print(example.rec_src, example.lig_src)
            print('One sample does not bind tightly. We can ignore it!')
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index)
        rec_atom_type = rec_atom_type[selected_mask]
        rec_position = rec_position[selected_mask]

        
        rec_n_atoms = len(rec_atom_type)


        lig_rec_squared_dist = torch.sum(torch.square(lig_position[:,None,:] - rec_position[None,:,:]), dim=-1) #[lig_n_atoms, rec_n_atoms]
        lig_internal_squared_dist = torch.sum(torch.square(lig_position[:,None,:] - lig_position[None,:,:]), dim=-1) #[lig_n_atoms, lig_n_atoms]


        # To find contact nodes and node in rec that are furthest from lig
        min_index = torch.argmin(lig_rec_squared_dist)
        lig_contact_id = min_index // rec_n_atoms
        rec_contact_id = min_index % rec_n_atoms
        rec_n_contact_id = torch.argmax(torch.sum(lig_rec_squared_dist, dim=0))

        # Start from the contact node in the lig
        perm = torch.arange(0, lig_n_atoms, dtype=int)
        perm[0] = lig_contact_id
        perm[lig_contact_id] = 0
        lig_atom_type, lig_position, lig_atom_bond_valency, lig_rec_squared_dist = lig_atom_type[perm], lig_position[perm], lig_atom_bond_valency[perm], lig_rec_squared_dist[perm]
        lig_con_mat, lig_internal_squared_dist = lig_con_mat[perm][:, perm], lig_internal_squared_dist[perm][:, perm]

        # prepare for Floyd
        lig_dist_mat = con2dist(lig_con_mat, lig_n_atoms)
        lig_con_mat = torch.tensor(lig_con_mat) #[lig_n_atoms, lig_n_atoms]

        # ring-first traversal
        start_atom = 0  # must start from the contact atom
        # start_atom = random.randint(0, lig_n_atoms - 1)
        standard_ring_first_order = IterFloyd(lig_n_atoms, lig_dist_mat, start_atom) #random.randint(0, lig_n_atoms - 1))
        # print(standard_ring_first_order)

        # focal list & neighbourhood priority
        focal_order = [-1]
        ring_first_order = [start_atom]
        ring_edges = []
        ring_edges_src, ring_edges_dst = [], []
        vis = [0 for _ in range(lig_n_atoms)]
        vis[start_atom] = 1
        remaining_edges = lig_con_mat.clone().detach()
        for idx in standard_ring_first_order:
            for new_idx in range(lig_n_atoms):
                if remaining_edges[idx][new_idx] != 0:
                    if vis[new_idx] == 1:
                        ring_edges.append((idx, new_idx))
                        ring_edges_src.append(idx)
                        ring_edges_dst.append(new_idx)
                    else:
                        vis[new_idx] = 1
                        focal_order.append(idx)
                        ring_first_order.append(new_idx)
                    remaining_edges[idx][new_idx] = 0
                    remaining_edges[new_idx][idx] = 0

        steps_can_cyclize = []
        for i in range(len(ring_first_order)):
            for j in range(i):
                steps_can_cyclize.append(int(lig_con_mat[ring_first_order[i], ring_first_order[j]]))
        steps_can_cyclize = torch.tensor(steps_can_cyclize, dtype=torch.long)
        focus_node_id = focal_order[1:]
        target_node_id = ring_first_order[1:]        

        # Decide the order among lig nodes
        nx_graph = nx.from_numpy_matrix(lig_internal_squared_dist.numpy())
        try:
            assert nx.is_connected(nx_graph) # New stuff, we assume the ligand to be a monolithic molecule
        except:
            print(example.rec_src, example.lig_src)
            print('The ligand is not a monolithic molecule. We can ignore it!')
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index)

        # no ring-first traversal
        # edges = list(tree.minimum_spanning_edges(nx_graph, algorithm='prim', data=False)) # return edges starts from the 0-th node (i.e., the contact node here) by default
        # focus_node_id, target_node_id = zip(*edges)        
        
        node_perm = torch.cat((torch.tensor([0]), torch.tensor(target_node_id)))
        lig_atom_type, lig_position, lig_atom_bond_valency, lig_rec_squared_dist = lig_atom_type[node_perm], lig_position[node_perm], lig_atom_bond_valency[node_perm], lig_rec_squared_dist[node_perm]
        lig_con_mat, lig_internal_squared_dist = lig_con_mat[node_perm][:, node_perm], lig_internal_squared_dist[node_perm][:, node_perm]



        # Prepare training data for sequential generation 
        focus_node_id = torch.tensor(focus_node_id)
        focus_ids = torch.nonzero(focus_node_id[:,None] == node_perm[None,:])[:,1] # focus_ids denotes the focus atom IDs that are indiced according to the order given by node_perm

        steps_cannot_focus = torch.empty([0,1], dtype=torch.float)
        idx_offsets = torch.cumsum(torch.arange(lig_n_atoms), dim=0) #[M]
        idx_offsets_brought_by_rec = rec_n_atoms*torch.arange(1, lig_n_atoms) #[M-1]




        for i in range(lig_n_atoms):
            if i==0:
                # In the 1st step, all we have is the rec. Note that contact classifier should be only applied for the 1st step in which we don't have any lig atoms
                steps_atom_type = rec_atom_type
                steps_rec_mask = torch.ones([rec_n_atoms], dtype=torch.bool)
                contact_y_or_n = torch.tensor([rec_contact_id, rec_n_contact_id], dtype=int) # The atom IDs of contact node and the node that are furthest from lig. 
                cannot_contact = torch.tensor([0,1], dtype=torch.float) # The groundtruth for contact_y_or_n
                steps_position = rec_position
                steps_batch = torch.tensor([i]).repeat(rec_n_atoms)
                steps_focus = torch.tensor([rec_contact_id], dtype=int)

                dist_to_focus = torch.sum(torch.square(rec_position[rec_contact_id] - rec_position), dim=-1)
                _, indices = torch.topk(dist_to_focus, 3, largest=False)
                one_step_c1, one_step_c2 = indices[1], indices[2]
                assert indices[0] == rec_contact_id
                steps_c1_focus = torch.tensor([one_step_c1, rec_contact_id], dtype=int).view(1,2)
                steps_c2_c1_focus = torch.tensor([one_step_c2, one_step_c1, rec_contact_id], dtype=int).view(1,3)

                focus_pos, new_pos = rec_position[rec_contact_id], lig_position[i]
                one_step_dis = torch.norm(new_pos - focus_pos)
                steps_dist = one_step_dis.view(1,1)

                c1_pos = rec_position[one_step_c1]
                a = ((c1_pos - focus_pos) * (new_pos - focus_pos)).sum(dim=-1)
                b = torch.cross(c1_pos - focus_pos, new_pos - focus_pos).norm(dim=-1)
                one_step_angle = torch.atan2(b,a)
                steps_angle = one_step_angle.view(1,1)

                c2_pos = rec_position[one_step_c2]
                plane1 = torch.cross(focus_pos - c1_pos, new_pos - c1_pos)
                plane2 = torch.cross(focus_pos - c1_pos, c2_pos - c1_pos)
                a = (plane1 * plane2).sum(dim=-1)
                b = (torch.cross(plane1, plane2) * (focus_pos - c1_pos)).sum(dim=-1) / torch.norm(focus_pos - c1_pos)
                one_step_torsion = torch.atan2(b, a)
                steps_torsion = one_step_torsion.view(1,1)


            else:
                one_step_atom_type = torch.cat((lig_atom_type[:i], rec_atom_type), dim=0)
                steps_atom_type = torch.cat((steps_atom_type, one_step_atom_type))
                one_step_rec_mask = torch.cat((torch.zeros([i], dtype=torch.bool), torch.ones([rec_n_atoms], dtype=torch.bool)), dim=0)
                steps_rec_mask = torch.cat((steps_rec_mask, one_step_rec_mask))
                one_step_position =  torch.cat((lig_position[:i], rec_position), dim=0)
                steps_position = torch.cat((steps_position, one_step_position))
                steps_batch = torch.cat((steps_batch, torch.tensor([i]).repeat(i + rec_n_atoms)))

                partial_lig_con_mat = lig_con_mat[:i, :i]
                bond_sum = partial_lig_con_mat.sum(dim=1, keepdim=True)
                steps_cannot_focus = torch.cat((steps_cannot_focus, (bond_sum == lig_atom_bond_valency[:i, None]).float()))

                focus_id = focus_ids[i-1]
                if i == 1: # c1, c2 must be in rec
                    dist_to_focus = lig_rec_squared_dist[focus_id]
                    _, indices = torch.topk(dist_to_focus, 2, largest=False)
                    one_step_c1, one_step_c2 = indices[0], indices[1]
                    one_step_c1_focus = torch.tensor([one_step_c1+idx_offsets[i]+idx_offsets_brought_by_rec[i-1], focus_id+idx_offsets_brought_by_rec[i-1]], dtype=int).view(1,2)
                    steps_c1_focus = torch.cat((steps_c1_focus, one_step_c1_focus), dim=0)
                    one_step_c2_c1_focus = torch.tensor([one_step_c2+idx_offsets[i]+idx_offsets_brought_by_rec[i-1],one_step_c1+idx_offsets[i]+idx_offsets_brought_by_rec[i-1], focus_id+idx_offsets_brought_by_rec[i-1]], dtype=int).view(1,3)
                    steps_c2_c1_focus = torch.cat((steps_c2_c1_focus, one_step_c2_c1_focus), dim=0)

                    focus_pos, new_pos = lig_position[focus_id], lig_position[i]
                    one_step_dis = torch.norm(new_pos - focus_pos).view(1,1)
                    steps_dist = torch.cat((steps_dist, one_step_dis), dim=0)

                    c1_pos = rec_position[one_step_c1]
                    a = ((c1_pos - focus_pos) * (new_pos - focus_pos)).sum(dim=-1)
                    b = torch.cross(c1_pos - focus_pos, new_pos - focus_pos).norm(dim=-1)
                    one_step_angle = torch.atan2(b,a).view(1,1)
                    steps_angle = torch.cat((steps_angle, one_step_angle), dim=0)

                    c2_pos = rec_position[one_step_c2]
                    plane1 = torch.cross(focus_pos - c1_pos, new_pos - c1_pos)
                    plane2 = torch.cross(focus_pos - c1_pos, c2_pos - c1_pos)
                    a = (plane1 * plane2).sum(dim=-1)
                    b = (torch.cross(plane1, plane2) * (focus_pos - c1_pos)).sum(dim=-1) / torch.norm(focus_pos - c1_pos)
                    one_step_torsion = torch.atan2(b, a).view(1,1)
                    steps_torsion = torch.cat((steps_torsion, one_step_torsion), dim=0)

                else: #c1, c2 could be in both lig and rec
                    dist_to_focus = torch.cat((lig_internal_squared_dist[focus_id, :i],lig_rec_squared_dist[focus_id]))
                    _, indices = torch.topk(dist_to_focus, 3, largest=False)
                    one_step_c1, one_step_c2 = indices[1], indices[2]

                    one_step_c1_focus = torch.tensor([one_step_c1+idx_offsets[i-1]+idx_offsets_brought_by_rec[i-1], focus_id+idx_offsets[i-1]+idx_offsets_brought_by_rec[i-1]], dtype=int).view(1,2)
                    one_step_c2_c1_focus = torch.tensor([one_step_c2+idx_offsets[i-1]+idx_offsets_brought_by_rec[i-1], one_step_c1+idx_offsets[i-1]+idx_offsets_brought_by_rec[i-1], focus_id+idx_offsets[i-1]+idx_offsets_brought_by_rec[i-1]], dtype=int).view(1,3)
                    if one_step_c1 < i: # c1 in lig
                        c1_pos = lig_position[one_step_c1]
                        if one_step_c2 < i: # c2 in lig
                            c2_pos = lig_position[one_step_c2]
                        else: 
                            c2_pos = rec_position[one_step_c2-i]
                    else: 
                        c1_pos = rec_position[one_step_c1-i]
                        if one_step_c2 < i: # c2 in lig
                            c2_pos = lig_position[one_step_c2]
                        else: # c2 in rec
                            c2_pos = rec_position[one_step_c2-i]
                    steps_c1_focus = torch.cat((steps_c1_focus, one_step_c1_focus), dim=0)
                    steps_c2_c1_focus = torch.cat((steps_c2_c1_focus, one_step_c2_c1_focus), dim=0)

                    focus_pos, new_pos = lig_position[focus_id], lig_position[i]
#                     if i==3 or i==4: # Use for debug. We have verified the id offset is correct.
#                         print(new_pos)
                    one_step_dis = torch.norm(new_pos - focus_pos).view(1,1)
                    steps_dist = torch.cat((steps_dist, one_step_dis), dim=0)

                    a = ((c1_pos - focus_pos) * (new_pos - focus_pos)).sum(dim=-1)
                    b = torch.cross(c1_pos - focus_pos, new_pos - focus_pos).norm(dim=-1)
                    one_step_angle = torch.atan2(b,a).view(1,1)
                    steps_angle = torch.cat((steps_angle, one_step_angle), dim=0)

                    plane1 = torch.cross(focus_pos - c1_pos, new_pos - c1_pos)
                    plane2 = torch.cross(focus_pos - c1_pos, c2_pos - c1_pos)
                    a = (plane1 * plane2).sum(dim=-1)
                    b = (torch.cross(plane1, plane2) * (focus_pos - c1_pos)).sum(dim=-1) / torch.norm(focus_pos - c1_pos)
                    one_step_torsion = torch.atan2(b, a).view(1,1)
                    steps_torsion = torch.cat((steps_torsion, one_step_torsion), dim=0)


        steps_focus = torch.cat((steps_focus, focus_ids+idx_offsets[:-1]+idx_offsets_brought_by_rec), dim=0)
        steps_new_atom_type = lig_atom_type

        # For example, for a rec-lig pair, rec has N atoms and lig has M atoms
        data_batch = {}
        data_batch['atom_type'] = steps_atom_type # [N+(1+N)+(2+N)+...+(M-1+N)], which correspond to M steps
        data_batch['position'] = steps_position # [N+(1+N)+(2+N)+...+(M-1+N), 3]
        data_batch['rec_mask'] = steps_rec_mask # [N+(1+N)+(2+N)+...+(M-1+N)]
        data_batch['batch'] = steps_batch # [N+(1+N)+(2+N)+...+(M-1+N)]
        data_batch['contact_y_or_n'] = contact_y_or_n # [2]
        data_batch['cannot_contact'] = cannot_contact # [2]
        data_batch['new_atom_type'] = steps_new_atom_type # [M]

        data_batch['focus'] = steps_focus[:,None] # [M, 1]
        data_batch['c1_focus'] = steps_c1_focus # [M, 2]
        data_batch['c2_c1_focus'] = steps_c2_c1_focus # [M, 3]

        data_batch['new_dist'] = steps_dist # [M, 1]
        data_batch['new_angle'] = steps_angle # [M, 1]
        data_batch['new_torsion'] = steps_torsion # [M, 1]

        data_batch['cannot_focus'] = steps_cannot_focus.view(-1) # [1+2+...+(M-1)]

        data_batch['bond_type'] = torch.Tensor(con2edge(lig_con_mat, rec_n_atoms, lig_n_atoms)).long() # [num_bonds in all batches, 3]

        assert steps_can_cyclize.shape == steps_cannot_focus.view(-1).shape

        data_batch['can_cyclize'] = steps_can_cyclize

        data_batch['lig_moltree_raw'] = lig_moltree_raw  # [1, ]


        return data_batch

        
       
