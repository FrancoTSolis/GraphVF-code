import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .net_utils import *
from .geometric_computing import *
from .features import dist_emb, angle_emb, torsion_emb
from .schnet import SchNet
from fast_jtnn import JTNNEncoder, MLPwoLastAct


class GraphVF(nn.Module):
    def __init__(self, cutoff, num_node_types, num_lig_node_types, num_interactions, num_filters, num_gaussians,
        hidden_channels, basis_emb_size, num_spherical, num_radial, num_flow_layers, deq_coeff=0.9, use_gpu=True, 
        num_bond_types=4, num_moltree_vocab=-1):

        super(GraphVF, self).__init__()
        self.use_gpu = use_gpu
        self.num_node_types = num_node_types
        self.num_lig_node_types = num_lig_node_types
        self.num_bond_types = num_bond_types

        # num_bond_types=4: New stuff, None/1/2/3 (Aroma Bond have been kekulized)
        # Only 4 types of bond: easy to learn their embeddings
        self.feat_net = SchNet(num_node_types, num_bond_types, hidden_channels, num_filters, num_interactions, num_gaussians, cutoff)

        node_feat_dim, dist_feat_dim, angle_feat_dim, torsion_feat_dim = hidden_channels, hidden_channels, hidden_channels * 2, hidden_channels * 3
        bond_feat_dim = hidden_channels

        self.node_flow_layers = nn.ModuleList([ST_Net_Exp(node_feat_dim, num_lig_node_types, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.cyclize_bond_flow_layers = nn.ModuleList([ST_Net_Exp(bond_feat_dim, num_bond_types, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])        
        self.dist_flow_layers = nn.ModuleList([ST_Net_Exp(dist_feat_dim, 1, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.angle_flow_layers = nn.ModuleList([ST_Net_Exp(angle_feat_dim, 1, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.torsion_flow_layers = nn.ModuleList([ST_Net_Exp(torsion_feat_dim, 1, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.focus_mlp = MLP(hidden_channels)
        self.contact_mlp = MLP(hidden_channels)
        self.deq_coeff = deq_coeff

        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent=5)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent=5)
        
        self.dist_lb2 = LB2(num_radial, basis_emb_size, hidden_channels)
        self.angle_lb2 = LB2(num_spherical * num_radial, basis_emb_size, hidden_channels)

        self.cyclize_bond_lb2 = LB2(bond_feat_dim, basis_emb_size, bond_feat_dim)

        # CPU caveat
        self.jtnn = JTNNEncoder(hidden_size=450, depth=20, embedding=nn.Embedding(num_embeddings=num_moltree_vocab, embedding_dim=450))     # hyper-params follow jtnn
        self.jtnn_latent_size = num_lig_node_types + num_bond_types + 1 + 1 + 1   # corresponding to: ligand atom type[27], bond type[4], dist[1], angle[1], torsion[1]
        self.encoder_head = MLPwoLastAct(input_size=450, output_sizes=[512, 2 * self.jtnn_latent_size], use_bn=True)

        if use_gpu:
            self.feat_net = self.feat_net.to('cuda')
            self.node_flow_layers = self.node_flow_layers.to('cuda')
            self.dist_flow_layers = self.dist_flow_layers.to('cuda')
            self.angle_flow_layers = self.angle_flow_layers.to('cuda')
            self.torsion_flow_layers = self.torsion_flow_layers.to('cuda')
            self.focus_mlp = self.focus_mlp.to('cuda')
            self.contact_mlp = self.contact_mlp.to('cuda')
            self.dist_lb2 = self.dist_lb2.to('cuda')
            self.angle_lb2 = self.angle_lb2.to('cuda')
            self.cyclize_bond_lb2 = self.cyclize_bond_lb2.to('cuda')
            self.dist_emb = self.dist_emb.to('cuda')
            self.angle_emb = self.angle_emb.to('cuda')
            self.cyclize_bond_flow_layers = self.cyclize_bond_flow_layers.to('cuda')
            self.jtnn = self.jtnn.to('cuda')
            self.encoder_head = self.encoder_head.to('cuda')


    def forward(self, data_batch):
        # New stuff, generate moltree embedding for each molecule
        lig_moltree = data_batch['lig_moltree']
        tree_vecs, _ = self.jtnn(*lig_moltree)
        jtnn_latent = self.encoder_head(tree_vecs)
        latent_mean, latent_logstd = torch.chunk(jtnn_latent, chunks=2, dim=-1)    # [jtnn_latent_size, ]
        # do not consider covariate matrix at this time, i.e. gaussians independent along different dimensions

        z, pos, batch, bond = data_batch['atom_type'], data_batch['position'], data_batch['batch'], data_batch['bond_type']
        node_feat = self.feat_net(z, pos, batch, bond)
        focus_score = self.focus_mlp(node_feat[~data_batch['rec_mask']])
        contact_score = self.contact_mlp(node_feat[data_batch['contact_y_or_n']])

        # generate atom types
        new_atom_type, focus = data_batch['new_atom_type'], data_batch['focus']
        x_z = F.one_hot(new_atom_type, num_classes=self.num_lig_node_types).float()
        x_z += self.deq_coeff * torch.rand(x_z.size(), device=x_z.device)
        local_node_type_feat = node_feat[focus[:,0]]
        node_latent, node_log_jacob = flow_forward(self.node_flow_layers, x_z, local_node_type_feat)
        node_type_emb_block = self.feat_net.embedding
        node_type_emb = node_type_emb_block(new_atom_type)
        node_emb = node_feat * node_type_emb[batch]

        # generate bond types
        cyclize_bond_type = data_batch['can_cyclize']   # Ground Truth for the new bond type
        cyclize_bond = F.one_hot(cyclize_bond_type, num_classes=self.num_bond_types).float()
        cyclize_bond += self.deq_coeff * torch.rand(cyclize_bond.size(), device=cyclize_bond.device)
        local_cyclize_bond_feat = node_emb[~data_batch['rec_mask']]
        cyclize_bond_latent, cyclize_bond_log_jacob = flow_forward(self.cyclize_bond_flow_layers, cyclize_bond, local_cyclize_bond_feat)
        cyclize_bond_emb = self.cyclize_bond_lb2(self.feat_net.edge_embedding(cyclize_bond_type))
        node_emb[~data_batch['rec_mask']] = node_emb[~data_batch['rec_mask']] * cyclize_bond_emb


        c1_focus, c2_c1_focus = data_batch['c1_focus'], data_batch['c2_c1_focus']
        dist, angle, torsion = data_batch['new_dist'], data_batch['new_angle'], data_batch['new_torsion']

        local_dist_feat = node_emb[focus[:,0]]
        dist_latent, dist_log_jacob = flow_forward(self.dist_flow_layers, dist, local_dist_feat)

        # d --> theta

        dist_emb = self.dist_lb2(self.dist_emb(dist.squeeze()[batch].to(torch.float)))
        node_emb = node_emb * dist_emb # [N, hidden] * [N, hidden]. N is the total number of steps for all molecules in the batch

        node_emb_clone = node_emb.clone() # Avoid changing node_emb in-place --> cannot comput gradient otherwise
        local_angle_feat = torch.cat((node_emb_clone[c1_focus[:,1]], node_emb_clone[c1_focus[:,0]]), dim=1)
        angle_latent, angle_log_jacob = flow_forward(self.angle_flow_layers, angle, local_angle_feat)


        #  d, theta --> phi
        dist_angle_emd = self.angle_lb2(self.angle_emb(dist.squeeze()[batch].to(torch.float), angle.squeeze()[batch].to(torch.float)))

        node_emb = node_emb * dist_angle_emd

        local_torsion_feat = torch.cat((node_emb[c2_c1_focus[:,2]], node_emb[c2_c1_focus[:,1]], node_emb[c2_c1_focus[:,0]]), dim=1)
        torsion_latent, torsion_log_jacob = flow_forward(self.torsion_flow_layers, torsion, local_torsion_feat)


        return (node_latent, node_log_jacob), (cyclize_bond_latent, cyclize_bond_log_jacob), focus_score, contact_score, (dist_latent, dist_log_jacob), (angle_latent, angle_log_jacob), (torsion_latent, torsion_log_jacob), (latent_mean, latent_logstd)


    def generate(self, type_to_atomic_number, rec_atom_type, rec_position, num_gen=100, temperature=[1.0, 1.0, 1.0, 1.0], min_atoms=2, max_atoms=35, focus_th=0.5, contact_th=0.5, add_final=False, contact_prob=False):
        with torch.no_grad(): # For unconditional ligand molecule generation, we set the priors to be N(0,1)
            if self.use_gpu:
                prior_node = torch.distributions.normal.Normal(torch.zeros([self.num_lig_node_types]).cuda(), temperature[0] * torch.ones([self.num_lig_node_types]).cuda())
                prior_dist = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[1] * torch.ones([1]).cuda())
                prior_angle = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[2] * torch.ones([1]).cuda())
                prior_torsion = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[3] * torch.ones([1]).cuda())
                prior_cyclize_bond = torch.distributions.normal.Normal(torch.zeros([self.num_bond_types]).cuda(), temperature[4] * torch.ones([self.num_bond_types]).cuda())
            else:
                prior_node = torch.distributions.normal.Normal(torch.zeros([self.num_lig_node_types]), temperature[0] * torch.ones([self.num_lig_node_types]))
                prior_dist = torch.distributions.normal.Normal(torch.zeros([1]), temperature[1] * torch.ones([1]))
                prior_angle = torch.distributions.normal.Normal(torch.zeros([1]), temperature[2] * torch.ones([1]))
                prior_torsion = torch.distributions.normal.Normal(torch.zeros([1]), temperature[3] * torch.ones([1]))
                prior_cyclize_bond = torch.distributions.normal.Normal(torch.zeros([self.num_bond_types]), temperature[4] * torch.ones([self.num_bond_types]))

            rec_n_atoms = len(rec_atom_type)
            node_type_emb_block = self.feat_net.embedding
            z_lig = torch.empty([num_gen, 0], dtype=int)
            pos_lig = torch.empty([num_gen, 0, 3], dtype=torch.float32)
            focuses = torch.empty([num_gen, 0], dtype=int) # Note that the 1st focus ID is the contact ID in rec
            bond = torch.empty([0, 3], dtype=int)
            if self.use_gpu:
                z_lig, pos_lig, bond, focuses = z_lig.cuda(), pos_lig.cuda(), bond.cuda(), focuses.cuda()
                rec_atom_type, rec_position = rec_atom_type.cuda(), rec_position.cuda()
            out_dict = {}

            feat_index = lambda node_id, f: f[torch.arange(num_gen), node_id]
            pos_index = lambda node_id, p: p[torch.arange(num_gen), node_id].view(num_gen,1,3)
            cyclize_edge = [[] for _ in range(num_gen)]

            for i in range(max_atoms):
                batch = torch.arange(num_gen, device=z_lig.device).view(num_gen, 1).repeat(1, i+rec_n_atoms)
                z = torch.cat((z_lig, rec_atom_type.repeat(num_gen, 1)), dim=1)
                pos = torch.cat((pos_lig, rec_position.repeat(num_gen, 1, 1)), dim=1)

                if i > 0:
                    bond_c = []
                    for idx_j, j in enumerate(cyclize_edge):
                        for k in j:
                            bond_c.append([k[0] + idx_j * (i+rec_n_atoms), k[1] + idx_j * (i+rec_n_atoms), k[2]])
                    bond = torch.tensor(bond_c, dtype=int)
                    if self.use_gpu:
                        bond = bond.cuda()

                node_feat = self.feat_net(z.view(-1), pos.view(-1,3), batch.view(-1), bond.view(-1, 3))

                if i == 0:
                    contact_score = self.contact_mlp(node_feat).view(num_gen, rec_n_atoms)
                    if contact_prob: # The prob of selecting a atom is propotional to the predicted prob
                        contact_mask = contact_score > contact_th
                        can_contact = contact_score
                        can_contact[contact_mask] = 0
                    else: # Contact atom is selected randomly from nodes with predicted score < contact_th
                        can_contact = contact_score < contact_th
                    focus_node_id = torch.multinomial(can_contact.float(), 1).view(num_gen)

                    node_feat = node_feat.view(num_gen, rec_n_atoms, -1)

                else:
                    rec_mask = torch.cat((torch.zeros([i], dtype=torch.bool), torch.ones([rec_n_atoms], dtype=torch.bool))).repeat(num_gen)
                    focus_score = self.focus_mlp(node_feat[~rec_mask]).view(num_gen, i)
                    # print(focus_score[0])
                    can_focus = (focus_score < focus_th)
                    complete_mask = (can_focus.sum(dim=-1) == 0)
                    if i > max(0, min_atoms-1) and torch.sum(complete_mask) > 0:
                        out_dict[i] = {}
                        out_node_types = z_lig[complete_mask].view(-1, i).cpu().numpy()
                        out_dict[i]['_atomic_numbers'] = type_to_atomic_number[out_node_types]
                        out_dict[i]['_positions'] = pos_lig[complete_mask].view(-1, i, 3).cpu().numpy()
                        out_dict[i]['_focus'] = focuses[complete_mask].view(-1, i).cpu().numpy()
                        
                    continue_mask = torch.logical_not(complete_mask)
                    # print(continue_mask)
                    dirty_mask = torch.nonzero(torch.isnan(focus_score).sum(dim=-1))[:,0]
                    if len(dirty_mask) > 0:
                        continue_mask[dirty_mask] = False
                    dirty_mask = torch.nonzero(torch.isinf(focus_score).sum(dim=-1))[:,0]
                    if len(dirty_mask) > 0:
                        continue_mask[dirty_mask] = False

                    if torch.sum(continue_mask) == 0:
                        break
                
                    node_feat = node_feat.view(num_gen, i+rec_n_atoms, -1)
                    num_gen = torch.sum(continue_mask).cpu().item()
                    z, pos, can_focus, focuses = z[continue_mask], pos[continue_mask], can_focus[continue_mask], focuses[continue_mask]
                    z_lig, pos_lig = z_lig[continue_mask], pos_lig[continue_mask]
                    focus_node_id = torch.multinomial(can_focus.float(), 1).view(num_gen)
                    node_feat = node_feat[continue_mask]
                    cyclize_edge = [_ for _idx, _ in enumerate(cyclize_edge) if continue_mask[_idx] == True]

                latent_node = prior_node.sample([num_gen])
                
                local_node_type_feat = feat_index(focus_node_id, node_feat)
                
                latent_node = flow_reverse(self.node_flow_layers, latent_node, local_node_type_feat)
                node_type_id = torch.argmax(latent_node, dim=1)
                node_type_emb = node_type_emb_block(node_type_id)
                node_emb = node_feat * node_type_emb.view(num_gen, 1, -1)

                
                # Simple valency checks are applied during generation to reduce the chances to break the valency constraints. 
                if i > 0:
                    rec_mask = torch.cat((torch.zeros([i], dtype=torch.bool), torch.ones([rec_n_atoms], dtype=torch.bool))).view(1, -1).repeat(num_gen, 1)
                    latent_cyclize_bond = prior_cyclize_bond.sample([num_gen]).repeat(1, i).view(num_gen * i, -1)
                    local_cyclize_bond_type_feat = node_emb[~rec_mask].view(num_gen * i, -1)
                    latent_cyclize_bond = flow_reverse(self.cyclize_bond_flow_layers, latent_cyclize_bond, local_cyclize_bond_type_feat)
                    cyclize_bond_type_id = torch.argmax(latent_cyclize_bond, dim=1)
                    # We reject bonds between current generated atom and non-focal atoms
                    cyclize_bond_type_id = torch.where(can_focus.view(-1), cyclize_bond_type_id, torch.zeros(cyclize_bond_type_id.shape).long().cuda())

                    # We keep only one edge between current generated atom and non-focus atoms
                    cyclize_focus_bond_type_id = torch.argmax(latent_cyclize_bond.view(num_gen, i, -1)[torch.arange(num_gen).long().cuda(), focus_node_id, :], dim=-1)
                    pre_cyclize_bond_type_id = cyclize_bond_type_id.clone()
                    cyclize_bond_type_id.view(num_gen, i)[torch.arange(num_gen, dtype=int, device=cyclize_bond_type_id.device), focus_node_id] = torch.zeros(num_gen).long().cuda()
                    new_cyclize_bond_type = []
                    for j in range(num_gen):
                        if torch.count_nonzero(cyclize_bond_type_id.view(num_gen, i)[j]) > 1:
                            candidate_bond_id = torch.arange(i, dtype=int, device=cyclize_bond_type_id.device)[cyclize_bond_type_id.view(num_gen, i)[j] > 0]
                            candidate_bond_type = cyclize_bond_type_id.view(num_gen, i)[j][candidate_bond_id]
                            sample_bond_id = random.randint(0, candidate_bond_id.shape[0] - 1)
                            modified_cyclize_bond_type = torch.zeros(i, dtype=int, device=cyclize_bond_type_id.device)
                            modified_cyclize_bond_type[candidate_bond_id[sample_bond_id]] = candidate_bond_type[sample_bond_id]
                            new_cyclize_bond_type.append(modified_cyclize_bond_type.unsqueeze(0))
                        else:
                            new_cyclize_bond_type.append(cyclize_bond_type_id.view(num_gen, i)[j:j+1])
                    cyclize_bond_type_id = torch.cat(new_cyclize_bond_type, dim=0).view(-1)
                    cyclize_bond_type_id.view(num_gen, i)[torch.arange(num_gen, dtype=int, device=cyclize_bond_type_id.device), focus_node_id] = cyclize_focus_bond_type_id

                    cyclize_bond_type_emb = self.cyclize_bond_lb2(self.feat_net.edge_embedding(cyclize_bond_type_id))
                    node_emb[~rec_mask] = node_emb[~rec_mask] * cyclize_bond_type_emb
                    for j in range(num_gen):
                        for k in range(i):
                            if int(cyclize_bond_type_id[j * i + k]) > 0:
                                cyclize_edge[j].append((k, i, int(cyclize_bond_type_id[j * i + k])))
                                cyclize_edge[j].append((i, k, int(cyclize_bond_type_id[j * i + k])))


                latent_dist = prior_dist.sample([num_gen])
                
                local_dist_feat = feat_index(focus_node_id, node_emb)
                
                dist = flow_reverse(self.dist_flow_layers, latent_dist, local_dist_feat)
                
                dist_emb = self.dist_lb2(self.dist_emb(dist.to(torch.float)))
                node_emb = node_emb * dist_emb.view(num_gen, 1, -1)
                
                dist_to_focus = torch.sum(torch.square(pos - pos_index(focus_node_id, pos)), dim=-1)
                _, indices = torch.topk(dist_to_focus, 3, largest=False)
                c1_node_id, c2_node_id = indices[:,1], indices[:,2]


                latent_angle = prior_angle.sample([num_gen])
                local_angle_feat = torch.cat((feat_index(focus_node_id, node_emb), feat_index(c1_node_id, node_emb)), dim=1)

                angle = flow_reverse(self.angle_flow_layers, latent_angle, local_angle_feat)


                dist_angle_emd = self.angle_lb2(self.angle_emb(dist.to(torch.float), angle.to(torch.float)))
                node_emb = node_emb * dist_angle_emd.view(num_gen, 1, -1)


                latent_torsion = prior_torsion.sample([num_gen])

                local_torsion_feat = torch.cat((feat_index(focus_node_id, node_emb), feat_index(c1_node_id, node_emb), feat_index(c2_node_id, node_emb)), dim=1)

                torsion = flow_reverse(self.torsion_flow_layers, latent_torsion, local_torsion_feat)
                new_pos = dattoxyz(pos_index(focus_node_id, pos), pos_index(c1_node_id, pos), pos_index(c2_node_id, pos), dist, angle, torsion)


                z_lig = torch.cat((z_lig, node_type_id[:, None]), dim=1)
                pos_lig = torch.cat((pos_lig, new_pos.view(num_gen, 1, 3)), dim=1)
                focuses = torch.cat((focuses, focus_node_id[:,None]), dim=1)


            if add_final and torch.sum(continue_mask) > 0:
                out_dict[i+1] = {}
                out_node_types = z_lig.view(-1,i+1).cpu().numpy()
                out_dict[i+1]['_atomic_numbers'] = type_to_atomic_number[out_node_types]
                out_dict[i+1]['_positions'] = pos_lig.view(-1, i+1, 3).cpu().numpy()
                out_dict[i+1]['_focus'] = focuses.view(-1, i+1).cpu().numpy()

            return out_dict