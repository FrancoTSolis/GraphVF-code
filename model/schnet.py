import torch
from torch.nn import Embedding, ModuleList, Sequential, Linear
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, MessagePassing
from math import pi as PI


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
    
    
class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W
    
    
class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians + hidden_channels, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x
    

class SchNet(torch.nn.Module):
    def __init__(self, num_node_types, num_bond_types, hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0):
        super(SchNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        
        self.embedding = Embedding(num_node_types, hidden_channels)
        self.edge_embedding = Embedding(num_bond_types, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)
        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()

    
    def forward(self, z, pos, batch, bond):
        h = self.embedding(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        global first_edge_index
        first_edge_index = edge_index.clone().detach()
        row, col = edge_index.clone().detach()
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        CONFIG_MAX_N_LIGAND = 100 # We assume the number of atoms in a ligand is smaller than this bias. We use this to accelerate the retrieval of bond types for edges in the radius graph.
        # bond size=[num_edges, 3] format=[(start_1, end_1, type_1) ... (start_n, end_n, type_n)]

        edge_type_con = torch.zeros(z.shape[0], 2 * CONFIG_MAX_N_LIGAND + 1).long().cuda()
        edge_type_con[bond[:, 0], bond[:, 0] - bond[:, 1] + CONFIG_MAX_N_LIGAND] = bond[:, 2]
        edge_type_con[bond[:, 0], bond[:, 0] - bond[:, 1] + CONFIG_MAX_N_LIGAND] = bond[:, 2]
        zero_tensor = torch.zeros(row.shape[0]).long().cuda()
        condition_tensor = (row - col < -CONFIG_MAX_N_LIGAND) | (row - col > CONFIG_MAX_N_LIGAND) # in order to save memory
        row[torch.arange(zero_tensor.shape[0])[condition_tensor]] = zero_tensor[torch.arange(zero_tensor.shape[0])[condition_tensor]]
        col[torch.arange(zero_tensor.shape[0])[condition_tensor]] = zero_tensor[torch.arange(zero_tensor.shape[0])[condition_tensor]]
        edge_type = edge_type_con[row, row - col + CONFIG_MAX_N_LIGAND]
        edge_attr = torch.cat((edge_attr, self.edge_embedding(edge_type)), dim=-1) # cat embeddings to gaussians

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
            h = F.normalize(h, dim=-1)

        return h