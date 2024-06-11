import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNH


class EvolveGCN_H_Encoder(torch.nn.Module):
    def __init__(self, num_of_nodes,in_channels):
        super().__init__()
        self.recurruent = EvolveGCNH(num_of_nodes,in_channels) 

    def forward(self, x, edge_index, edge_weight=None):
        x = self.recurruent(x, edge_index, edge_weight)
        return F.relu(x)