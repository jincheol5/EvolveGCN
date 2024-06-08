import torch
import torch.nn.functional as F
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import EvolveGCNH


class EvolveGCN_H(torch.nn.Module):
    def __init__(self, node_features, out_channels):
        super().__init__()
        self.recurrent = EvolveGCNH(node_features, out_channels)
        self.linear = torch.nn.Linear(out_channels, 1)
    
    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h