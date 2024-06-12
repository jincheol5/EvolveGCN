import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from torch_geometric.nn import GCNConv
from compute_tR import get_tR_matrix
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader

loader = WikiMathsDatasetLoader()
dataset = loader.get_dataset()
snapshot=dataset[0]


class EvolveGCN_H_Encoder(torch.nn.Module):
    def __init__(self, num_of_nodes,in_channels):
        super().__init__()
        self.recurruent = EvolveGCNH(num_of_nodes,in_channels)
        self.conv1= GCNConv(in_channels,in_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.recurruent(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv1(x,get_tR_matrix(snapshot))
        return x