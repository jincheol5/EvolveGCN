import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from torch_geometric.nn import GCNConv
from compute_tR import get_tR_matrix
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = WikiMathsDatasetLoader()
dataset = loader.get_dataset()
snapshot=dataset[0]
tR_indices=get_tR_matrix(snapshot).to(device)

class EvolveGCN_H_Encoder(torch.nn.Module):
    def __init__(self, num_of_nodes,in_channels):
        super().__init__()
        self.recurruent = EvolveGCNH(num_of_nodes,in_channels)
        self.conv1= GCNConv(in_channels,16)
        self.conv2= GCNConv(16,in_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.recurruent(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = self.conv2(x,tR_indices)
        return x