import torch
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from tqdm import tqdm

class EvolveGCN(torch.nn.Module):
    def __init__(self, input_dim, node_count):
        super().__init__()