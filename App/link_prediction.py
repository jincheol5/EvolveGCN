import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges

from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

from evolve_gcn import EvolveGCN_H

# GPU 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')




### data load
loader = WikiMathsDatasetLoader()
dataset = loader.get_dataset()


# train, test 
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

def preprocess_snapshot(snapshot):
    data = Data(x=snapshot.x, edge_index=snapshot.edge_index)
    data = train_test_split_edges(data)
    return data

train_data_list = [preprocess_snapshot(snapshot) for snapshot in train_dataset]
test_data_list = [preprocess_snapshot(snapshot) for snapshot in test_dataset]

model = EvolveGCN_H(node_features=dataset.num_features, out_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    cost = 0
    for snapshot in train_data_list:
        edge_index = snapshot.train_pos_edge_index
        neg_edge_index = snapshot.train_neg_edge_index

        y = torch.cat([torch.ones(edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(device)
        edge_index_combined = torch.cat([edge_index, neg_edge_index], dim=1).to(device)

        y_hat = model(snapshot.x, edge_index_combined)

        loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(), y)
        cost += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return cost / len(train_data_list)

def test():
    model.eval()
    cost = 0
    for snapshot in test_data_list:
        edge_index = snapshot.test_pos_edge_index
        neg_edge_index = snapshot.test_neg_edge_index

        y = torch.cat([torch.ones(edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(device)
        edge_index_combined = torch.cat([edge_index, neg_edge_index], dim=1).to(device)

        y_hat = model(snapshot.x, edge_index_combined)
        
        loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(), y)
        cost += loss.item()
    return cost / len(test_data_list)


for epoch in range(1, 51):
    train_cost = train()
    test_cost = test()
    print(f"Epoch: {epoch}, Train Cost: {train_cost:.4f}, Test Cost: {test_cost:.4f}")