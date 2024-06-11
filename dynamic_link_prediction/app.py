import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE
from torch_geometric.utils import train_test_split_edges
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader
from model import EvolveGCN_H_Encoder
from tqdm import tqdm

# 데이터셋 로드 및 전처리
# TwitterTennis dataset은 dynamic graph temporal signal
# node수는 고정 
loader = TwitterTennisDatasetLoader()
dataset = loader.get_dataset()

num_node=dataset[0].x.shape[0]
dim_node_feature=dataset[0].x.shape[1]


# Split the dataset
def temporal_signal_split_for_link_prediction(data_iterator):
    # 마지막 snapshot만 test로 두고 나머지 train으로 분류

    train_iterator = data_iterator[0:-1]
    test_iterator = data_iterator[-1:]

    return train_iterator, test_iterator

train_dataset, test_dataset = temporal_signal_split_for_link_prediction(dataset)

# model
model = GAE(EvolveGCN_H_Encoder(num_node, dim_node_feature))

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train and test function
def train(snapshot):
    model.train()

    snapshot= train_test_split_edges(snapshot)
    x = snapshot.x.to(device)
    train_pos_edge_index = snapshot.train_pos_edge_index.to(device)

    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()

    return float(loss)


def test(x,train_pos_edge_index,pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

# train
epochs = 100
for epoch in tqdm(range(1, epochs + 1)):

    for i, snapshot in enumerate(train_dataset):
        train(snapshot)