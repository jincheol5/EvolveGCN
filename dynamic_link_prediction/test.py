import torch
from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader

# 데이터셋 로드
loader = TwitterTennisDatasetLoader()
dataset = loader.get_dataset()

# 첫 번째 스냅샷 확인
snapshot = dataset[0]

# 스냅샷의 속성 확인
print("Snapshot attributes:")
print(snapshot)

# train_pos_edge_index와 test_pos_edge_index 확인
if hasattr(snapshot, 'train_pos_edge_index'):
    print("Train Positive Edge Index:")
    print(snapshot.train_pos_edge_index)
else:
    print("No train_pos_edge_index in snapshot")

if hasattr(snapshot, 'test_pos_edge_index'):
    print("Test Positive Edge Index:")
    print(snapshot.test_pos_edge_index)
else:
    print("No test_pos_edge_index in snapshot")

# 필요한 경우 train_neg_edge_index 및 test_neg_edge_index도 확인
if hasattr(snapshot, 'train_neg_edge_index'):
    print("Train Negative Edge Index:")
    print(snapshot.train_neg_edge_index)
else:
    print("No train_neg_edge_index in snapshot")

if hasattr(snapshot, 'test_neg_edge_index'):
    print("Test Negative Edge Index:")
    print(snapshot.test_neg_edge_index)
else:
    print("No test_neg_edge_index in snapshot")