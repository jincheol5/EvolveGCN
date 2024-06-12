import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader


# load dataset
loader = WikiMathsDatasetLoader()
dataset = loader.get_dataset()

snapshot=dataset[0]


# edge_index로부터 NetworkX 그래프 생성
edge_index = snapshot.edge_index.numpy()
G = nx.Graph()
G.add_edges_from(edge_index.T)

# NetworkX 그래프에서 도달 가능성 행렬 계산
reachability_matrix = nx.floyd_warshall_numpy(G)
reachability_matrix[reachability_matrix != np.inf] = 1
reachability_matrix[reachability_matrix == np.inf] = 0

# 도달 가능성 행렬을 COO 형식의 희소 행렬로 변환
coo_reachability_matrix = coo_matrix(reachability_matrix)

# 결과 출력
print("COO 형식의 도달 가능성 행렬:")
print("data:", coo_reachability_matrix.data)
print("row indices:", coo_reachability_matrix.row)
print("col indices:", coo_reachability_matrix.col)