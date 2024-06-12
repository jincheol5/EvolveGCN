import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
import torch




def get_tR_matrix(snapshot):

    # edge_index로부터 NetworkX 방향 그래프 생성
    edge_index = snapshot.edge_index.numpy()
    G = nx.DiGraph()
    G.add_edges_from(edge_index.T)

    # # 강하게 연결된 그래프인지 확인
    # is_strongly_connected = nx.is_strongly_connected(G)
    # print("is connected?", is_strongly_connected)

    # NetworkX 그래프에서 도달 가능성 행렬 계산
    reachability_matrix = nx.floyd_warshall_numpy(G)
    reachability_matrix[reachability_matrix != np.inf] = 1
    reachability_matrix[reachability_matrix == np.inf] = 0

    # 도달 가능성 행렬을 COO 형식의 희소 행렬로 변환
    coo_reachability_matrix = coo_matrix(reachability_matrix)

    # 인덱스들을 LongTensor로 변환
    indices = np.vstack((coo_reachability_matrix.row, coo_reachability_matrix.col))
    coo_tensor = torch.LongTensor(indices)

    return coo_tensor