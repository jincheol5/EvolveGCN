import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 그래프 생성
G = nx.DiGraph()
edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
G.add_edges_from(edges)

# 인접 행렬 생성
adj_matrix = nx.adjacency_matrix(G).todense()

# Reachability 행렬 계산 (플로이드-워셜 알고리즘 사용)
reach_matrix = nx.floyd_warshall_numpy(G)
reach_matrix = np.where(reach_matrix != np.inf, 1, 0)

# 그래프 그리기
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=15, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{u}->{v}' for u, v in G.edges()}, font_color='red')
plt.title("Graph")
plt.show()

# 인접 행렬 출력
print("Adjacency Matrix:")
print(adj_matrix)

# Reachability 행렬 출력
print("Reachability Matrix:")
print(reach_matrix)