import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

# Distâncias entre os itens x1 a x6
distances = [
    [0, 0.24, 0.22, 0.37, 0.34, 0.23],
    [0.24, 0, 0.15, 0.20, 0.14, 0.25],
    [0.22, 0.15, 0, 0.15, 0.28, 0.11],
    [0.37, 0.20, 0.15, 0, 0.29, 0.22],
    [0.34, 0.14, 0.28, 0.29, 0, 0.39],
]

# Converter lista de listas em uma matriz simétrica
dist_matrix = np.array(distances)

# Converter a matriz simétrica em uma forma condensada
condensed_dist_matrix = squareform(dist_matrix)

# Realizando os agrupamentos para cada tipo de linkage
single_linkage = linkage(condensed_dist_matrix, method='single')
complete_linkage = linkage(condensed_dist_matrix, method='complete')
average_linkage = linkage(condensed_dist_matrix, method='average')

# Plotando os dendrogramas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
dendrogram(single_linkage, ax=axes[0], labels=['x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
axes[0].set_title('Single Linkage')

dendrogram(complete_linkage, ax=axes[1], labels=['x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
axes[1].set_title('Complete Linkage')

dendrogram(average_linkage, ax=axes[2], labels=['x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
axes[2].set_title('Average Linkage')

plt.tight_layout()
plt.show()
