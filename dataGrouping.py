from sklearn.cluster import KMeans
import numpy as np

# Gerando dados aleatórios
np.random.seed(0) # Descomente para resultados reprodutíveis
X = np.random.rand(100, 2)  # 100 pontos em 2 dimensões

# Definindo o modelo K-means
kmeans = KMeans(n_clusters=5)  # Definindo o número de clusters

# Fitando o modelo
kmeans.fit(X)

# Predições de clusters
y_kmeans = kmeans.predict(X)

# Centros dos clusters
centroids = kmeans.cluster_centers_

print("Centros dos clusters:\n", centroids)
print("Atribuições de cluster para cada ponto:\n", y_kmeans)

# Visualização dos resultados (opcional)
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5)
plt.title("Visualização do Agrupamento K-means")
plt.show()
