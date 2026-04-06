import numpy as np
from matplotlib import pyplot as plt

# don't change the seed
RNG = np.random.default_rng(3)
normal = RNG.multivariate_normal

t, n = 3, 2500
spread = np.identity(2)
X = normal([-t, t], np.identity(2), n)
X = np.vstack((X, normal([t, t], spread, n)))
X = np.vstack((X, normal([t, -t], spread, n)))
X = np.vstack((X, normal([-t, -t], spread, n)))

centroids = np.array([[-3, 3], [3, 3], [3, -3], [-3, -3]], dtype=float)

for iteration in range(10):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    new_cent = np.array([X[labels == k].mean(axis=0) for k in range(4)])

    if np.all(centroids == new_cent):
        print(f"Converged at iteration {iteration}")
        break
    centroids = new_cent

print("Final Optimized Centriods:\n", centroids)

colors = ['r', 'g', 'b', 'y']
plt.figure(figsize=(8,6))

for k in range(4):
    cluster_points = X[labels == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], s=1, alpha=0.3, label=f'Cluster {k}')

    plt.scatter(centroids[k, 0], centroids[k, 1], c='black', marker='x', s=100, linewidths=3)

plt.title('Converged K-Means Clusters')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()