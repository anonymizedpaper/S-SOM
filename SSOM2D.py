import numpy as np
from sklearn.preprocessing import normalize
import pyvista as pv
from collections import defaultdict
import itertools
from collections import Counter

class SphereSOM:
    def __init__(self, mesh: pv.PolyData):
        self.mesh = mesh
        self.positions = normalize(mesh.points.copy())           # (N, 3) fixed node positions        
        self.n_nodes = mesh.n_points
        # self.lr = lr
        # self.radius = radius

    def _initialize_nodes(self, n_dim):
        # Every node weight is set to the same constant vector (all 100s), shape
        # (n_nodes, n_dim). This is a uniform, data-independent init placed far
        # outside the [0,1] feature range. NOTE: identical weights make argmin
        # pick node 0 for the first BMU, and the far offset lets only that node's
        # neighborhood ever update -> the map tends to collapse to one cluster.
        # Seeding from random data samples instead keeps all nodes competing.
        self.weights = np.ones((self.n_nodes, n_dim), dtype=np.float32)*2
    def get_n_ring_neighbors(self, mu_idx, n):
        visited = set()
        current_ring = set([mu_idx])
        all_neighbors = set()
        all_neighbors.add(mu_idx)
        for _ in range(n):
            next_ring = set()
            for vid in current_ring:
                neighbors = self.mesh.point_neighbors(vid)
                next_ring.update(neighbors)
            next_ring -= visited
            all_neighbors.update(next_ring)
            visited.update(current_ring)
            current_ring = next_ring
        return all_neighbors
    
    def train(self, data: np.ndarray, n_epochs: int = 1000, n_rings: int = 2, lr: float = 0.1, radius: float = 0.2):
        self._initialize_nodes(data.shape[1])
        for epoch in range(n_epochs):
            x = data[np.random.randint(0, len(data))]

            # Find BMU (closest in weight space)
            bmu_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
            bmu_pos = self.positions[bmu_idx]

            searched_nodes = self.get_n_ring_neighbors(bmu_idx, n_rings) if n_rings >= 0 else np.arange(self.n_nodes)
            for i in searched_nodes:
                dist = np.linalg.norm(self.positions[i] - bmu_pos)
                if dist <= radius:
                    influence = np.exp(-dist**2 / (2 * radius**2))
                    self.weights[i] += lr * influence * (x - self.weights[i])   


    def predict(self, data):
        bmu_indices = []
        for item in data:
            distances = np.linalg.norm(self.weights - item, axis=1)
            bmu_idx = np.argmin(distances)
            bmu_indices.append(bmu_idx)
        return np.array(bmu_indices)      
    
    def get_weights(self) -> np.ndarray:
        return self.weights
    
    def get_mesh(self) -> pv.PolyData:
        self.mesh.points = self.weights[:, :3]
        return self.mesh
    