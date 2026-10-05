"""Open planar SOM and a shared trainer for controlled topology comparisons.

``paper`` reproduces the moving-weight distance gate in SphereSOM3D.train.
``lattice`` uses a Gaussian on fixed graph distances for BOTH map topologies.
Weights remain unconstrained 3-D vectors; the planar *connectivity* stays fixed.
"""

import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def lattice_edges(mesh):
    """Unique undirected edges of a triangular lattice."""
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                            triangles[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


class TopologySOM3D:
    """A SOM with fixed connectivity and moving, unconstrained feature weights."""

    def __init__(self, mesh, radius=0.1, edges=None, boundary_mask=None):
        if radius <= 0 or not np.isfinite(radius):
            raise ValueError("radius must be finite and positive")
        self.mesh = mesh.copy(deep=True)
        self.positions = np.asarray(mesh.points, dtype=float).copy()
        self.n_nodes = mesh.n_points
        self.radius = float(radius)
        self.edges = lattice_edges(mesh) if edges is None else np.asarray(edges)
        self.adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=bool)
        self.adjacency[self.edges[:, 0], self.edges[:, 1]] = True
        self.adjacency[self.edges[:, 1], self.edges[:, 0]] = True
        self._adj = ([set(int(j) for j in mesh.point_neighbors(i))
                      for i in range(self.n_nodes)] if edges is None else
                     [set(np.flatnonzero(self.adjacency[i]).tolist())
                      for i in range(self.n_nodes)])
        self.graph_distances = shortest_path(csr_matrix(self.adjacency),
                                             directed=False, unweighted=True)
        if not np.all(np.isfinite(self.graph_distances)):
            raise ValueError("The SOM lattice must be connected")
        self.boundary_mask = (np.zeros(self.n_nodes, dtype=bool)
                              if boundary_mask is None else
                              np.asarray(boundary_mask, dtype=bool).copy())
        self.weights = self.positions.copy()
        self._ring_cache = {}

    def get_n_ring_neighbors(self, mu_idx, n):
        if n < 0:
            return np.arange(self.n_nodes)
        key = (int(mu_idx), int(n))
        if key not in self._ring_cache:
            # Preserve SphereSOM3D's set iteration order: its sequential paper
            # updates use a live BMU position and can depend on that order.
            visited, current, neighbors = set(), {int(mu_idx)}, {int(mu_idx)}
            for _ in range(n):
                following = set()
                for node in current:
                    following.update(self._adj[node])
                following -= visited
                neighbors.update(following)
                visited.update(current)
                current = following
            self._ring_cache[key] = neighbors
        return self._ring_cache[key]

    def initialize(self, data, init="pca", init_neuron_size=2.0, seed=0):
        """PCA sheet, isotropic Gaussian, sampled descriptors, or enclosing sphere."""
        data = self._check_data(data)
        if init_neuron_size <= 0 or not np.isfinite(init_neuron_size):
            raise ValueError("init_neuron_size must be finite and positive")
        rng = np.random.default_rng(seed)
        if init == "pca":
            centered = data - data.mean(axis=0)
            eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(data))
            order = np.argsort(eigenvalues)[::-1][:2]
            coordinates = self.positions[:, :2].copy()
            extent = np.ptp(coordinates, axis=0)
            coordinates = 2 * (coordinates - coordinates.min(axis=0)) / np.where(extent > 0, extent, 1) - 1
            basis = eigenvectors[:, order].T
            # Fix PCA sign ambiguity for reproducibility across implementations.
            for vector in basis:
                if vector[np.argmax(np.abs(vector))] < 0:
                    vector *= -1
            self.weights = (data.mean(axis=0) + init_neuron_size *
                            (coordinates * np.sqrt(np.maximum(eigenvalues[order], 0))) @ basis)
        elif init == "random":
            self.weights = rng.normal(0, init_neuron_size / np.sqrt(3), (self.n_nodes, 3))
        elif init == "sample":
            self.weights = data[rng.integers(len(data), size=self.n_nodes)].copy()
        elif init == "enclosing":
            lengths = np.linalg.norm(self.positions, axis=1, keepdims=True)
            if np.any(lengths == 0):
                raise ValueError("Enclosing initialization needs nonzero spherical positions")
            self.weights = init_neuron_size * self.positions / lengths
        else:
            raise ValueError(f"Unknown initialization: {init}")
        self.initial_weights = self.weights.copy()
        return self.weights

    @staticmethod
    def _check_data(data):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 3 or len(data) == 0 or not np.all(np.isfinite(data)):
            raise ValueError("Expected a nonempty finite (N, 3) descriptor array")
        return data

    def neighborhood_support(self, sigma, n_rings=-1):
        """Total lattice-Gaussian influence for every possible BMU (includes itself)."""
        if sigma <= 0 or not np.isfinite(sigma):
            raise ValueError("sigma must be finite and positive")
        kernel = np.exp(-self.graph_distances**2 / (2 * sigma**2))
        if n_rings >= 0:
            kernel[self.graph_distances > n_rings] = 0
        return kernel.sum(axis=1)

    def train(self, data, n_epochs=2000, n_rings=-1, init_neuron_size=2.0,
              lr0=0.2, lr_end=0.01, verbose=True, *, init="pca", seed=0,
              kernel="lattice", sigma0=3.0, sigma_end=0.5,
              sample_indices=None, initial_weights=None):
        data = self._check_data(data)
        if not isinstance(n_epochs, (int, np.integer)) or n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer (one sample per step)")
        if not isinstance(n_rings, (int, np.integer)) or n_rings < -1:
            raise ValueError("n_rings must be -1 (all nodes) or a nonnegative integer")
        if min(lr0, lr_end, sigma0, sigma_end) <= 0 or not np.all(np.isfinite([lr0, lr_end, sigma0, sigma_end])):
            raise ValueError("Learning rates and lattice sigmas must be finite and positive")
        if kernel not in ("paper", "lattice"):
            raise ValueError("kernel must be 'paper' or 'lattice'")
        if initial_weights is None:
            self.initialize(data, init, init_neuron_size, seed)
        else:
            weights = np.asarray(initial_weights, dtype=float)
            if weights.shape != (self.n_nodes, 3) or not np.all(np.isfinite(weights)):
                raise ValueError("initial_weights must be a finite (n_nodes, 3) array")
            self.weights = weights.copy()
            self.initial_weights = weights.copy()
        if sample_indices is None:
            # RandomState matches np.random.seed(seed) + randint in the original.
            sample_indices = np.random.RandomState(seed).randint(len(data), size=n_epochs)
        sample_indices = np.asarray(sample_indices)
        if (sample_indices.shape != (n_epochs,) or sample_indices.dtype.kind not in "iu"
                or np.any(sample_indices < 0) or np.any(sample_indices >= len(data))):
            raise ValueError("sample_indices must contain n_epochs valid integer data indices")
        self.sample_indices = sample_indices.copy()
        windows = (self.graph_distances <= n_rings if n_rings >= 0 else
                   np.ones_like(self.adjacency))
        ring_nodes = ([self.get_n_ring_neighbors(node, n_rings) for node in range(self.n_nodes)]
                      if kernel == "paper" else None)
        neighbor_count = 0
        multi_updates = 0
        if verbose:
            print(f"Training {self.n_nodes} neurons: {kernel} kernel, {n_epochs} sample updates")
        for step, sample_index in enumerate(sample_indices):
            fraction = step / n_epochs  # identical learning-rate schedule to SphereSOM3D
            lr = lr0 * (lr_end / lr0)**fraction
            x = data[sample_index]
            bmu = int(np.argmin(np.sum((self.weights - x)**2, axis=1)))
            if kernel == "lattice":
                sigma = sigma0 * (sigma_end / sigma0)**fraction
                influence = np.exp(-self.graph_distances[bmu]**2 / (2 * sigma**2))
                influence *= windows[bmu]
                self.weights += lr * influence[:, None] * (x - self.weights)
                updated_neighbors = np.count_nonzero((influence > 1e-6) & (np.arange(self.n_nodes) != bmu))
            else:
                # Keep the original live BMU view and sequential update semantics.
                # The radius gate uses CURRENT feature weights, not grid coordinates.
                bmu_position = self.weights[bmu]
                updated_neighbors = 0
                for node in ring_nodes[bmu]:
                    distance = np.linalg.norm(self.weights[node] - bmu_position)
                    if distance <= self.radius:
                        influence = np.exp(-distance**2 / (2 * self.radius**2))
                        self.weights[node] += lr * influence * (x - self.weights[node])
                        updated_neighbors += int(node != bmu)
            neighbor_count += updated_neighbors
            multi_updates += int(updated_neighbors > 0)
        self.training_stats = {
            "mean_updated_non_bmu_neurons": neighbor_count / n_epochs,
            "fraction_updates_with_non_bmu": multi_updates / n_epochs,
        }
        self.mesh.points = self.weights.copy()
        return self

    def predict(self, data, batch_size=2048):
        data = self._check_data(data)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        labels = np.empty(len(data), dtype=np.int64)
        for start in range(0, len(data), batch_size):
            chunk = data[start:start + batch_size]
            distances = np.sum((chunk[:, None] - self.weights[None, :])**2, axis=2)
            labels[start:start + len(chunk)] = np.argmin(distances, axis=1)
        return labels

    def get_weights(self):
        return self.weights

    def get_mesh(self):
        self.mesh.points = self.weights.copy()
        return self.mesh

    def wireframe(self, trained=True):
        points = self.weights if trained else self.positions
        lines = np.column_stack([np.full(len(self.edges), 2), self.edges]).ravel()
        return pv.PolyData(points.copy(), lines=lines)

    def boundary_wireframe(self, trained=True):
        edges = self.edges[np.all(self.boundary_mask[self.edges], axis=1)]
        # Only actual outer edges, not a diagonal joining two boundary nodes.
        if hasattr(self, "boundary_edges"):
            edges = self.boundary_edges
        lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
        return pv.PolyData((self.weights if trained else self.positions).copy(), lines=lines)


class PlanarSOM3D(TopologySOM3D):
    """An open staggered triangular grid (six interior neighbors), or square grid.

    There is no wrap-around. The rim is a property of connectivity, even when
    trained prototypes bend into a curved sheet in normal space.
    """

    def __init__(self, rows=9, cols=18, radius=0.1, grid="hex"):
        if rows < 2 or cols < 2:
            raise ValueError("The planar grid needs at least two rows and columns")
        if grid not in ("hex", "rect"):
            raise ValueError("grid must be 'hex' or 'rect'")
        self.rows, self.cols, self.grid = rows, cols, grid
        row, col = np.indices((rows, cols))
        x = col.astype(float) + (0.5 * (row % 2) if grid == "hex" else 0)
        y = row.astype(float) * (np.sqrt(3) / 2 if grid == "hex" else 1)
        points = np.column_stack([x.ravel(), y.ravel(), np.zeros(rows * cols)])
        points[:, :2] -= points[:, :2].mean(axis=0)
        triangles = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                a, b = r * cols + c, (r + 1) * cols + c
                if grid == "hex" and r % 2 == 0:
                    triangles.extend([(a, a + 1, b), (a + 1, b + 1, b)])
                else:
                    triangles.extend([(a, a + 1, b + 1), (a, b + 1, b)])
        triangles = np.asarray(triangles, dtype=np.int64)
        faces = np.column_stack([np.full(len(triangles), 3), triangles]).ravel()
        mesh = pv.PolyData(points, faces)
        edges = lattice_edges(mesh)
        if grid == "rect":
            delta = np.abs(edges[:, 0] - edges[:, 1])
            edges = edges[(delta == 1) | (delta == cols)]
            quads = [(r * cols + c, r * cols + c + 1,
                      (r + 1) * cols + c + 1, (r + 1) * cols + c)
                     for r in range(rows - 1) for c in range(cols - 1)]
            faces = np.column_stack([np.full(len(quads), 4), np.asarray(quads)]).ravel()
            mesh = pv.PolyData(points, faces)
        boundary = ((row == 0) | (row == rows - 1) | (col == 0) | (col == cols - 1)).ravel()
        super().__init__(mesh, radius, edges, boundary)
        # Triangle edge multiplicity identifies the true outer rim.
        all_edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                            triangles[:, [2, 0]]]), axis=1)
        unique, counts = np.unique(all_edges, axis=0, return_counts=True)
        self.boundary_edges = unique[counts == 1]


def matched_grid_shape(n_nodes):
    """Closest-to-square factorization, preserving the exact spherical node count."""
    factors = [(r, n_nodes // r) for r in range(2, int(np.sqrt(n_nodes)) + 1)
               if n_nodes % r == 0]
    if not factors:
        raise ValueError(f"Cannot make a rectangular planar grid with {n_nodes} nodes")
    return min(factors, key=lambda shape: shape[1] / shape[0])
