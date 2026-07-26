"""Ablation of seg_part.py: the ONLY change is the SOM lattice topology.

seg_part.py        : closed spherical lattice (SSOM2D.SphereSOM on an icosphere) --
                     the node POSITIONS are fixed on the unit sphere and the
                     neighborhood is Gaussian over the position distance, so every
                     node has a full ring of neighbours and there is no map border.
seg_part_planar.py : traditional planar SOM -- the fixed node positions form an
                     open H x W grid (default 9x18 = 162 nodes, matching the
                     icosphere) whose four sides are boundary neurons with
                     truncated neighbourhoods. The grid pitch is set to the mean
                     edge length of the icosphere lattice, so the Gaussian
                     neighborhood radius means the same thing in both arms.

The weights live in the feature space (SDF / curvature), exactly as in seg_part.py.
Two standard weight initialisations are provided (--init):
  pca    (default) the textbook linear initialisation: the grid spans the leading
         principal directions of the feature data.
  random weights drawn uniformly within the per-dimension range of the data.

Everything else (SDF + curvature features, training rule, small-face merge,
disconnected-component separation, power-based region merge, the 2x3 plot layout
and the live power_thr / alpha sliders) is copied from seg_part.py unchanged.

Usage (same flags as seg_part.py, plus the grid size and the init scheme):
  python seg_part_planar.py --input ./datasets/Princeton/1.obj
  python seg_part_planar.py --input ./datasets/Princeton/1.obj --init random --seed 0
"""
import numpy as np
import pyvista as pv
from helper import *
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pymeshlab
import os

from seg_part import (get_color_map, plot, comp_cur, clamp_percentiles,
                      smooth_face_scalar, make_feature_distribution_figure)


def icosphere_mean_edge_length(obj_path="regular_sphere.obj"):
    """Mean edge length of the (unit-normalized) icosphere lattice of seg_part.py.
    Used as the planar grid pitch so both arms share the same neighborhood scale."""
    try:
        from sklearn.preprocessing import normalize
        mesh = pv.read(obj_path)
        pts = normalize(mesh.points)
        lengths = []
        for i in range(mesh.n_points):
            for j in mesh.point_neighbors(i):
                if j > i:
                    lengths.append(np.linalg.norm(pts[i] - pts[j]))
        return float(np.mean(lengths))
    except Exception:
        return 0.28   # typical value for the 162-node icosphere


class PlanarSOM:
    """Traditional planar SOM: SSOM2D.SphereSOM with the fixed spherical node
    positions replaced by an open H x W rectangular grid (4-neighbourhood).
    Same API, same update rule -- only the lattice differs, so any difference in
    the result is attributable to the lattice topology."""

    def __init__(self, grid_h: int, grid_w: int, pitch: float = 0.28):
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.n_nodes = self.grid_h * self.grid_w
        # Fixed node positions: a flat sheet with the icosphere's edge length.
        rr, cc = np.meshgrid(np.arange(self.grid_h), np.arange(self.grid_w), indexing="ij")
        self.positions = np.column_stack([rr.ravel() * pitch, cc.ravel() * pitch,
                                          np.zeros(self.n_nodes)])
        # Fixed grid adjacency (the planar counterpart of the icosphere edges).
        self._adj = [set() for _ in range(self.n_nodes)]
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                i = r * self.grid_w + c
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < self.grid_h and 0 <= c2 < self.grid_w:
                        self._adj[i].add(r2 * self.grid_w + c2)
        self._ring_cache = {}
        self.weights = None

    def boundary_mask(self):
        """True for the map-edge neurons (truncated neighbourhood). The closed
        spherical lattice of seg_part.py has no such neurons at all."""
        m = np.zeros(self.n_nodes, dtype=bool)
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                if r in (0, self.grid_h - 1) or c in (0, self.grid_w - 1):
                    m[r * self.grid_w + c] = True
        return m

    def get_n_ring_neighbors(self, mu_idx, n):
        key = (int(mu_idx), int(n))
        cached = self._ring_cache.get(key)
        if cached is not None:
            return cached
        visited = set()
        current_ring = {int(mu_idx)}
        all_neighbors = {int(mu_idx)}
        for _ in range(n):
            next_ring = set()
            for vid in current_ring:
                next_ring.update(self._adj[vid])
            next_ring -= visited
            all_neighbors.update(next_ring)
            visited.update(current_ring)
            current_ring = next_ring
        self._ring_cache[key] = all_neighbors
        return all_neighbors

    def _initialize_nodes(self, data, init):
        """Initial weights in FEATURE space, chosen by `init`:

        init="pca"    the textbook planar-SOM linear initialisation: the grid
                      spans the leading principal directions of the data (one
                      direction per grid axis; with 1-D features only the grid
                      width spans the data).
        init="random" weights drawn uniformly within the per-dimension range of
                      the data (the classic random-value initialisation).
        """
        X = np.asarray(data, dtype=float)
        n_dim = X.shape[1]
        if init == "pca":
            mu = X.mean(axis=0)
            _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
            b = np.linspace(-1.0, 1.0, self.grid_w)[:, None] * (s[0] / np.sqrt(len(X)))
            W = mu[None, None, :] + b[None, :, :] * vt[0][None, None, :]
            W = np.broadcast_to(W, (self.grid_h, self.grid_w, n_dim)).copy()
            if n_dim >= 2 and len(s) >= 2:
                a = np.linspace(-1.0, 1.0, self.grid_h)[:, None] * (s[1] / np.sqrt(len(X)))
                W += a[:, None, :] * vt[1][None, None, :]
            self.weights = W.reshape(self.n_nodes, n_dim).astype(np.float32)
        elif init == "random":
            lo, hi = X.min(axis=0), X.max(axis=0)
            self.weights = np.random.uniform(lo, hi, (self.n_nodes, n_dim)).astype(np.float32)
        else:
            raise ValueError(f"unknown init {init!r}; expected 'pca' or 'random'")

    def train(self, data: np.ndarray, n_epochs: int = 1000, n_rings: int = 2,
              lr: float = 0.1, radius: float = 0.2, init: str = "pca"):
        """Identical to SSOM2D.SphereSOM.train: BMU in weight space, neighborhood
        Gaussian over the FIXED lattice positions within an n-ring window, gated by
        `radius`. Only the initialisation (see _initialize_nodes) differs from the
        constant init of seg_part.py, which on the planar lattice would collapse
        the map to a single cluster just as it tends to on the sphere."""
        self._initialize_nodes(data, init)
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
        data = np.asarray(data)
        sq_dists = (np.sum(data**2, axis=1)[:, None]
                    - 2.0 * (data @ self.weights.T)
                    + np.sum(self.weights**2, axis=1)[None, :])
        return np.argmin(sq_dists, axis=1)

    def get_weights(self) -> np.ndarray:
        return self.weights


def main(input, radius, n_rings, lr, power_thr=0.15, max_merges=10000,
         feature_name="sdf", visualize=True, sdf_cap_pct=1, sdf_smooth_iter=5,
         grid_h=9, grid_w=18, init="pca"):
    obj_mesh = load_obj_with_face_normals(input)
    face_adjacency = build_face_adjacency(obj_mesh)

    cur = comp_cur(obj_mesh)
    cur = cur/np.max(cur)
    cur = cur.reshape(-1, 1)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input)
    ms.compute_scalar_by_shape_diameter_function_per_vertex_gpu(coneangle = 120, onprimitive = 'On Faces', removeoutliers = True, numberrays = 180)
    sdf = ms.current_mesh().face_scalar_array()
    sdf = smooth_face_scalar(obj_mesh, face_adjacency, sdf, n_iter=sdf_smooth_iter)  # Gaussian-smooth over face neighbors
    sdf = clamp_percentiles(sdf, sdf_cap_pct)   # clamp lowest/highest sdf_cap_pct% and rescale to [0,1]
    sdf = sdf.reshape(-1, 1)

    if feature_name == "sdf": # Select feature set based on CLI argument
        features = sdf
    elif feature_name == "cur":
        features = cur
    elif feature_name == "sdf_cur":
        features = np.concatenate((sdf, cur), axis=1)
        norms = np.linalg.norm(features, axis=1)
        max_norm = float(np.max(norms))
        features = features/max_norm
    else:
        raise ValueError(f"Unsupported features option '{feature_name}'. Use 'sdf', 'cur' or 'sdf_cur'.")

    obj_mesh.cell_data['features'] = features

    # >>> THE ONLY PIPELINE CHANGE vs seg_part.py: planar grid instead of icosphere.
    som = PlanarSOM(grid_h, grid_w, pitch=icosphere_mean_edge_length())
    som.train(features, n_epochs=1000, n_rings=n_rings if n_rings > 0 else 2,
              lr=lr, radius=radius, init=init)

    #Predict labels
    bmu_labels = som.predict(features)
    print("raw_labels", bmu_labels)
    bmu_labels = merge_small_faces(obj_mesh, bmu_labels, face_adjacency, area_ratio=0.03) # Merge small faces into their largest neighbor's cluster (if the face is <5% of the average face area)

    raw_labels, raw_labels_count = remap_labels(bmu_labels)  # Convert to face labels 0-based indices
    print("Planar SOM ({} init) clustering: there are {} clusters".format(init, raw_labels_count))
    obj_mesh.cell_data["raw_labels"] = raw_labels # Assign cluster labels to each face

    #Separate disconnected components
    separated_region_labels = separate_disconnected_components(obj_mesh, face_adjacency, raw_labels)
    separated_region_labels, _ = remap_labels(separated_region_labels, mesh=obj_mesh)
    obj_mesh.cell_data["separated_region_labels"] = separated_region_labels

    # Merge region based on power
    merged_region_label = separated_region_labels.copy()
    merged_region_label = merge_region_based_on_power(obj_mesh, merged_region_label, face_adjacency,  power_thr=power_thr, max_merges=max_merges, target_n_regs=None, feature_name = "features")

    merged_similar_region_labels, _ = remap_labels(merged_region_label)  # Convert to face labels 0-based indices
    obj_mesh.cell_data["merged_similar_region_labels"] = merged_similar_region_labels

    print("merged_similar_region_labels", merged_similar_region_labels)
    # Written to *_planar.seg (PCA init) or *_planar_random.seg (random init) so it
    # never overwrites the S-SOM result of seg_part.py, nor the other init's result.
    suffix = '_planar.seg' if init == "pca" else '_planar_random.seg'
    seg_path = input.replace('.obj', suffix)
    with open(seg_path, 'w') as f:
        for segment_index in merged_similar_region_labels:
            f.write(f'{segment_index}\n')
        print("segmentation saved to ", seg_path)

    # Create a 2x3 grid plotter #################################################################
    if visualize:
        plotter = pv.Plotter(shape=(2, 3), title= f"Part segmentation by planar SOM (topology ablation, {init} init)")

        # Add SDF visualization
        plotter.subplot(0, 0)
        plotter.add_text("SDF Values", font_size=12)
        obj_mesh_sdf = obj_mesh.copy()
        obj_mesh_sdf["sdf"] = sdf
        plotter.add_mesh(obj_mesh_sdf, scalars='sdf', cmap = "jet", show_scalar_bar=True, show_edges=True, edge_opacity=0.2)

        # Add curvature visualization
        plotter.subplot(0, 1)
        plotter.add_text("Curvature Values", font_size=12)
        obj_mesh_curv = obj_mesh.copy()
        plotter.add_mesh(obj_mesh_curv, scalars='Smoothed Mean Curvature',  cmap = "jet", show_scalar_bar=True, show_edges=True, edge_opacity=0.2)

        # 2D distribution of the per-face feature vectors, colored by SOM cluster.
        plotter.subplot(0, 2)
        plotter.add_text("Feature Distribution", font_size=12)
        feat_fig = make_feature_distribution_figure(features, feature_name, labels=raw_labels)
        plotter.add_chart(pv.ChartMPL(feat_fig))

        plotter.subplot(1, 0)
        plotter.add_text("Initial SOM Clustering", font_size=12)
        plot(plotter, obj_mesh, "raw_labels", raw_labels_count)

        plotter.subplot(1, 1)
        plotter.add_text("After Disconnected Component Separation", font_size=12)
        obj_mesh11 = obj_mesh.copy()
        plot(plotter,obj_mesh11, "separated_region_labels",  len(np.unique(separated_region_labels)))

        plotter.subplot(1, 2)
        plotter.add_text("Final Segmentation", font_size=12)
        obj_mesh12 = obj_mesh.copy()

        # Live controls: re-run ONLY the region merge whenever a slider is
        # released, and redraw this subplot. SOM/SDF stay fixed. Shared state
        # holds the current power_thr and alpha (beta = 1 - alpha).
        ctrl = {"power_thr": float(power_thr), "alpha": 1/3}

        def render_final_segmentation():
            alpha = float(ctrl["alpha"]); beta = 1.0 - alpha
            merged = merge_region_based_on_power(
                obj_mesh, separated_region_labels.copy(), face_adjacency,
                alpha=alpha, beta=beta,
                power_thr=ctrl["power_thr"], max_merges=max_merges,
                target_n_regs=None, feature_name="features", verbose=False)
            labels, count = remap_labels(merged)
            count = max(int(count), 1)
            obj_mesh12.cell_data["merged_similar_region_labels"] = labels
            plotter.subplot(1, 2)
            try:
                plotter.remove_scalar_bar("Segment ID")   # avoid stacking on redraw
            except Exception:
                pass
            plotter.add_mesh(
                obj_mesh12, scalars="merged_similar_region_labels",
                cmap=get_color_map(count), show_edges=True, edge_opacity=0.2,
                show_scalar_bar=True, name="final_seg",     # name => replaces old actor
                scalar_bar_args={"fmt": "%.0f", "n_labels": min(count, 20), "title": "Segment ID"})
            plotter.add_text(
                f"power_thr={ctrl['power_thr']:.3f}  alpha={alpha:.2f} beta={beta:.2f}  |  {count} regions",
                position="lower_left", font_size=9, name="final_seg_info")
            return count

        render_final_segmentation()                        # initial draw at CLI/default values

        def on_power_thr(value):
            ctrl["power_thr"] = float(value)
            render_final_segmentation()
            plotter.render()

        def on_alpha(value):
            ctrl["alpha"] = float(value)
            render_final_segmentation()
            plotter.render()

        plotter.subplot(1, 2)
        plotter.add_slider_widget(
            on_power_thr, rng=[0.0, 1.0], value=ctrl["power_thr"], title="power_thr",
            pointa=(0.30, 0.92), pointb=(0.95, 0.92), style="modern", fmt="%.3f")
        plotter.add_slider_widget(
            on_alpha, rng=[0.0, 1.0], value=ctrl["alpha"], title="alpha (beta=1-alpha)",
            pointa=(0.30, 0.78), pointb=(0.95, 0.78), style="modern", fmt="%.2f")

        plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planar SOM part segmentation (topology ablation of seg_part.py).")
    parser.add_argument("--radius", type=float, default=1, help="Neighborhood radius for SOM")
    parser.add_argument("--input", type=str, default = "./datasets/Princeton/1.obj",  help="Path to the OBJ file")
    parser.add_argument("--n_rings", type=int, default=0, help="Number of rings in the neighborhood for SOM training (<=0 uses the seg_part.py default of 2)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SOM training")
    parser.add_argument("--power_thr", type=float, default=0.2, help="stop when best remaining power < this value")
    parser.add_argument("--fea", type=str, default="sdf_cur", help="Features set to use")
    parser.add_argument("--sdf_cap_pct", type=float, default=1.0, help="Winsorize: clip the lowest/highest this %% of SDF values (0 disables)")
    parser.add_argument("--sdf_smooth_iter", type=int, default=5, help="Gaussian-smoothing iterations for per-face SDF over neighbors (0 disables)")
    parser.add_argument("--grid_h", type=int, default=9, help="Planar SOM grid height")
    parser.add_argument("--grid_w", type=int, default=18, help="Planar SOM grid width (9x18 = 162 nodes = the icosphere of seg_part.py)")
    parser.add_argument("--init", type=str, choices=["pca", "random"], default="pca",
                        help="Weight initialisation: 'pca' = grid spans the leading "
                             "principal directions of the feature data (textbook linear "
                             "init, default); 'random' = uniform within the data range")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for the weight initialisation and training sampling "
                             "(default: not seeded; mainly useful with --init random)")

    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    main(args.input, args.radius, args.n_rings, args.lr, args.power_thr,
         feature_name=args.fea, sdf_cap_pct=args.sdf_cap_pct,
         sdf_smooth_iter=args.sdf_smooth_iter,
         grid_h=args.grid_h, grid_w=args.grid_w, init=args.init)
