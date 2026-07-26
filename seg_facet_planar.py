"""Ablation of seg_facet.py: the ONLY change is the SOM lattice topology.

seg_facet.py       : closed spherical lattice (SphereSOM3D on an icosphere) -- no
                     boundary neurons, every node has a full ring of neighbours.
seg_facet_planar.py: traditional planar SOM -- an open H x W grid (default 9x18 =
                     162 nodes, matching the icosphere) whose four sides are
                     boundary neurons with truncated neighbourhoods.

Everything else (training rule, learning-rate schedule, small-face merge,
disconnected-component separation, power-based region merge, the 2x3 plot layout
and the live power_thr slider) is copied from seg_facet.py unchanged.

What to look for in subplot (0, 1): the planar map's boundary neurons are drawn
with a red ring. Face normals live on the closed sphere S^2, so a flat open sheet
must be cut/folded to cover them: normal clusters that straddle the cut are split
across the map edge (two far-apart groups of neurons share nearly the same normal
direction), while the closed S-SOM lattice of seg_facet.py has no edge to split on.

Usage (same flags as seg_facet.py, plus the grid size and the init scheme):
  python seg_facet_planar.py --input ./datasets/3DPuzzle/brick_part01.obj
  python seg_facet_planar.py --grid_h 9 --grid_w 18 --n_rings 3 --radius 0.8
  python seg_facet_planar.py --init random --seed 0   # random weight init (PCA is the default)
"""
import numpy as np
import pyvista as pv
from SSOM3D import *
from helper import *
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


class PlanarSOM3D:
    """Traditional planar SOM: SphereSOM3D with the icosphere lattice replaced by
    an open H x W rectangular grid (4-neighbourhood). Same API, same update rule,
    same learning-rate schedule -- only the neighbourhood graph differs, so any
    difference in the result is attributable to the lattice topology."""

    def __init__(self, grid_h: int, grid_w: int, radius: float = 0.1):
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.n_nodes = self.grid_h * self.grid_w
        self.radius = radius
        # Fixed grid adjacency (the planar counterpart of the icosphere edges).
        self._adj = [set() for _ in range(self.n_nodes)]
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                i = r * self.grid_w + c
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.grid_h and 0 <= cc < self.grid_w:
                        self._adj[i].add(rr * self.grid_w + cc)
        self._ring_cache = {}
        self.points = None

    def boundary_mask(self):
        """True for the map-edge neurons (truncated neighbourhood). The closed
        spherical lattice of seg_facet.py has no such neurons at all."""
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

    def train(self, data: np.ndarray, n_epochs: int = 1000, n_rings: int = 0,
              init_neuron_size: float = 2, lr0: float = 0.1, lr_end: float = 0.01,
              init: str = "pca", verbose: bool = True):
        """Identical to SphereSOM3D.train, except the initial node layout, chosen
        by `init`:

        init="pca"    (default) a flat sheet spanning the two leading principal
                      directions of the data (the textbook planar-SOM linear
                      initialisation).
        init="random" weights drawn from an isotropic Gaussian (uniform random
                      directions), i.e. the classic random initialisation.

        Either way the weights are rescaled so their mean radius equals
        init_neuron_size, like the sphere init of seg_facet.py."""
        if init == "pca":
            X = np.asarray(data, dtype=float)
            mu = X.mean(axis=0)
            _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
            a = np.linspace(-1.0, 1.0, self.grid_h)[:, None] * (s[0] / np.sqrt(len(X)))
            b = np.linspace(-1.0, 1.0, self.grid_w)[:, None] * (s[1] / np.sqrt(len(X)))
            W = (mu[None, None, :] + a[:, None, None] * vt[0][None, None, :]
                 + b[None, :, None] * vt[1][None, None, :]).reshape(-1, 3)
        elif init == "random":
            W = np.random.randn(self.n_nodes, 3)
        else:
            raise ValueError(f"unknown init {init!r}; expected 'pca' or 'random'")
        W *= init_neuron_size / max(np.linalg.norm(W, axis=1).mean(), 1e-12)
        points = W

        if verbose:
            print(f"Training planar SOM ({self.grid_h}x{self.grid_w}, {init} init) with {n_epochs} epochs, "
                  f"initial radius {n_rings} rings, learning rate {lr0} decaying to {lr_end}...")

        for epoch in range(n_epochs):
            frac = epoch / n_epochs
            lr = lr0 * (lr_end / lr0) ** frac

            x = data[np.random.randint(0, len(data))]
            bmu_idx = np.argmin(np.linalg.norm(points - x, axis=1))
            bmu_pos = points[bmu_idx]
            searched_nodes = self.get_n_ring_neighbors(bmu_idx, n_rings)
            for i in searched_nodes:
                dist = np.linalg.norm(points[i] - bmu_pos)
                if dist <= self.radius:
                    influence = np.exp(-dist**2 / (2 * self.radius**2))
                    points[i] += lr * influence * (x - points[i])
        self.points = points

    def predict(self, data):
        # Euclidean nearest node (BMU), consistent with training (as in SphereSOM3D).
        data = np.asarray(data)
        sq_dists = (np.sum(data**2, axis=1)[:, None]
                    - 2.0 * (data @ self.points.T)
                    + np.sum(self.points**2, axis=1)[None, :])
        return np.argmin(sq_dists, axis=1)

    def get_weights(self) -> np.ndarray:
        return self.points

    def get_mesh(self) -> pv.PolyData:
        # Quad mesh of the (deformed) sheet, for the wireframe plot.
        h, w = self.grid_h, self.grid_w
        faces = []
        for r in range(h - 1):
            for c in range(w - 1):
                faces.extend([4, r * w + c, r * w + c + 1,
                              (r + 1) * w + c + 1, (r + 1) * w + c])
        return pv.PolyData(self.points, faces=np.asarray(faces, dtype=np.int64))


def get_color_map(class_count):
    cmap = plt.get_cmap('gist_rainbow')
    colors = []
    i = 0
    while (len(colors) < class_count):
        c = cmap(i / class_count)[:3]  # Use n+1 to avoid endpoints
        if not np.allclose(c, (0, 0, 0)):  # Exclude black
            colors.append(mcolors.to_hex(c))
        i += 1
    return colors


def plot(plotter, mesh, scalar_name, class_count, max_bar_labels=20):
    assert(class_count > 0)
    colors = get_color_map(class_count)
    n_labels = min(class_count, max_bar_labels)
    plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.3, scalar_bar_args={"fmt": "%.0f", "n_labels": n_labels})


def main(input, radius, n_rings, init_neuron_size, lr, grid_h, grid_w,
         power_thr=0.15, max_merges=1000, init="pca"):

    obj_mesh = load_obj_with_face_normals(input)
    face_adjacency = build_face_adjacency(obj_mesh)

    # Feature-preserving normal-field smoothing to denoise the SOM input (reduces
    # salt-and-pepper labels). Geometry is untouched; sharp edges are preserved.
    smooth_normals(obj_mesh, face_adjacency, n_iter=5, feature_angle_deg=30.0)

    import time
    start_time = time.time()
    data_for_som = obj_mesh.cell_data['Normals']   # already unit-length (compute_normals + smooth_normals)

    # >>> THE ONLY PIPELINE CHANGE vs seg_facet.py: planar grid instead of icosphere.
    som = PlanarSOM3D(grid_h, grid_w, radius=radius)
    som.train(data_for_som, n_epochs=2000, n_rings=n_rings, init_neuron_size=init_neuron_size, lr0=lr, init=init)

    #Predict labels
    bmu_labels = som.predict(data_for_som)                         # node id per face
    bmu_labels = merge_small_faces(obj_mesh, bmu_labels, face_adjacency, area_ratio=0.03) # Merge small faces into their largest neighbor's cluster (if the face is <5% of the average face area)

    raw_labels, raw_labels_count = remap_labels(bmu_labels, mesh=obj_mesh)  # Convert to face labels 0-based indices
    print("Planar SOM clustering: there are {} clusters".format(raw_labels_count))
    obj_mesh.cell_data["raw_labels"] = raw_labels # Assign cluster labels to each face

    #Separate disconnected components
    separated_region_labels = separate_disconnected_components(obj_mesh, face_adjacency, raw_labels)
    separated_region_labels, _ = remap_labels(separated_region_labels, mesh=obj_mesh)  # Convert to face labels 0-based indices

    separated_region_labels, _ = remap_labels(separated_region_labels, mesh=obj_mesh)
    obj_mesh.cell_data["separated_region_labels"] = separated_region_labels

    # The power-based region merge is driven by the power_thr slider in
    # subplot(1, 1) (see render_segmentation below), so it is not run here.

    end_time = time.time()
    running_time = end_time - start_time
    print(f"Total running time: {running_time:.2f} seconds")

    # Start plotting, Create a 2x3 grid plotter ##################################################
    plotter = pv.Plotter(shape=(2, 3), title= f"Facet segmentation by planar SOM (topology ablation, {init} init)")

    plotter.subplot(0, 0) #-----------------------------------------------
    plotter.add_mesh(obj_mesh, color='grey', show_edges=True, edge_opacity=0.2)

    plotter.subplot(0, 1) #-----------------------------------------------
    plotter.add_mesh(pv.Sphere(radius=1.0), color='white', opacity=0.15)

    # Per-node colors. Winning neurons (BMU of >=1 face) get their label color
    # (matching the segmentation plots); neurons that win no data are black.
    neuron_points = som.get_weights()
    n_nodes = len(neuron_points)
    winners = np.unique(bmu_labels)                       # neurons that are a BMU (node ids)
    node_to_seg = {node: i for i, node in enumerate(winners)}
    seg_colors_rgb = (np.array([mcolors.to_rgb(c) for c in get_color_map(raw_labels_count)]) * 255).astype(np.uint8)
    neuron_rgb = np.zeros((n_nodes, 3), dtype=np.uint8)   # black by default (no winning data)
    for i in winners:
        neuron_rgb[i] = seg_colors_rgb[node_to_seg[i]]

    # 1) The SOM input as a point cloud on the unit sphere: each face normal is a
    #    unit vector, so it lands on the surface of the unit sphere. Color each data
    #    point by the color of its BMU neuron.
    data_cloud = pv.PolyData(np.asarray(data_for_som, dtype=float))
    data_cloud.point_data["cluster"] = raw_labels              # cluster id per face (BMU)
    plotter.add_mesh(data_cloud, scalars="cluster", cmap=get_color_map(raw_labels_count),
                     render_points_as_spheres=True, point_size=5, opacity=0.6,  show_scalar_bar=True, label="data normals",
                     scalar_bar_args={"title": "cluster", "fmt": "%.0f", "n_labels": min(raw_labels_count, 20), "vertical": False})

    # 2) The trained SOM grid (node topology, deformed to where the neurons moved).
    plotter.add_mesh(som.get_mesh(), style='wireframe', color='grey',
                     opacity=0.5, line_width=1, label="SOM grid")

    # 3) The map-edge (boundary) neurons, in red: the open sheet's border. This is
    #    what the closed spherical lattice of seg_facet.py does not have -- normal
    #    clusters cut by this border are split across the map.
    boundary = som.boundary_mask()
    plotter.add_mesh(pv.PolyData(neuron_points[boundary]).glyph(scale=False, geom=pv.Sphere(radius=0.045)),
                     color='red', label="boundary neurons")

    # 4) The trained SOM neurons, colored as computed above.
    neuron_glyphs = pv.PolyData(neuron_points).glyph(scale=False, geom=pv.Sphere(radius=0.03))
    points_per_neuron = neuron_glyphs.n_points // n_nodes   # divides evenly: same geom per node
    neuron_glyphs.point_data["colors"] = np.repeat(neuron_rgb, points_per_neuron, axis=0)
    plotter.add_mesh(neuron_glyphs, scalars="colors", rgb=True, label="SOM neurons")
    #plotter.add_legend()

    plotter.subplot(1, 0) #-----------------------------------------------
    plot(plotter, obj_mesh, "raw_labels", raw_labels_count)

    plotter.subplot(1, 1) #-----------------------------------------------
    obj_mesh01 = obj_mesh.copy()
    plot(plotter, obj_mesh01, "separated_region_labels",  len(np.unique(separated_region_labels)))

    plotter.subplot(1, 2) #-----------------------------------------------
    # Live power_thr control: re-run ONLY the region merge (SOM training is not
    # repeated) whenever the slider is released, and redraw this subplot.
    obj_mesh12 = obj_mesh.copy()

    # Holds the most recently rendered merge so it can be saved after the
    # interactive session closes (reflects the final power_thr the user explored).
    latest = {"labels": None}

    def render_segmentation(power_thr_val):
        merged = merge_region_based_on_power(
            obj_mesh, separated_region_labels.copy(), face_adjacency,
            power_thr=float(power_thr_val), max_merges=max_merges,
            target_n_regs=None, verbose=False)
        labels, count = remap_labels(merged)          # 0-based labels + region count
        count = max(int(count), 1)
        latest["labels"] = labels                     # remember for saving on exit
        obj_mesh12.cell_data["merged_similar_region_labels"] = labels
        plotter.subplot(1, 2)
        try:
            plotter.remove_scalar_bar("merged")        # avoid stacking on redraw
        except Exception:
            pass
        plotter.add_mesh(
            obj_mesh12, scalars="merged_similar_region_labels",
            cmap=get_color_map(count), show_edges=True, edge_opacity=0.3,
            show_scalar_bar=True, name="seg11",        # name => replaces old actor
            scalar_bar_args={"fmt": "%.0f", "n_labels": min(count, 20), "title": "merged"})
        plotter.add_text(f"power_thr = {float(power_thr_val):.3f}   |   {count} regions",
                         position="lower_left", font_size=9, name="seg11_info")
        return count

    render_segmentation(power_thr)                     # initial draw at the CLI value

    def on_power_thr(value):
        render_segmentation(value)
        plotter.render()

    plotter.subplot(1, 2)
    plotter.add_slider_widget(
        on_power_thr, rng=[0.0, 1.0], value=power_thr, title="power_thr",
        pointa=(0.30, 0.92), pointb=(0.95, 0.92), style="modern", fmt="%.3f")

    plotter.show()

    # Save the final (slider-tuned) segmentation: one face label per line. Written
    # to *_planar.seg (PCA init) or *_planar_random.seg (random init) so it never
    # overwrites the S-SOM result of seg_facet.py, nor the other init's result.
    merged_similar_region_labels = latest["labels"]
    if merged_similar_region_labels is not None:
        suffix = '_planar.seg' if init == "pca" else '_planar_random.seg'
        seg_path = input.replace('.obj', suffix)
        with open(seg_path, 'w') as f:
            for segment_index in merged_similar_region_labels:
                f.write(f'{segment_index}\n')
        print("segmentation saved to ", seg_path)

     #End plotting               ############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planar SOM surface segmentation (topology ablation of seg_facet.py).")
    parser.add_argument("--input", type=str, default = "./datasets/3DPuzzle/brick_part01.obj",  help="Path to the OBJ file")
    parser.add_argument("--radius", type=float, default=0.1, help="Neighborhood radius for SOM")
    parser.add_argument("--n_rings", type=int, default=0, help="Number of rings in the neighborhood for SOM training")
    parser.add_argument("--init_neu_size", type=float, default=2, help="Initial distance of neurons to the origin")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate for SOM training")
    parser.add_argument("--power_thr", type=float, default=0.39, help="stop when best remaining power < this value")
    parser.add_argument("--grid_h", type=int, default=9, help="Planar SOM grid height")
    parser.add_argument("--grid_w", type=int, default=18, help="Planar SOM grid width (9x18 = 162 nodes = the icosphere of seg_facet.py)")
    parser.add_argument("--init", type=str, choices=["pca", "random"], default="pca",
                        help="Weight initialisation: 'pca' = flat sheet spanning the two leading "
                             "principal directions of the data (textbook linear init, default); "
                             "'random' = isotropic Gaussian directions (classic random init)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for the weight initialisation and training sampling "
                             "(default: not seeded; mainly useful with --init random)")

    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    main(args.input, args.radius, args.n_rings, args.init_neu_size, args.lr,
         args.grid_h, args.grid_w, args.power_thr, init=args.init)
