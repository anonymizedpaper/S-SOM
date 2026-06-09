"""Ablation on the S-SOM *topology*: spherical lattice (SphereSOM3D) vs a planar
2D-grid SOM, with everything else held equal.

Why a separate ablation from `ablation_seg_facet_3D.py`?
-------------------------------------------------------
The older script compares the two SOMs through the full segmentation pipeline in a
single run. That conflates two effects the paper attributes to the spherical map:
  (1) the lattice *topology* (closed icosphere vs an open H x W grid with a seam), and
  (2) the weight *initialization* (sphere-shell init vs random R^3 init).

This script isolates the topology by:
  * MATCHING the initialization (`--planar_init sphere`, the default): both maps
    start with weights on the same radius-2 shell, so the ONLY difference is the
    neighborhood graph used to spread updates.
  * SWEEPING the neighborhood size so the SOM actually self-organizes (otherwise the
    comparison only reflects the initial node layout, not learned topology).

Which knob controls the neighborhood here?
------------------------------------------
Training spreads each update over the BMU's `n_rings` graph neighborhood AND gates it
by a Euclidean `radius` (`if ||p_i - p_bmu|| <= radius`). On the radius-2 weight shell,
adjacent nodes are ~0.8 apart, so at the paper's default `radius=0.1` EVERY neighbor is
rejected and only the BMU updates -- both SOMs collapse to k-means and `n_rings` has no
effect (you will see a flat sweep). The *binding* neighborhood knob is therefore
`radius`. We sweep it by default (`--sweep radius`); `--sweep rings` is also available
but is only meaningful at a `radius` large enough to admit neighbors.

Headline metric (topology preservation)
---------------------------------------
  TE     : topographic error -- fraction of data whose 1st/2nd BMU are NOT lattice
           neighbors. The canonical SOM topology-preservation measure. lower = better.
           A closed spherical lattice whose index-adjacency matches the (closed) normal
           feature space keeps TE low; a planar grid's open boundary/seam cannot fold
           around the sphere, so consecutive-best nodes are often non-adjacent.
Supporting metrics
  edgeCV : std/mean of lattice-edge lengths in feature space. Low = neighbor weights
           stay close & regular. (Also rises as the map adapts to density, so read it
           alongside TE, not alone.)                                  lower = better
  QE     : quantization error (mean ||x - BMU||).                     lower = better
  dead   : fraction of nodes that never win a face.                   lower = better
  ARI    : adjusted Rand index vs ground-truth facets (synthetic), raw & final.
  final  : final segment count after the same downstream pipeline as seg_facet_3D.

Usage
-----
  python ablation_topology.py --synthetic --obj_file ./datasets/Synthetic/icosahedron.obj
  python ablation_topology.py --sweep radius --sweep_values 0.1 0.3 0.6 1.0 --seeds 5 --plot
  python ablation_topology.py --sweep rings --radius 0.8 --sweep_values 0 1 2 3 --show
  python ablation_topology.py --planar_init random           # faithful planar baseline
"""

import os
import argparse
import time

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from SSOM3D import (
    SphereSOM3D,
    load_obj_with_face_normals,
    build_face_adjacency,
    smooth_normals,
    separate_disconnected_components,
    merge_zero_area_regions,
    merge_region_based_on_power,
    remap_labels,
)

# Reuse the validated building blocks from the original ablation.
from ablation_seg_facet_3D import (
    PlanarSOM3D,
    sphere_adjacency,
    quantization_error,
    topographic_error,
    dead_fraction,
    ground_truth_labels,
    get_color_map,
    _plot_labels,
)

# Radius of the weight shell both maps start on when initializations are matched.
# Matches SphereSOM3D.train's default init_neuron_size and seg_facet_3D's default.
INIT_SHELL = 2.0


# --------------------------------------------------------------------------- #
# Planar SOM whose init can be MATCHED to the spherical SOM (radius-2 shell), so
# the comparison isolates the lattice topology from the initialization.
# --------------------------------------------------------------------------- #
class PlanarSOMTopo(PlanarSOM3D):
    def __init__(self, grid_h, grid_w, radius=0.1, seed=None,
                 init_mode="sphere", init_shell=INIT_SHELL):
        super().__init__(grid_h, grid_w, radius=radius, seed=seed)
        self.init_mode = init_mode          # "sphere" (matched) or "random" (baseline)
        self.init_shell = float(init_shell)

    def train(self, data, n_epochs=1000, n_rings=0, lr0=0.1, lr_end=0.01, verbose=True):
        if self.init_mode == "sphere":
            # Same radius-2 shell as SphereSOM3D, so only the lattice graph differs.
            v = self._init_rng.normal(size=(self.n_nodes, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            self.weights = (v * self.init_shell).astype(float)
        else:
            # Faithful "traditional planar SOM" baseline: free random R^3 weights.
            self.weights = self._init_rng.uniform(-1.0, 1.0,
                                                  size=(self.n_nodes, 3)).astype(float)
        points = self.weights
        if verbose:
            print(f"Training Planar SOM ({self.grid_h}x{self.grid_w}, init={self.init_mode}) "
                  f"with {n_epochs} epochs, n_rings={n_rings}, radius={self.radius}, "
                  f"lr {lr0}->{lr_end}...")
        for epoch in range(n_epochs):
            lr = lr0 * (lr_end / lr0) ** (epoch / n_epochs)
            x = data[np.random.randint(0, len(data))]
            bmu_idx = np.argmin(np.linalg.norm(points - x, axis=1))
            bmu_pos = points[bmu_idx]
            for i in self.get_n_ring_neighbors(bmu_idx, n_rings):
                dist = np.linalg.norm(points[i] - bmu_pos)
                if dist <= self.radius:
                    influence = np.exp(-dist ** 2 / (2 * self.radius ** 2))
                    points[i] += lr * influence * (x - points[i])
        self.weights = points


# --------------------------------------------------------------------------- #
# Topology-distortion metric: coefficient of variation of lattice-edge lengths.
# --------------------------------------------------------------------------- #
def lattice_edge_cv(W, nbrs):
    """std/mean of ||w_i - w_j|| over all lattice edges (i, j).

    A topology-preserving map keeps adjacent nodes close and roughly equidistant in
    feature space, so the CV is small. Folding and open-boundary artifacts create a
    mix of very short and very long edges -> larger CV. Scale-invariant, so it is a
    fair comparison even when the two maps differ in overall magnitude.
    """
    lens = []
    for i, ns in enumerate(nbrs):
        for j in ns:
            if j > i:
                lens.append(float(np.linalg.norm(W[i] - W[j])))
    lens = np.asarray(lens, dtype=float)
    if lens.size == 0 or lens.mean() <= 0:
        return float("nan")
    return float(lens.std() / lens.mean())


# --------------------------------------------------------------------------- #
# One run: train + topology metrics + same downstream pipeline as seg_facet_3D.
# --------------------------------------------------------------------------- #
def run_once(label, som, nbrs, data, mesh, face_adjacency,
             n_epochs, n_rings, lr0, lr_end, gt, verbose):
    t0 = time.time()
    som.train(data, n_epochs=n_epochs, n_rings=n_rings, lr0=lr0, lr_end=lr_end,
              verbose=verbose)
    W = np.asarray(som.get_weights())
    bmu = som.predict(data)

    qe = quantization_error(W, data)
    te = topographic_error(W, data, nbrs)
    dead = dead_fraction(bmu, len(W))
    edge_cv = lattice_edge_cv(W, nbrs)

    # Identical downstream pipeline to seg_facet_3D.py.
    raw, raw_count = remap_labels(bmu, mesh=mesh)
    separated = separate_disconnected_components(mesh, face_adjacency, raw)
    separated, _ = remap_labels(separated, mesh=mesh)
    separated = merge_zero_area_regions(mesh, separated, face_adjacency)
    separated, _ = remap_labels(separated, mesh=mesh)
    final_lbl = merge_region_based_on_power(
        mesh, separated.copy(), face_adjacency, target_n_regs=None, verbose=verbose)
    final, final_count = remap_labels(final_lbl)

    ari_raw = float(adjusted_rand_score(gt, raw)) if gt is not None else np.nan
    ari = float(adjusted_rand_score(gt, final)) if gt is not None else np.nan
    return {
        "label": label, "W": W, "bmu": bmu, "nbrs": nbrs,
        "raw": raw, "raw_count": raw_count,
        "final": final, "final_count": final_count,
        "qe": qe, "te": te, "dead": dead, "edge_cv": edge_cv,
        "ari_raw": ari_raw, "ari": ari, "time": time.time() - t0,
    }


def _settings_for(axis_value, args):
    """Map a sweep-axis value to the (radius, n_rings) actually used for a run."""
    if args.sweep == "radius":
        return float(axis_value), int(args.n_rings)
    else:  # sweep == "rings"
        return float(args.radius), int(axis_value)


def evaluate_dataset(obj_file, args, seeds):
    print(f"\n################  {obj_file}  ################")
    mesh = load_obj_with_face_normals(obj_file)
    face_adjacency = build_face_adjacency(mesh)
    smooth_normals(mesh, face_adjacency, n_iter=args.smooth_iter,
                   feature_angle_deg=args.feature_angle_deg)
    data = mesh.cell_data["Normals"]   # unit-length after compute_normals + smooth_normals

    gt, gt_n = (ground_truth_labels(mesh, face_adjacency)
                if args.synthetic else (None, None))
    if gt is not None:
        print(f"ground-truth facets: {gt_n}")

    # results[axis_value][method] -> list of run dicts;  last[axis_value][method] -> last run
    results = {v: {"S-SOM": [], "Planar": []} for v in args.sweep_values}
    last = {v: {} for v in args.sweep_values}

    for v in args.sweep_values:
        radius, n_rings = _settings_for(v, args)
        for s in seeds:
            # --- Spherical SOM: the exact SphereSOM3D used by seg_facet_3D.py ---
            np.random.seed(s)
            sphere_mesh = pv.read(args.sphere_obj)
            ico_nbrs = sphere_adjacency(sphere_mesh)
            ssom = SphereSOM3D(sphere_mesh, radius=radius)
            res = run_once("S-SOM", ssom, ico_nbrs, data, mesh, face_adjacency,
                           args.n_epochs, n_rings, args.lr, args.lr_end, gt,
                           verbose=args.verbose)
            results[v]["S-SOM"].append(res); last[v]["S-SOM"] = res

            # --- Planar SOM: same algorithm + (matched) init, 2D-grid lattice ---
            np.random.seed(s)
            psom = PlanarSOMTopo(args.grid_h, args.grid_w, radius=radius, seed=s,
                                 init_mode=args.planar_init)
            res = run_once("Planar", psom, psom.adjacency(), data, mesh, face_adjacency,
                           args.n_epochs, n_rings, args.lr, args.lr_end, gt,
                           verbose=args.verbose)
            results[v]["Planar"].append(res); last[v]["Planar"] = res

    return results, last, mesh, data, gt_n


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _agg(runs, key):
    v = np.array([r[key] for r in runs], dtype=float)
    return np.nanmean(v), np.nanstd(v)


def print_sweep(obj_file, results, args, gt_n):
    n_seeds = len(results[args.sweep_values[0]]["S-SOM"])
    axis = "radius" if args.sweep == "radius" else "n_rings"
    fixed = (f"n_rings={args.n_rings}" if args.sweep == "radius"
             else f"radius={args.radius}")
    print(f"\n=========  TOPOLOGY ABLATION: {obj_file}  =========")
    print(f"sweep = {axis}   fixed = {fixed}   planar init = {args.planar_init}   "
          f"grid = {args.grid_h}x{args.grid_w} (= {args.grid_h * args.grid_w} nodes)   "
          f"seeds = {n_seeds}")
    if gt_n:
        print(f"ground-truth facets = {gt_n}")
    hdr = (f"{axis:>7} {'method':<7}{'QE':>13}{'TE':>13}{'edgeCV':>13}"
           f"{'dead':>11}{'ARI_raw':>13}{'ARI_fin':>13}{'final':>11}")
    print(hdr)
    print("-" * len(hdr))
    for v in args.sweep_values:
        for m in ("Planar", "S-SOM"):
            runs = results[v][m]
            qe = _agg(runs, "qe"); te = _agg(runs, "te"); cv = _agg(runs, "edge_cv")
            dd = _agg(runs, "dead"); arr = _agg(runs, "ari_raw")
            ar = _agg(runs, "ari"); fc = _agg(runs, "final_count")
            print(f"{v:>7} {m:<7}"
                  f"{qe[0]:>7.4f}+/-{qe[1]:<3.2f}"
                  f"{te[0]:>7.4f}+/-{te[1]:<3.2f}"
                  f"{cv[0]:>7.3f}+/-{cv[1]:<3.2f}"
                  f"{dd[0]:>6.2f}+/-{dd[1]:<3.2f}"
                  f"{arr[0]:>7.3f}+/-{arr[1]:<3.2f}"
                  f"{ar[0]:>7.3f}+/-{ar[1]:<3.2f}"
                  f"{fc[0]:>6.1f}+/-{fc[1]:<3.1f}")
        print()
    print("TE is the headline topology-preservation metric (lower = better); "
          "edgeCV is supporting.")


# --------------------------------------------------------------------------- #
# Sweep plot (matplotlib): the headline ablation figure.
# --------------------------------------------------------------------------- #
def plot_sweep(obj_file, results, args, has_gt):
    xs = args.sweep_values
    xlabel = "neighborhood radius" if args.sweep == "radius" else "n_rings"
    # edge_cv is intentionally NOT a headline panel: a well-trained SOM stretches its
    # lattice to match data density, so its CV rises with neighborhood size even
    # though topology is preserved. TE is the clean topology metric. The middle panel
    # shows the downstream payoff (ARI vs GT) when available, else dead-node fraction.
    mid = ("ari", "ARI final vs GT (higher is better)") if has_gt \
        else ("dead", "Dead-node fraction (lower is better)")
    panels = [("te", "Topographic error (lower is better)"),
              mid,
              ("qe", "Quantization error (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, title) in zip(axes, panels):
        for m, color in (("Planar", "tab:red"), ("S-SOM", "tab:blue")):
            means, stds = [], []
            for v in xs:
                mu, sd = _agg(results[v][m], key)
                means.append(mu); stds.append(sd)
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3,
                        label=m, color=color)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
    axes[0].legend()
    base = os.path.splitext(os.path.basename(obj_file))[0]
    fig.suptitle(f"S-SOM vs Planar SOM topology ({base}, sweep={args.sweep}, "
                 f"planar init={args.planar_init})")
    fig.tight_layout()
    out = f"ablation_topology_{base}_{args.sweep}.png"
    fig.savefig(out, dpi=150)
    print(f"saved sweep plot -> {out}")
    plt.show()


# --------------------------------------------------------------------------- #
# 3D visualization (one sweep value, last seed): lattices folded onto the
# normal sphere + the resulting segmentations.
# --------------------------------------------------------------------------- #
def _plot_lattice_on_normals(p, W, nbrs, bmu, data_normals, raw, raw_count, title):
    # faint unit sphere = the closed manifold the face normals live on
    p.add_mesh(pv.Sphere(radius=1.0), color="white", opacity=0.12)
    cloud = pv.PolyData(np.asarray(data_normals, dtype=float))
    cloud.point_data["cluster"] = raw
    p.add_mesh(cloud, scalars="cluster", cmap=get_color_map(raw_count),
               render_points_as_spheres=True, point_size=4, opacity=0.5,
               show_scalar_bar=False)
    lines = []
    for i, ns in enumerate(nbrs):
        for j in ns:
            if j > i:
                lines.extend([2, i, j])
    poly = pv.PolyData(W, lines=np.array(lines, dtype=np.int64))
    p.add_mesh(poly, color="dimgrey", line_width=1, opacity=0.85)
    winners = np.unique(bmu)
    if len(winners):
        p.add_mesh(pv.PolyData(W[winners]).glyph(scale=False, geom=pv.Sphere(radius=0.03)),
                   color="red")
    p.add_text(title, font_size=8)


def visualize_topology(mesh, data_normals, last, value_to_show, args):
    methods = [("Planar", "planar"), ("S-SOM", "sphere")]
    stages = [("raw", "raw_count", "raw (BMU)"),
              ("final", "final_count", "power-merged")]
    sel = last[value_to_show]
    for m, prefix in methods:
        for key, _, _ in stages:
            mesh.cell_data[f"{prefix}_{key}"] = sel[m][key]

    axis = "radius" if args.sweep == "radius" else "n_rings"
    p = pv.Plotter(shape=(2, 3),
                   title=f"Topology ablation ({axis}={value_to_show}): "
                         f"Planar (top) vs Spherical (bottom)")
    for row, (m, prefix) in enumerate(methods):
        p.subplot(row, 0)
        r = sel[m]
        _plot_lattice_on_normals(
            p, r["W"], r["nbrs"], r["bmu"], data_normals, r["raw"], r["raw_count"],
            f"{m} lattice on normal sphere\nTE={r['te']:.3f}  edgeCV={r['edge_cv']:.2f}")
        for col, (key, cnt, title) in enumerate(stages, start=1):
            p.subplot(row, col)
            _plot_labels(p, mesh.copy(), f"{prefix}_{key}", sel[m][cnt], f"{m} {title}")
    p.link_views()
    p.show()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="S-SOM topology ablation: spherical lattice vs planar grid.")
    ap.add_argument("--obj_file", type=str, default=None,
                    help="Single mesh; if omitted, runs the default dataset list.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Compute ARI against ground-truth facets (flat-faced solids).")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sweep", choices=["radius", "rings"], default="radius",
                    help="Which neighborhood knob to sweep. 'radius' is the binding "
                         "knob in this code; 'rings' is only meaningful at large radius.")
    ap.add_argument("--sweep_values", type=float, nargs="+", default=None,
                    help="Values to sweep. Default: radius -> 0.1 0.3 0.6 1.0; "
                         "rings -> 0 1 2 3.")
    ap.add_argument("--planar_init", choices=["sphere", "random"], default="sphere",
                    help="'sphere' matches the S-SOM init to isolate topology; "
                         "'random' is the faithful traditional planar SOM baseline.")
    ap.add_argument("--sphere_obj", type=str, default="regular_sphere.obj")
    ap.add_argument("--grid_h", type=int, default=9)
    ap.add_argument("--grid_w", type=int, default=18)   # 9*18 = 162 == icosphere nodes
    # --- fixed values for the non-swept knob ---
    ap.add_argument("--n_rings", type=int, default=3,
                    help="Fixed n_rings when sweeping radius.")
    ap.add_argument("--radius", type=float, default=0.6,
                    help="Fixed radius when sweeping rings.")
    # --- defaults match seg_facet_3D.py ---
    ap.add_argument("--n_epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--lr_end", type=float, default=0.01)
    ap.add_argument("--smooth_iter", type=int, default=5)
    ap.add_argument("--feature_angle_deg", type=float, default=30.0)
    ap.add_argument("--plot", action="store_true",
                    help="Save/show the TE/edgeCV/QE-vs-sweep figure.")
    ap.add_argument("--show", action="store_true",
                    help="Show the 3D lattice/segmentation comparison (largest value).")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-epoch training and per-merge progress.")
    args = ap.parse_args()

    if args.sweep_values is None:
        args.sweep_values = [0.1, 0.3, 0.6, 1.0] if args.sweep == "radius" else [0, 1, 2, 3]
    if args.sweep == "rings":
        args.sweep_values = [int(v) for v in args.sweep_values]

    if args.obj_file:
        datasets = [(args.obj_file, args.synthetic)]
    else:
        datasets = [
            ("./datasets/Synthetic/icosahedron.obj", True),
            # ("./datasets/Synthetic/octahedron.obj", True),
            # ("./datasets/3DPuzzle/brick_part01.obj", False),
        ]

    seeds = list(range(args.seeds))
    for obj_file, synth in datasets:
        args.synthetic = synth
        results, last, mesh, data, gt_n = evaluate_dataset(obj_file, args, seeds)
        print_sweep(obj_file, results, args, gt_n)
        if args.plot:
            plot_sweep(obj_file, results, args, has_gt=gt_n is not None)
        if args.show:
            visualize_topology(mesh, data, last, max(args.sweep_values), args)


if __name__ == "__main__":
    main()
