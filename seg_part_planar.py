"""Planar / spherical SOM ablation for the part pipeline in seg_part.py.

The default 'reference' initialization reproduces SSOM2D.SphereSOM exactly.
Use --init sample to give BOTH maps the same data-sampled initial weights.
Unlike the facet trainer, this trainer uses fixed lattice positions for its
radius gate and a constant learning rate. Feature weights may be 1-D or 2-D.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv

import seg_part as part
from PlanarSOM import PlanarSOM3D, lattice_edges, matched_grid_shape
from SSOM2D import SphereSOM
from helper import (build_face_adjacency, compute_face_areas, merge_small_faces,
                    remap_labels, separate_disconnected_components,
                    merge_region_based_on_power)


ROOT = Path(__file__).resolve().parent
RIM_COLOR = "#e03131"


def mesh_signature(mesh):
    digest = hashlib.sha256()
    digest.update(np.asarray(mesh.points).tobytes())
    digest.update(np.asarray(mesh.faces).tobytes())
    return digest.hexdigest()


def prepare_features(args):
    """Compute descriptors ONCE, using the original part preprocessing."""
    mesh = part.load_obj_with_face_normals(str(args.input))
    if mesh.n_cells == 0 or not mesh.is_all_triangles:
        raise ValueError("Input must be a nonempty triangular mesh")
    adjacency = build_face_adjacency(mesh)
    signature = mesh_signature(mesh)
    if args.features_file is not None:
        with np.load(args.features_file, allow_pickle=False) as cached:
            if str(cached["mesh_sha256"].item()) != signature:
                raise ValueError("Feature cache belongs to a different mesh or face ordering")
            if (float(cached["sdf_cap_pct"]) != args.sdf_cap_pct
                    or int(cached["sdf_smooth_iter"]) != args.sdf_smooth_iter):
                raise ValueError("Feature cache SDF settings differ from the requested settings")
            sdf = cached["sdf"].copy().reshape(-1, 1)
            curvature = cached["curvature"].copy().reshape(-1, 1)
    else:
        curvature = part.comp_cur(mesh)
        curvature = (curvature / np.max(curvature)).reshape(-1, 1)
        ms = part.pymeshlab.MeshSet()
        ms.load_new_mesh(str(args.input))
        # This is the same GPU SDF filter and settings as seg_part.py.
        ms.compute_scalar_by_shape_diameter_function_per_vertex_gpu(
            coneangle=120, onprimitive="On Faces", removeoutliers=True, numberrays=180)
        sdf = ms.current_mesh().face_scalar_array()
        sdf = part.smooth_face_scalar(mesh, adjacency, sdf, n_iter=args.sdf_smooth_iter)
        sdf = part.clamp_percentiles(sdf, args.sdf_cap_pct).reshape(-1, 1)
    for name, values in (("SDF", sdf), ("curvature", curvature)):
        if len(values) != mesh.n_cells or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid {name} descriptors")
    if args.fea == "sdf_cur":
        data = np.concatenate((sdf, curvature), axis=1)
        maximum = float(np.linalg.norm(data, axis=1).max())
        if maximum <= 0:
            raise ValueError("Descriptors have zero maximum norm")
        data = data / maximum
    else:
        data = sdf.copy() if args.fea == "sdf" else curvature.copy()
    mesh.cell_data["features"] = data
    mesh.cell_data["SDF"] = sdf.ravel()
    mesh.cell_data["curvature"] = curvature.ravel()
    return mesh, adjacency, data, sdf, curvature


def make_maps(spherical_mesh, args):
    """Keep the real SSOM2D trainer for BOTH topologies.

    Match the median edge length so --radius has comparable local units; do
    not normalize planar positions radially, which would destroy its grid.
    """
    sphere = SphereSOM(spherical_mesh.copy(deep=True))
    sphere.edges = lattice_edges(spherical_mesh)
    rows, cols = ((args.rows, args.cols) if args.rows is not None
                  else matched_grid_shape(sphere.n_nodes))
    if rows * cols != sphere.n_nodes:
        raise ValueError("Planar and spherical maps must have equal neuron counts")
    grid = PlanarSOM3D(rows, cols, grid=args.grid)
    planar = SphereSOM(grid.mesh.copy(deep=True))
    spherical_lengths = np.linalg.norm(
        sphere.positions[sphere.edges[:, 0]] - sphere.positions[sphere.edges[:, 1]], axis=1)
    planar_lengths = np.linalg.norm(
        grid.positions[grid.edges[:, 0]] - grid.positions[grid.edges[:, 1]], axis=1)
    scale = float(np.median(spherical_lengths) / np.median(planar_lengths))
    planar.positions = grid.positions.copy() * scale
    planar.mesh.points = planar.positions.copy()
    planar.edges = grid.edges.copy()
    planar.boundary_edges = grid.boundary_edges.copy()
    planar.boundary_mask = grid.boundary_mask.copy()
    planar.flat_positions = grid.positions.copy()
    planar.rows, planar.cols = rows, cols
    # For quads, VTK point_neighbors also includes diagonal vertices. Use the
    # grid's explicit edges so --grid rect really has four interior neighbors.
    planar.get_n_ring_neighbors = grid.get_n_ring_neighbors
    sphere.boundary_mask = np.zeros(sphere.n_nodes, dtype=bool)
    sphere.boundary_edges = np.empty((0, 2), dtype=int)
    for som, name in ((planar, "planar"), (sphere, "spherical")):
        som.topology = name
        som.adjacency = np.zeros((som.n_nodes, som.n_nodes), dtype=bool)
        som.adjacency[som.edges[:, 0], som.edges[:, 1]] = True
        som.adjacency[som.edges[:, 1], som.edges[:, 0]] = True
        som.median_edge_length = float(np.median(np.linalg.norm(
            som.positions[som.edges[:, 0]] - som.positions[som.edges[:, 1]], axis=1)))
    return planar, sphere


def train_map(som, data, args, seed, shared_initial=None):
    """Observe the actual trainer without changing its update arithmetic/order."""
    initialize = som._initialize_nodes
    neighbors = som.get_n_ring_neighbors
    counts = []

    def recorded_initialize(n_dim):
        if shared_initial is None:
            initialize(n_dim)
        else:
            som.weights = shared_initial.copy()
        som.initial_weights = som.weights.copy()

    def recorded_neighbors(bmu, rings):
        nodes = neighbors(bmu, rings) if rings >= 0 else np.arange(som.n_nodes)
        counts.append(sum(node != bmu and np.linalg.norm(
            som.positions[node] - som.positions[bmu]) <= args.radius for node in nodes))
        return nodes

    som._initialize_nodes = recorded_initialize
    som.get_n_ring_neighbors = recorded_neighbors
    som.sample_indices = np.random.RandomState(seed).randint(len(data), size=args.epochs)
    state = np.random.get_state()
    start = time.perf_counter()
    try:
        np.random.seed(seed)
        som.train(data, n_epochs=args.epochs, n_rings=args.n_rings,
                  lr=args.lr, radius=args.radius)
    finally:
        np.random.set_state(state)
        som._initialize_nodes = initialize
        som.get_n_ring_neighbors = neighbors
    # The original trainer bypasses this observer for n_rings=-1; report None
    # rather than inventing neighbor counts for that mode.
    som.training_seconds = time.perf_counter() - start
    som.training_stats = {
        "mean_updated_non_bmu_neurons": float(np.mean(counts)) if counts else None,
        "fraction_updates_with_non_bmu": float(np.mean(np.asarray(counts) > 0)) if counts else None,
    }
    return som


def evaluate_map(som, data, mesh):
    bmus = som.predict(data)
    errors = np.empty(len(data))
    topo_errors = np.empty(len(data), dtype=bool)
    ties = np.empty(len(data), dtype=bool)
    for start in range(0, len(data), 2048):
        chunk = data[start:start + 2048]
        distances = np.linalg.norm(chunk[:, None] - som.weights[None, :], axis=2)
        first_two = np.argsort(distances, axis=1, kind="stable")[:, :2]
        first = bmus[start:start + len(chunk)]
        second = np.where(first_two[:, 0] == first, first_two[:, 1], first_two[:, 0])
        index = np.arange(len(chunk))
        errors[start:start + len(chunk)] = distances[index, first]
        topo_errors[start:start + len(chunk)] = ~som.adjacency[first, second]
        ties[start:start + len(chunk)] = distances[index, first] == distances[index, second]
    rim_faces = som.boundary_mask[bmus]
    areas = np.asarray(compute_face_areas(mesh))
    hits = np.bincount(bmus, minlength=som.n_nodes)
    node_qe = np.bincount(bmus, weights=errors, minlength=som.n_nodes) / np.maximum(hits, 1)
    node_qe[hits == 0] = np.nan

    def subset_mean(values, mask):
        return float(np.mean(values[mask])) if np.any(mask) else None

    metrics = {
        "n_neurons": som.n_nodes, "active_neurons": int(np.count_nonzero(hits)),
        "quantization_error": float(errors.mean()),
        "topographic_error": float(topo_errors.mean()),
        "nearest_prototype_tie_fraction": float(ties.mean()),
        "boundary_neurons": int(som.boundary_mask.sum()),
        "boundary_face_fraction": float(rim_faces.mean()),
        "boundary_area_fraction": float(areas[rim_faces].sum() / areas.sum()) if areas.sum() > 0 else None,
        "boundary_quantization_error": subset_mean(errors, rim_faces),
        "interior_quantization_error": subset_mean(errors, ~rim_faces),
        "boundary_topographic_error": subset_mean(topo_errors, rim_faces),
        "interior_topographic_error": subset_mean(topo_errors, ~rim_faces),
        "median_lattice_edge_length": som.median_edge_length,
        "training_seconds": som.training_seconds, **som.training_stats,
    }
    diagnostics = dict(bmus=bmus, quantization_errors=errors,
                       topographic_errors=topo_errors, rim_faces=rim_faces,
                       hits=hits, node_qe=node_qe)
    return metrics, diagnostics


def cluster_mesh(mesh, adjacency, som):
    """Exactly the original seg_part.py raw and connected-component steps."""
    corrected = merge_small_faces(mesh, som.predict(mesh.cell_data["features"]),
                                  adjacency, area_ratio=0.03)
    raw, count = remap_labels(corrected)
    separated = separate_disconnected_components(mesh, adjacency, raw)
    separated, _ = remap_labels(separated, mesh=mesh)
    return raw, int(count), separated


def update_merge(result, mesh, adjacency, threshold, alpha, max_merges):
    start = time.perf_counter()
    merged = merge_region_based_on_power(
        mesh, result["separated"].copy(), adjacency, alpha=alpha, beta=1 - alpha,
        power_thr=threshold, max_merges=max_merges, target_n_regs=None,
        feature_name="features", verbose=False)
    result["merged"], count = remap_labels(merged)
    merges = len(np.unique(result["separated"])) - int(count)
    result["metrics"].update(
        merged_regions=int(count), merges=merges, merge_seconds=time.perf_counter() - start,
        merge_budget_reached=(merges == max_merges), power_threshold=float(threshold),
        alpha=float(alpha), beta=float(1 - alpha))
    result["metrics"]["postprocessing_seconds"] = (
        result["metrics"]["clustering_seconds"] + result["metrics"]["merge_seconds"])


def segment(som, data, mesh, adjacency, args, seed):
    metrics, diagnostics = evaluate_map(som, data, mesh)
    start = time.perf_counter()
    raw, count, separated = cluster_mesh(mesh, adjacency, som)
    metrics.update(seed=int(seed), initialization=args.init, raw_clusters=count,
                   separated_regions=int(len(np.unique(separated))),
                   clustering_seconds=time.perf_counter() - start)
    result = dict(name="Planar SOM" if som.topology == "planar" else "S-SOM",
                  som=som, metrics=metrics, diagnostics=diagnostics,
                  raw=raw, separated=separated)
    update_merge(result, mesh, adjacency, args.power_thr, args.alpha, args.max_merges)
    return result


def feature_figure(data, result, feature_name):
    """Feature weights and topological edges; red marks the open planar rim."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    som = result["som"]
    xy = data if data.shape[1] == 2 else np.column_stack([data[:, 0], np.zeros(len(data))])
    weights = (som.weights if data.shape[1] == 2
               else np.column_stack([som.weights[:, 0], np.zeros(som.n_nodes)]))
    count = result["metrics"]["raw_clusters"]
    cloud = ax.scatter(xy[:, 0], xy[:, 1], c=result["raw"], s=4, alpha=0.5,
                       cmap=ListedColormap(part.get_color_map(count)))
    ax.add_collection(LineCollection(weights[som.edges], colors="#888888", linewidths=0.6, alpha=0.5))
    active = result["diagnostics"]["hits"] > 0
    ax.scatter(weights[~active, 0], weights[~active, 1], c="black", s=9, label="Inactive neuron")
    ax.scatter(weights[active, 0], weights[active, 1], facecolors="none",
               edgecolors="#333333", s=25, label="Active neuron")
    if som.boundary_mask.any():
        ax.add_collection(LineCollection(weights[som.boundary_edges], colors=RIM_COLOR, linewidths=1.5))
        ax.scatter(weights[som.boundary_mask, 0], weights[som.boundary_mask, 1],
                   c=RIM_COLOR, s=12, label="Open rim neuron")
    ax.set_xlabel("SDF" if feature_name != "cur" else "Curvature")
    ax.set_ylabel("Curvature" if data.shape[1] == 2 else "")
    metrics = result["metrics"]
    ax.set_title(f"{result['name']}: feature weights + fixed graph\n"
                 f"QE={metrics['quantization_error']:.4f}; TE={metrics['topographic_error']:.1%}; "
                 f"rim faces={metrics['boundary_face_fraction']:.1%}", fontsize=9)
    ax.legend(fontsize=6, loc="best")
    fig.colorbar(cloud, ax=ax, label="Raw cluster", shrink=0.8)
    return fig


def plot_labels(plotter, mesh, labels, title, actor):
    view = mesh.copy(deep=True)
    view.cell_data[actor] = labels
    count = len(np.unique(labels))
    if actor in plotter.scalar_bars:
        plotter.remove_scalar_bar(actor)
    plotter.add_mesh(view, scalars=actor, cmap=part.get_color_map(count),
                     show_edges=True, edge_opacity=0.2, name=actor,
                     scalar_bar_args=dict(title=actor, fmt="%.0f", n_labels=min(count, 10)))
    plotter.add_text(title, font_size=10, color="#202020", name=actor + "_title")


def render_dashboard(results, mesh, data, adjacency, args, destination, interactive):
    import matplotlib.pyplot as plt

    plotter = pv.Plotter(shape=(4, 3), off_screen=not interactive, window_size=(2100, 1800))
    figures, object_views = [], []
    for index, result in enumerate(results):
        row = index * 2
        for col, field, title in ((0, "SDF", "SDF"), (1, "curvature", "Curvature")):
            plotter.subplot(row, col)
            plotter.set_background("white")
            # Each mapper needs independent active scalars: reusing one mesh
            # here would make its SDF panel display the last selected curvature.
            view = mesh.copy(deep=True)
            plotter.add_mesh(view, scalars=field, cmap="jet", show_edges=True, edge_opacity=0.2,
                             scalar_bar_args=dict(title=f"{result['name']} {title}"))
            plotter.add_text(f"{result['name']}: {title}", font_size=10, color="#202020")
            plotter.view_isometric()
        plotter.subplot(row, 2)
        plotter.set_background("white")
        fig = feature_figure(data, result, args.fea)
        figures.append(fig)
        plotter.add_chart(pv.ChartMPL(fig))
        fig.savefig(destination.parent / f"{result['som'].topology}_feature_map.png", dpi=180)
        for col, stage in enumerate(("raw", "separated", "merged")):
            plotter.subplot(row + 1, col)
            plotter.set_background("white")
            count = len(np.unique(result[stage]))
            plot_labels(plotter, mesh, result[stage],
                        f"{result['name']}: {stage} ({count} regions)", f"{result['som'].topology} {stage}")
            plotter.view_isometric()
        object_views.extend([row * 3, row * 3 + 1, (row + 1) * 3, (row + 1) * 3 + 1, (row + 1) * 3 + 2])
    plotter.link_views(object_views)
    if interactive:
        control = dict(threshold=args.power_thr, alpha=args.alpha)

        def redraw():
            for index, result in enumerate(results):
                update_merge(result, mesh, adjacency, control["threshold"], control["alpha"], args.max_merges)
                plotter.subplot(2 * index + 1, 2)
                plot_labels(plotter, mesh, result["merged"],
                            f"{result['name']}: merged ({result['metrics']['merged_regions']} regions)\n"
                            f"threshold={control['threshold']:.3f}; alpha={control['alpha']:.3f}",
                            f"{result['som'].topology} merged")
            plotter.render()

        def on_threshold(value):
            control["threshold"] = float(value)
            redraw()

        def on_alpha(value):
            control["alpha"] = float(value)
            redraw()

        plotter.subplot(1, 2)
        plotter.add_slider_widget(on_threshold, rng=[0, 1], value=args.power_thr,
                                  title="Shared power threshold", pointa=(0.1, 0.05), pointb=(0.9, 0.05),
                                  interaction_event="end", fmt="%.3f")
        plotter.subplot(1, 2)
        plotter.add_slider_widget(on_alpha, rng=[0, 1], value=args.alpha,
                                  title="Shared alpha (beta = 1 - alpha)", pointa=(0.1, 0.15), pointb=(0.9, 0.15),
                                  interaction_event="end", fmt="%.3f")
        plotter.show(auto_close=False)
        plotter.screenshot(str(destination))
    else:
        plotter.show(screenshot=str(destination), auto_close=False)
    plotter.close()
    for fig in figures:
        plt.close(fig)


def render_planar_diagnostics(result, directory):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm

    som = result["som"]
    xy = som.flat_positions[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, field, title in ((axes[0], "hits", "BMU hits + 1 (log scale)"),
                             (axes[1], "node_qe", "Mean QE (inactive neurons: grey)")):
        values = result["diagnostics"][field]
        ax.add_collection(LineCollection(xy[som.edges], colors="#aaaaaa", linewidths=0.6))
        if field == "hits":
            scatter = ax.scatter(xy[:, 0], xy[:, 1], c=values + 1, s=50,
                                 norm=LogNorm(vmin=1, vmax=max(int(values.max()) + 1, 2)))
        else:
            ax.scatter(xy[:, 0], xy[:, 1], c="#d8dce2", s=50)
            valid = np.isfinite(values)
            scatter = ax.scatter(xy[valid, 0], xy[valid, 1], c=values[valid], s=50, cmap="magma")
        ax.add_collection(LineCollection(xy[som.boundary_edges], colors=RIM_COLOR, linewidths=2))
        ax.scatter(xy[som.boundary_mask, 0], xy[som.boundary_mask, 1],
                   facecolors="none", edgecolors=RIM_COLOR, s=90)
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(scatter, ax=ax, shrink=0.8)
    fig.suptitle("Fixed planar lattice; red = open rim (not a segmentation error)")
    fig.savefig(directory / "planar_diagnostics.png", dpi=180)
    fig.savefig(directory / "planar_diagnostics.svg")
    plt.close(fig)


def save_results(results, mesh, data, sdf, curvature, args, directory):
    np.savez_compressed(directory / "features.npz", features=data, sdf=sdf, curvature=curvature,
                        mesh_sha256=mesh_signature(mesh), sdf_cap_pct=args.sdf_cap_pct,
                        sdf_smooth_iter=args.sdf_smooth_iter)
    records = []
    for result in results:
        som = result["som"]
        record = dict(topology=som.topology, **result["metrics"])
        records.append(record)
        output_mesh = mesh.copy(deep=True)
        for stage in ("raw", "separated", "merged"):
            np.savetxt(directory / f"{som.topology}_{stage}.seg", result[stage], fmt="%d")
            output_mesh.cell_data[stage] = result[stage]
        output_mesh.cell_data["bmu"] = result["diagnostics"]["bmus"]
        output_mesh.cell_data["rim_assignment"] = result["diagnostics"]["rim_faces"].astype(np.uint8)
        output_mesh.save(directory / f"{som.topology}.vtp")
        np.savez_compressed(directory / f"{som.topology}_som.npz", initial_weights=som.initial_weights,
                            weights=som.weights, positions=som.positions, edges=som.edges,
                            boundary_mask=som.boundary_mask, training_indices=som.sample_indices,
                            **result["diagnostics"])
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration["seed"] = int(results[0]["metrics"]["seed"])
    notes = [
        "Both topologies call the actual SSOM2D.SphereSOM trainer with identical sampled faces and learning settings.",
        "Both use identical initial feature weights; reference retains the original uniform float32 value 2.",
        "The radius gate is measured on FIXED 3-D lattice positions; feature weights have 1 or 2 dimensions.",
        "Planar median edge length matches the normalized spherical median edge length.",
        "seg_part.py currently uses 1000 updates and trainer-default n_rings=2; its CLI n_rings/init_neu_size are not forwarded.",
        "alpha=0.75 matches the original saved segmentation; its interactive view initially uses alpha=1/3.",
        "QE/TE are SOM diagnostics, not segmentation accuracy; tied prototypes can make TE ambiguous.",
        "Red marks planar rim neurons; rim occupancy alone is not proof of an error.",
        "Output labels are saved here, never beside the input mesh.",
    ]
    payload = dict(configuration=configuration, mesh_sha256=mesh_signature(mesh),
                   descriptor_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                   n_faces=len(data), grid_shape=[results[0]["som"].rows, results[0]["som"].cols],
                   effective_merge={key: results[0]["metrics"][key]
                                    for key in ("power_threshold", "alpha", "beta")},
                   notes=notes, results=records)
    (directory / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def run(args):
    if args.headless or args.no_render:
        import matplotlib
        matplotlib.use("Agg")
    start = time.perf_counter()
    mesh, adjacency, data, sdf, curvature = prepare_features(args)
    preprocessing_seconds = time.perf_counter() - start
    sphere_mesh = pv.read(str(args.sphere_mesh))
    seeds = args.seeds if args.seeds is not None else [args.seed]
    records = []
    experiment = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()
                  if key not in ("headless", "no_render", "output_dir", "features_file", "seed", "seeds")}
    experiment.update(pipeline="seg_part_planar_v1", mesh_sha256=mesh_signature(mesh),
                      descriptor_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                      sphere_sha256=mesh_signature(sphere_mesh))
    if args.init == "reference":
        print("NOTE: Reference initialization gives all neurons the same weight 2. "
              "Use --init sample for a shared, nondegenerate initialization control.")
    if args.n_rings == 0:
        print("NOTE: n_rings=0 disables topology during learning; shared initialization should give identical maps.")
    for index, seed in enumerate(seeds):
        signature = hashlib.sha256(json.dumps(dict(experiment, seed=int(seed)), sort_keys=True).encode()).hexdigest()[:8]
        directory = args.output_dir / f"{args.input.stem}_{args.fea}_{args.grid}_{args.init}_seed{seed}_{signature}"
        directory.mkdir(parents=True, exist_ok=True)
        maps = make_maps(sphere_mesh, args)
        shared_initial = (data[np.random.default_rng(seed).integers(len(data), size=maps[0].n_nodes)].astype(np.float32)
                          if args.init == "sample" else None)
        results = []
        for som in maps:
            train_map(som, data, args, seed, shared_initial)
            result = segment(som, data, mesh, adjacency, args, seed)
            result["metrics"]["shared_preprocessing_seconds"] = preprocessing_seconds
            results.append(result)
        if not args.no_render:
            render_dashboard(results, mesh, data, adjacency, args, directory / "comparison.png",
                             interactive=not args.headless and index == 0)
            render_planar_diagnostics(results[0], directory)
        current = save_results(results, mesh, data, sdf, curvature, args, directory)
        records.extend(current)
        for record in current:
            neighbors = record["fraction_updates_with_non_bmu"]
            support = f"{neighbors:.1%}" if neighbors is not None else "not measured for all-node mode"
            print(f"{record['topology']} seed={seed}: QE={record['quantization_error']:.5f}; "
                  f"TE={record['topographic_error']:.2%}; "
                  f"regions={record['raw_clusters']} -> {record['separated_regions']} -> {record['merged_regions']}; "
                  f"updates with neighbors={support}")
            if record["merge_budget_reached"]:
                print("NOTE: Merge budget reached; raise --max-merges before interpreting final region counts.")
        print(f"Saved {directory}")
    if len(seeds) > 1:
        summary = {}
        for topology in ("planar", "spherical"):
            paired = [r for r in records if r["topology"] == topology]
            summary[topology] = {key: dict(mean=float(np.mean([r[key] for r in paired])),
                                          std=float(np.std([r[key] for r in paired])))
                                 for key in ("quantization_error", "topographic_error", "active_neurons",
                                             "raw_clusters", "separated_regions", "merged_regions")}
        signature = hashlib.sha256(json.dumps(dict(experiment, seeds=seeds), sort_keys=True).encode()).hexdigest()[:8]
        path = args.output_dir / f"{args.input.stem}_{args.fea}_{args.init}_{signature}_summary.json"
        path.write_text(json.dumps(dict(seeds=seeds, summary=summary, runs=records), indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {path}")
    return records


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "datasets/Princeton/1.obj")
    parser.add_argument("--sphere-mesh", type=Path, default=ROOT / "regular_sphere.obj")
    parser.add_argument("--fea", choices=["sdf", "cur", "sdf_cur"], default="sdf_cur")
    parser.add_argument("--init", choices=["reference", "sample"], default="reference",
                        help="Reference: original identical weights=2; sample: identical data-sampled weights for both maps")
    parser.add_argument("--epochs", type=int, default=1000, help="Sampled updates, not whole-dataset passes")
    parser.add_argument("--n_rings", "--n-rings", type=int, default=2,
                        help="Effective seg_part.py trainer default is 2; 0 disables topology; -1 searches all nodes")
    parser.add_argument("--radius", type=float, default=1.0, help="Gate and Gaussian width in fixed lattice coordinates")
    parser.add_argument("--lr", type=float, default=0.01, help="Constant learning rate, as in SSOM2D.py")
    parser.add_argument("--power_thr", "--power-thr", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="Smoothness weight; beta=1-alpha. Default matches original saved .seg; original interactive view uses 1/3")
    parser.add_argument("--max-merges", type=int, default=10000)
    parser.add_argument("--sdf_cap_pct", "--sdf-cap-pct", type=float, default=1.0)
    parser.add_argument("--sdf_smooth_iter", "--sdf-smooth-iter", type=int, default=5)
    parser.add_argument("--features-file", type=Path, help="Reuse an exported features.npz for this mesh and SDF settings; skips GPU SDF")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--grid", choices=["hex", "rect"], default="hex")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--headless", action="store_true", help="Render figures offscreen and exit")
    parser.add_argument("--no-render", action="store_true", help="Compute and export labels/metrics without figures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/topology_ablation/part")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if not args.input.is_file() or not args.sphere_mesh.is_file():
        parser.error("Input and spherical mesh must exist")
    if (args.rows is None) != (args.cols is None):
        parser.error("Specify both --rows and --cols, or neither")
    if (args.epochs <= 0 or args.n_rings < -1 or args.max_merges < 0 or args.sdf_smooth_iter < 0
            or not 0 <= args.sdf_cap_pct < 50 or not 0 <= args.power_thr <= 1
            or not 0 <= args.alpha <= 1 or args.radius <= 0 or args.lr <= 0
            or not np.all(np.isfinite([args.radius, args.lr]))):
        parser.error("Invalid training, preprocessing or merge settings")
    if any(seed < 0 or seed > 2**32 - 1 for seed in (args.seeds or [args.seed])):
        parser.error("Seeds must be in [0, 2**32 - 1]")
    run(args)


if __name__ == "__main__":
    main()
