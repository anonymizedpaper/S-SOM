"""Facet segmentation and border-effect diagnostics for an open planar SOM.

Examples:
    python seg_facet_planar.py --input datasets/3DPuzzle/brick_part01.obj
    python seg_facet_planar.py --demo-sphere --headless
    python seg_facet_planar.py --kernel paper --n_rings 2 --radius 0.1

The S-SOM branch calls the same SphereSOM3D trainer and facet processing
functions as seg_facet.py. Each topology has a six-panel (2 x 3) view.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import seg_facet as facet

from PlanarSOM import PlanarSOM3D, TopologySOM3D, matched_grid_shape
from helper import build_face_adjacency, compute_face_areas, smooth_normals


ROOT = Path(__file__).resolve().parent
RIM_COLOR = "#e03131"
TEAR_COLOR = "#c026d3"


def shared_mesh_edges(mesh):
    """Return interior mesh edges and the two incident faces, without duplicates."""
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                    triangles[:, [2, 0]]]), axis=1)
    face_ids = np.tile(np.arange(len(triangles)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, face_ids = edges[order], face_ids[order]
    same = np.flatnonzero(np.all(edges[1:] == edges[:-1], axis=1))
    return edges[same], np.column_stack([face_ids[same], face_ids[same + 1]])


def evaluate_map(som, data, mesh, mesh_edges, face_pairs, seam_angle=5.0,
                 tear_hops=2, sigma=3.0, n_rings=-1):
    """Measure the true BMUs, before any relabeling or small-face correction.

    A tear is an edge whose incident normals differ by <= seam_angle degrees
    but whose BMUs are > tear_hops graph steps apart. This is a map-continuity
    diagnostic, not a segmentation-accuracy score.
    """
    # Keep the real predictor, including SphereSOM3D's original arithmetic.
    bmus = som.predict(data)
    errors = np.empty(len(data))
    topo_errors = np.empty(len(data), dtype=bool)
    for start in range(0, len(data), 2048):
        chunk = data[start:start + 2048]
        distances = np.sum((chunk[:, None] - som.weights[None, :])**2, axis=2)
        # Stable ordering specifies how exact prototype ties are treated.
        first_two = np.argsort(distances, axis=1, kind="stable")[:, :2]
        stop = start + len(chunk)
        first = bmus[start:stop]
        second = np.where(first_two[:, 0] == first, first_two[:, 1], first_two[:, 0])
        errors[start:stop] = np.sqrt(distances[np.arange(len(chunk)), first])
        topo_errors[start:stop] = ~som.adjacency[first, second]
    rim_faces = som.boundary_mask[bmus]
    areas = np.asarray(compute_face_areas(mesh))
    normal_cos = np.einsum("ij,ij->i", data[face_pairs[:, 0]], data[face_pairs[:, 1]])
    smooth_edges = normal_cos >= np.cos(np.deg2rad(seam_angle))
    hops = som.graph_distances[bmus[face_pairs[:, 0]], bmus[face_pairs[:, 1]]]
    tears = smooth_edges & (hops > tear_hops)
    hits = np.bincount(bmus, minlength=som.n_nodes)
    counts = np.maximum(hits, 1)
    node_qe = np.bincount(bmus, weights=errors, minlength=som.n_nodes) / counts
    node_te = np.bincount(bmus, weights=topo_errors.astype(float), minlength=som.n_nodes) / counts
    node_qe[hits == 0] = np.nan
    node_te[hits == 0] = np.nan
    support = som.neighborhood_support(sigma, n_rings)
    rim = som.boundary_mask
    interior = ~rim

    def mean_or_none(values):
        return float(np.mean(values)) if len(values) else None

    metrics = {
        "n_neurons": som.n_nodes,
        "active_neurons": int(np.count_nonzero(hits)),
        "quantization_error": float(errors.mean()),
        "topographic_error": float(topo_errors.mean()),
        "boundary_neurons": int(rim.sum()),
        "boundary_neuron_fraction": float(rim.mean()),
        "boundary_face_fraction": float(rim_faces.mean()),
        "boundary_area_fraction": float(areas[rim_faces].sum() / areas.sum()) if areas.sum() > 0 else None,
        "boundary_quantization_error": mean_or_none(errors[rim_faces]),
        "interior_quantization_error": mean_or_none(errors[~rim_faces]),
        "boundary_topographic_error": mean_or_none(topo_errors[rim_faces]),
        "interior_topographic_error": mean_or_none(topo_errors[~rim_faces]),
        "boundary_mean_degree": mean_or_none(som.adjacency.sum(axis=1)[rim]),
        "interior_mean_degree": mean_or_none(som.adjacency.sum(axis=1)[interior]),
        "boundary_lattice_support": mean_or_none(support[rim]),
        "interior_lattice_support": mean_or_none(support[interior]),
        "smooth_mesh_edges": int(smooth_edges.sum()),
        "tear_edges": int(tears.sum()),
        "tear_edge_fraction": float(tears.sum() / smooth_edges.sum()) if np.any(smooth_edges) else None,
        **som.training_stats,
    }
    diagnostics = {"bmus": bmus, "quantization_errors": errors,
                   "topographic_errors": topo_errors, "rim_faces": rim_faces,
                   "tear_edges": mesh_edges[tears], "hits": hits,
                   "node_qe": node_qe, "node_te": node_te,
                   "lattice_support": support}
    return metrics, diagnostics


def segment(som, data, mesh, adjacency, mesh_edges, face_pairs, args, name):
    start = time.perf_counter()
    metrics, diagnostics = evaluate_map(som, data, mesh, mesh_edges, face_pairs,
                                        args.seam_angle, args.tear_hops,
                                        args.sigma, args.n_rings)
    # Include the original's two mesh-aware remaps; IDs affect merge tie breaking.
    corrected, raw, n_clusters, separated = facet.cluster_facet_mesh(mesh, adjacency, som)
    n_separated = len(np.unique(separated))
    result = {"name": name, "som": som, "diagnostics": diagnostics,
              "corrected_bmus": corrected, "raw": raw,
              "separated": separated, "metrics": metrics}
    update_merge(result, mesh, adjacency, args.power_thr, args.max_merges)
    metrics.update(raw_clusters=int(n_clusters), separated_regions=int(n_separated),
                   postprocessing_seconds=time.perf_counter() - start)
    return result


def update_merge(result, mesh, adjacency, threshold, max_merges):
    merged, count = facet.merge_facet_regions(
        mesh, result["separated"], adjacency, threshold, max_merges)
    result["merged"] = merged
    result["metrics"].update(merged_regions=int(count), power_threshold=float(threshold),
                              merge_budget_reached=(len(np.unique(result["separated"])) - count == max_merges))


def color_table(count):
    return facet.get_color_map(count)


def plot_labels(plotter, mesh, labels, title, actor_name):
    view = mesh.copy(deep=True)
    view.cell_data["region"] = labels
    count = len(np.unique(labels))
    if actor_name in plotter.scalar_bars:
        plotter.remove_scalar_bar(actor_name)
    plotter.add_mesh(view, scalars="region", cmap=color_table(count),
                     show_scalar_bar=True, show_edges=True, edge_opacity=0.3,
                     scalar_bar_args={"title": actor_name, "fmt": "%.0f",
                                      "n_labels": min(count, 10)}, name=actor_name)
    plotter.add_text(title, font_size=10, color="#202020", name=actor_name + "_title")


def plot_normal_map(plotter, som, data, result, title):
    from matplotlib.colors import to_rgb
    plotter.add_mesh(pv.Sphere(radius=1), color="white", opacity=0.15)
    cloud = pv.PolyData(np.asarray(data, dtype=float))
    cloud.point_data["cluster"] = result["raw"]
    count = result["metrics"]["raw_clusters"]
    colors = color_table(count)
    plotter.add_mesh(cloud, scalars="cluster", cmap=colors,
                     render_points_as_spheres=True, point_size=5, opacity=0.6,
                     scalar_bar_args={"title": result["name"] + " clusters", "fmt": "%.0f",
                                      "n_labels": min(count, 10), "vertical": False})
    plotter.add_mesh(som.wireframe(), color="grey", opacity=0.5, line_width=1)
    neuron_rgb = np.zeros((som.n_nodes, 3), dtype=np.uint8)
    for node in np.unique(result["corrected_bmus"]):
        label = result["raw"][np.flatnonzero(result["corrected_bmus"] == node)[0]]
        neuron_rgb[node] = (np.asarray(to_rgb(colors[label])) * 255).astype(np.uint8)
    glyphs = pv.PolyData(som.weights.copy()).glyph(scale=False, orient=False, geom=pv.Sphere(radius=0.03))
    glyphs.point_data["colors"] = np.repeat(neuron_rgb, glyphs.n_points // som.n_nodes, axis=0)
    plotter.add_mesh(glyphs, scalars="colors", rgb=True)
    if np.any(som.boundary_mask):
        plotter.add_mesh(som.boundary_wireframe(), color=RIM_COLOR, line_width=3)
        plotter.add_mesh(pv.PolyData(som.weights[som.boundary_mask]), color=RIM_COLOR,
                         point_size=8, render_points_as_spheres=True)
    plotter.add_text(title, font_size=10, color="#202020")
    plotter.view_isometric()
    plotter.reset_camera()


def plot_border_effect(plotter, mesh, diagnostics, metrics):
    view = mesh.copy(deep=True)
    view.cell_data["rim_assignment"] = diagnostics["rim_faces"].astype(np.uint8)
    plotter.add_mesh(view, scalars="rim_assignment", cmap=["#d8dce2", RIM_COLOR],
                     clim=[0, 1], show_scalar_bar=False)
    tears = diagnostics["tear_edges"]
    if len(tears):
        lines = np.column_stack([np.full(len(tears), 2), tears]).ravel()
        plotter.add_mesh(pv.PolyData(mesh.points.copy(), lines=lines),
                         color=TEAR_COLOR, line_width=3)
    fraction = metrics["tear_edge_fraction"]
    rate = f"{fraction:.1%}" if fraction is not None else "N/A"
    plotter.add_text(f"Red: faces assigned to rim ({metrics['boundary_face_fraction']:.1%})\n"
                     f"Magenta: near-normal map tears ({rate})", font_size=10, color="#202020")


def render_dashboard(results, mesh, data, adjacency, args, destination, interactive=False):
    plotter = pv.Plotter(shape=(2 * len(results), 3), off_screen=not interactive,
                         window_size=(2100, 1000 * len(results)),
                         title="Planar SOM boundary ablation")
    object_views = []
    for index, result in enumerate(results):
        row = 2 * index
        metrics = result["metrics"]
        som = result["som"]
        name = result["name"]
        plotter.subplot(row, 0)
        plotter.set_background("white")
        plotter.add_mesh(mesh, color="grey", show_edges=True, edge_opacity=0.2)
        plotter.add_text(f"{name}: input mesh\n{mesh.n_cells} faces", font_size=10, color="#202020")
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.subplot(row, 1)
        plotter.set_background("white")
        plot_normal_map(plotter, som, data, result,
                        f"{name}: trained lattice in normal space\n"
                        + ("Red: open rim; colors: face clusters\n" if np.any(som.boundary_mask)
                           else "Closed lattice; colors: face clusters\n") +
                        f"QE={metrics['quantization_error']:.4f}; TE={metrics['topographic_error']:.1%}")
        plotter.subplot(row, 2)
        plotter.set_background("white")
        plot_border_effect(plotter, mesh, result["diagnostics"], metrics)
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.subplot(row + 1, 0)
        plotter.set_background("white")
        plot_labels(plotter, mesh, result["raw"],
                    f"{name}: raw clustering\n{metrics['raw_clusters']} clusters", f"{name} raw")
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.subplot(row + 1, 1)
        plotter.set_background("white")
        plot_labels(plotter, mesh, result["separated"],
                    f"{name}: disconnected components separated\n{metrics['separated_regions']} regions",
                    f"{name} separated")
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.subplot(row + 1, 2)
        plotter.set_background("white")
        plot_labels(plotter, mesh, result["merged"],
                    f"{name}: power merge\n{metrics['merged_regions']} regions; threshold={args.power_thr:.3f}",
                    f"{name} merged")
        plotter.view_isometric()
        plotter.reset_camera()
        object_views.extend([row * 3, row * 3 + 2,
                             (row + 1) * 3, (row + 1) * 3 + 1, (row + 1) * 3 + 2])
    # Link only views of the object; descriptor-space cameras have a different scale.
    plotter.link_views(object_views)
    if interactive:
        def on_threshold(value):
            for index, result in enumerate(results):
                update_merge(result, mesh, adjacency, value, args.max_merges)
                plotter.subplot(2 * index + 1, 2)
                plot_labels(plotter, mesh, result["merged"],
                            f"{result['name']}: power merge\n{result['metrics']['merged_regions']} regions; "
                            f"threshold={value:.3f}", f"{result['name']} merged")
            plotter.render()

        plotter.subplot(1, 2)
        plotter.add_slider_widget(on_threshold, rng=[0, 1], value=args.power_thr,
                                  title="Shared merge threshold", pointa=(0.12, 0.08),
                                  pointb=(0.88, 0.08), interaction_event="end", fmt="%.3f")
        # Save the view and labels at the last explored slider value.
        plotter.show(auto_close=False)
        plotter.screenshot(str(destination))
    else:
        plotter.show(screenshot=str(destination), auto_close=False)
    plotter.close()


def render_planar_diagnostics(result, args, directory):
    """Flat-grid views keep the outer rim visible even if the learned map folds."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    som, diagnostics = result["som"], result["diagnostics"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    maps = [(diagnostics["hits"] + 1, "BMU hits + 1 (log scale)", "viridis",
             LogNorm(vmin=1, vmax=max(int(diagnostics["hits"].max()) + 1, 2))),
            (diagnostics["node_qe"], "Mean quantization error (inactive nodes: grey)", "magma", None),
            (diagnostics["node_te"], "Topographic error per BMU (inactive nodes: grey)", "magma", None),
            (diagnostics["lattice_support"],
             f"Fixed lattice support at sigma={args.sigma:g}\n"
             + ("Training kernel" if args.kernel == "lattice" else "Geometric diagnostic; NOT the paper training kernel"),
             "viridis", None)]
    for ax, (values, title, cmap_name, norm) in zip(axes.flat, maps):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#d8dce2")
        xy = som.positions[:, :2]
        for a, b in som.edges:
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], color="#b5bcc5", lw=0.5, zorder=0)
        valid = np.isfinite(values)
        if norm is None:
            maximum = 1 if "Topographic" in title else max(float(values[valid].max()), 1e-6) if np.any(valid) else 1
            norm = Normalize(vmin=0, vmax=maximum)
        ax.scatter(xy[:, 0], xy[:, 1], color="#d8dce2", s=65)
        scatter = ax.scatter(xy[valid, 0], xy[valid, 1], c=values[valid],
                             cmap=cmap, norm=norm, s=65)
        for a, b in som.boundary_edges:
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=RIM_COLOR, lw=2, zorder=2)
        ax.scatter(xy[som.boundary_mask, 0], xy[som.boundary_mask, 1],
                   facecolors="none", edgecolors=RIM_COLOR, s=105, linewidths=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("Fixed lattice x")
        ax.set_ylabel("Fixed lattice y")
        fig.colorbar(scatter, ax=ax, shrink=0.8)
    fig.suptitle(f"Open planar {som.grid} grid: {som.rows} x {som.cols}; red = topological rim\n"
                 f"init={args.init}; kernel={args.kernel}; seed={result['metrics']['seed']}")
    fig.savefig(directory / "planar_diagnostics.png", dpi=200)
    fig.savefig(directory / "planar_diagnostics.svg")
    plt.close(fig)


def save_results(results, mesh, data, args, directory):
    records = []
    for result in results:
        stem = "planar" if isinstance(result["som"], PlanarSOM3D) else "spherical"
        metrics, som = result["metrics"], result["som"]
        records.append({"topology": stem, **metrics})
        for stage in ("raw", "separated", "merged"):
            np.savetxt(directory / f"{stem}_{stage}.seg", result[stage], fmt="%d")
        output_mesh = mesh.copy(deep=True)
        output_mesh.cell_data["bmu"] = result["diagnostics"]["bmus"]
        output_mesh.cell_data["rim_assignment"] = result["diagnostics"]["rim_faces"].astype(np.uint8)
        for stage in ("raw", "separated", "merged"):
            output_mesh.cell_data[stage] = result[stage]
        output_mesh.save(directory / f"{stem}.vtp")
        np.savez_compressed(directory / f"{stem}_som.npz", initial_weights=som.initial_weights,
                            weights=som.weights, positions=som.positions, edges=som.edges,
                            boundary_mask=som.boundary_mask, training_indices=som.sample_indices,
                            **result["diagnostics"])
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    notes = ["All diagnostics use original BMUs before small-face correction.",
             "Red identifies rim assignments; rim occupancy alone is not an error.",
             "Map tears and topographic error do not measure segmentation accuracy.",
             "Exact prototype ties are broken by lowest node index."]
    if args.kernel == "lattice":
        notes.append("The lattice kernel applies only to the planar baseline; S-SOM retains seg_facet.py's trainer.")
    else:
        notes.append("The paper kernel uses the original moving-weight neighborhood rule.")
    if args.planar_only:
        notes.append("This run contains only the planar baseline.")
    else:
        notes.append("S-SOM uses the shared seg_facet.py training, prediction, and processing functions with the same seed/settings.")
        notes.append("The planar and enclosing-sphere initializations differ; this compares topology plus initialization.")
    (directory / "metrics.json").write_text(json.dumps({"configuration": configuration,
                                                        "grid_shape": [results[0]["som"].rows, results[0]["som"].cols],
                                                        "descriptor_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                                                        "n_faces": len(data), "notes": notes,
                                                        "results": records}, indent=2, allow_nan=False),
                                            encoding="utf-8")
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def spherical_reference_map(data, spherical_mesh, args, seed):
    """Attach read-only diagnostics to the real seg_facet.py S-SOM model."""
    reference = facet.train_spherical_som(
        data, spherical_mesh, args.radius, args.n_rings, args.init_neu_size,
        args.lr, seed, args.epochs, args.lr_end)
    diagnostic_map = TopologySOM3D(spherical_mesh, args.radius)
    diagnostic_map.weights = reference.get_weights()
    diagnostic_map.initial_weights = reference.initial_weights
    diagnostic_map.sample_indices = reference.sample_indices
    diagnostic_map.training_stats = reference.training_stats
    diagnostic_map.predict = reference.predict
    diagnostic_map.reference_som = reference
    return diagnostic_map


def run(args):
    if args.headless or args.no_render:
        import matplotlib
        matplotlib.use("Agg")
    if args.demo_sphere:
        # Deterministic, closed directional distribution; no injected noise.
        mesh = pv.Sphere(theta_resolution=64, phi_resolution=32).triangulate()
        mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        source_name = "normal_sphere"
    else:
        if not args.input.is_file():
            raise ValueError(f"Input mesh does not exist: {args.input}")
        mesh, adjacency = facet.prepare_facet_mesh(str(args.input), args.smooth_iters)
        source_name = args.input.stem
    if mesh.n_cells == 0:
        raise ValueError("Input mesh has no faces")
    if args.demo_sphere:
        adjacency = build_face_adjacency(mesh)
        smooth_normals(mesh, adjacency, n_iter=args.smooth_iters, feature_angle_deg=30.0)
    # Preserve the original Normals dtype throughout training and prediction.
    data = np.asarray(mesh.cell_data["Normals"])
    if np.any(np.linalg.norm(data, axis=1) < 0.99):
        raise ValueError("The mesh has degenerate faces with non-unit normals")
    mesh_edges, face_pairs = shared_mesh_edges(mesh)
    spherical_mesh = pv.read(str(args.sphere_mesh))
    if (args.rows is None) != (args.cols is None):
        raise ValueError("Specify both --rows and --cols, or neither")
    rows, cols = ((args.rows, args.cols) if args.rows is not None else
                  matched_grid_shape(spherical_mesh.n_points))
    if not args.planar_only and rows * cols != spherical_mesh.n_points:
        raise ValueError(f"Comparison requires equal neuron counts: grid has {rows * cols}, "
                         f"sphere has {spherical_mesh.n_points}. Use a matching sphere or --planar-only.")
    if args.n_rings == 0:
        print("NOTE: n_rings=0 updates only the BMU. This is a topology-disabled control.")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    all_records = []
    for index, seed in enumerate(seeds):
        # Distinct experiment settings get distinct directories, including grid
        # type, budget, and thresholds. A rerun of identical settings replaces
        # only that run's generated artifacts.
        experiment = {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()
                      if key not in ("headless", "no_render", "output_dir", "seed", "seeds")}
        experiment.update(seed=int(seed), grid_shape=[rows, cols], pipeline="seg_facet_shared_v2",
                          descriptor_sha256=hashlib.sha256(data.tobytes()).hexdigest())
        signature = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()[:8]
        run_name = f"{source_name}_{args.kernel}_{args.grid}_{args.init}_{args.sphere_init}_seed{seed}_{signature}"
        directory = args.output_dir / run_name
        directory.mkdir(parents=True, exist_ok=True)
        samples = np.random.RandomState(seed).randint(len(data), size=args.epochs)
        planar = PlanarSOM3D(rows, cols, args.radius, args.grid)
        results = []
        topologies = ["planar"] if args.planar_only else ["planar", "spherical"]
        for topology in topologies:
            start = time.perf_counter()
            if topology == "planar":
                name, som, initialization, kernel = "Planar SOM", planar, args.init, args.kernel
                som.train(data, n_epochs=args.epochs, n_rings=args.n_rings,
                          init_neuron_size=args.init_neu_size, lr0=args.lr, lr_end=args.lr_end,
                          init=initialization, seed=seed, kernel=kernel,
                          sigma0=args.sigma, sigma_end=args.sigma_end, sample_indices=samples)
            else:
                name, initialization, kernel = "S-SOM", "enclosing", "paper"
                som = spherical_reference_map(data, spherical_mesh, args, seed)
            training_time = time.perf_counter() - start
            result = segment(som, data, mesh, adjacency, mesh_edges, face_pairs, args, name)
            result["metrics"].update(seed=int(seed), initialization=initialization,
                                      kernel=kernel, training_seconds=training_time)
            results.append(result)
        if not args.no_render:
            render_dashboard(results, mesh, data, adjacency, args, directory / "comparison.png",
                              interactive=not args.headless and index == 0)
            render_planar_diagnostics(results[0], args, directory)
        records = save_results(results, mesh, data, args, directory)
        all_records.extend(records)
        for record in records:
            tear_fraction = record["tear_edge_fraction"]
            tear_text = f"{tear_fraction:.2%}" if tear_fraction is not None else "N/A"
            print(f"{record['topology']} seed={seed}: QE={record['quantization_error']:.5f}; "
                  f"TE={record['topographic_error']:.2%}; tears={tear_text}; "
                  f"regions={record['raw_clusters']} -> {record['separated_regions']} -> {record['merged_regions']}; "
                  f"updates with neighbors={record['fraction_updates_with_non_bmu']:.1%}")
            if record["merge_budget_reached"]:
                print("NOTE: Merge budget reached; increase --max-merges before interpreting final region counts.")
        print(f"Saved {directory}")
    if len(seeds) > 1:
        # Preserve every seed, and aggregate paired runs without selecting a winner.
        summary = {}
        for topology in ("planar", "spherical"):
            records = [record for record in all_records if record["topology"] == topology]
            if not records:
                continue
            summary[topology] = {"n_runs": len(records)}
            for metric in ("quantization_error", "topographic_error", "tear_edge_fraction",
                           "active_neurons", "raw_clusters", "separated_regions", "merged_regions"):
                values = [record[metric] for record in records if record[metric] is not None]
                summary[topology][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                             "n_valid": len(values)} if values else None
        aggregate_experiment = {**experiment, "seed": seeds}
        aggregate_signature = hashlib.sha256(json.dumps(aggregate_experiment, sort_keys=True).encode()).hexdigest()[:8]
        aggregate_path = args.output_dir / f"{source_name}_{args.kernel}_{args.grid}_{args.init}_{args.sphere_init}_{aggregate_signature}_summary.json"
        aggregate_path.write_text(json.dumps({"seeds": seeds, "summary": summary, "runs": all_records},
                                             indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {aggregate_path}")
    return all_records


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "--obj_file", type=Path, default=ROOT / "datasets/3DPuzzle/brick_part01.obj")
    parser.add_argument("--demo-sphere", action="store_true", help="Closed smooth sphere: a topology stress test, not facet accuracy")
    parser.add_argument("--sphere-mesh", type=Path, default=ROOT / "regular_sphere.obj")
    parser.add_argument("--rows", type=int, help="Automatic: exact factorization of the spherical neuron count")
    parser.add_argument("--cols", type=int)
    parser.add_argument("--grid", choices=["hex", "rect"], default="hex", help="Hex has six interior neighbors, like the icosphere")
    parser.add_argument("--init", choices=["pca", "random", "sample"], default="pca")
    parser.add_argument("--sphere-init", choices=["enclosing"], default="enclosing",
                        help="S-SOM always uses the original enclosing-sphere initialization")
    parser.add_argument("--kernel", choices=["paper", "lattice"], default="paper",
                        help="Planar training kernel; S-SOM always uses the actual seg_facet.py trainer")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of sampled updates, not passes over the data")
    parser.add_argument("--n_rings", "--n-rings", type=int, default=0, help="Same default as seg_facet.py: 0 (BMU only)")
    parser.add_argument("--radius", type=float, default=0.1, help="Feature-space gate for --kernel paper only")
    parser.add_argument("--sigma", type=float, default=3.0, help="Initial lattice Gaussian width in graph hops")
    parser.add_argument("--sigma-end", type=float, default=0.5)
    parser.add_argument("--init_neu_size", "--init-neu-size", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--lr-end", type=float, default=0.01)
    parser.add_argument("--power_thr", "--power-thr", type=float, default=0.39)
    parser.add_argument("--max-merges", type=int, default=1000,
                        help="Common merge budget for both maps; raise if the run reports exhaustion")
    parser.add_argument("--smooth-iters", type=int, default=5)
    parser.add_argument("--seam-angle", type=float, default=5.0, help="Near-normal mesh-edge threshold, degrees")
    parser.add_argument("--tear-hops", type=int, default=2, help="Flag near-normal pairs whose BMUs are farther apart than this")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", help="Paired repeat runs; also writes mean/std metrics")
    parser.add_argument("--planar-only", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Render PNG/SVG figures offscreen and exit")
    parser.add_argument("--no-render", action="store_true", help="Compute labels and metrics without rendering figures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/topology_ablation")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if (args.epochs <= 0 or args.n_rings < -1 or args.max_merges < 0 or args.smooth_iters < 0
            or args.tear_hops < 0 or not 0 <= args.power_thr <= 1 or not 0 <= args.seam_angle <= 180):
        parser.error("Invalid iteration count, graph window, merge threshold, or seam threshold")
    if any(seed < 0 or seed > 2**32 - 1 for seed in (args.seeds or [args.seed])):
        parser.error("Seeds must be in [0, 2**32 - 1]")
    try:
        run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
