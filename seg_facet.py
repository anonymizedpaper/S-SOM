from turtle import title
import numpy as np
import pyvista as pv
from SSOM3D import *
from helper import *
import argparse
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pathlib import Path


def get_color_map(class_count):
    # 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    #                   'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    #                   'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
    #                   'turbo', 'nipy_spectral', 'gist_ncar']
    cmap = plt.get_cmap('gist_rainbow')
    colors = []
    i = 0
    while (len(colors) < class_count):
        c = cmap(i / class_count)[:3]  # Use n+1 to avoid endpoints
        if not np.allclose(c, (0, 0, 0)):  # Exclude black
            colors.append(mcolors.to_hex(c))
        i += 1
    return colors
def plot_with_title(plotter, mesh, scalar_name, class_count, title):
    assert(class_count > 0)
    colors = get_color_map(class_count)
    if(class_count <20):
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": class_count, "title": title})
    else:
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": 20, "title": title})
def plot(plotter, mesh, scalar_name, class_count, max_bar_labels=20):
    assert(class_count > 0)
    colors = get_color_map(class_count)
    # Cap the number of ticks drawn on the scalar bar: with many regions, drawing
    # one tick per class produces an unreadable, overlapping bar. Colors still span
    # all classes; only the displayed tick labels are thinned out.
    n_labels = min(class_count, max_bar_labels)
    plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.3, scalar_bar_args={"fmt": "%.0f", "n_labels": n_labels})


def prepare_facet_mesh(input, smooth_iters=5):
    """Shared input preparation for the original and topology comparison scripts."""
    obj_mesh = load_obj_with_face_normals(input)
    face_adjacency = build_face_adjacency(obj_mesh)
    smooth_normals(obj_mesh, face_adjacency, n_iter=smooth_iters, feature_angle_deg=30.0)
    return obj_mesh, face_adjacency


def train_spherical_som(data, spherical_mesh, radius=0.1, n_rings=0,
                        init_neuron_size=2, lr=0.2, seed=0, n_epochs=2000,
                        lr_end=0.01, verbose=True):
    """Run the actual SphereSOM3D trainer with a reproducible sample sequence."""
    som = SphereSOM3D(spherical_mesh.copy(deep=True), radius=radius)
    som.initial_weights = normalize(som.mesh.points) * init_neuron_size
    som.sample_indices = np.random.RandomState(seed).randint(len(data), size=n_epochs)
    neighbor_counts = []
    original_neighbors = som.get_n_ring_neighbors

    def monitored_neighbors(bmu, rings):
        nodes = original_neighbors(bmu, rings)
        neighbor_counts.append(0)

        def iterate():
            for node in nodes:
                if node != bmu and np.linalg.norm(som.mesh.points[node] - som.mesh.points[bmu]) <= radius:
                    neighbor_counts[-1] += 1
                yield node
        return iterate()

    # Observe eligible neighbor updates without changing their order or weights.
    som.get_n_ring_neighbors = monitored_neighbors
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        som.train(data, n_epochs=n_epochs, n_rings=n_rings,
                  init_neuron_size=init_neuron_size, lr0=lr, lr_end=lr_end, verbose=verbose)
    finally:
        np.random.set_state(state)
        som.get_n_ring_neighbors = original_neighbors
    som.training_stats = {
        "mean_updated_non_bmu_neurons": float(np.mean(neighbor_counts)),
        "fraction_updates_with_non_bmu": float(np.mean(np.asarray(neighbor_counts) > 0)),
    }
    return som


def cluster_facet_mesh(obj_mesh, face_adjacency, som):
    """The original BMU, small-face, and component steps, including label ordering."""
    bmu_labels = som.predict(obj_mesh.cell_data['Normals'])
    bmu_labels = merge_small_faces(obj_mesh, bmu_labels, face_adjacency, area_ratio=0.03)
    raw_labels, raw_labels_count = remap_labels(bmu_labels, mesh=obj_mesh)
    obj_mesh.cell_data["raw_labels"] = raw_labels
    separated = separate_disconnected_components(obj_mesh, face_adjacency, raw_labels)
    separated, _ = remap_labels(separated, mesh=obj_mesh)
    separated, _ = remap_labels(separated, mesh=obj_mesh)
    obj_mesh.cell_data["separated_region_labels"] = separated
    return bmu_labels, raw_labels, raw_labels_count, separated


def merge_facet_regions(obj_mesh, separated, face_adjacency, power_thr, max_merges=1000):
    """The original slider's merge operation, shared by both entry points."""
    merged = merge_region_based_on_power(
        obj_mesh, separated.copy(), face_adjacency,
        power_thr=float(power_thr), max_merges=max_merges,
        target_n_regs=None, verbose=False)
    return remap_labels(merged)


def main(input, radius, n_rings, init_neuron_size, lr, power_thr=0.15, max_merges=1000,
         seed=0, n_epochs=2000, lr_end=0.01, smooth_iters=5, sphere_mesh=None):

    obj_mesh, face_adjacency = prepare_facet_mesh(input, smooth_iters)

    import time
    start_time = time.time()
    data_for_som = obj_mesh.cell_data['Normals']   # already unit-length (compute_normals + smooth_normals)

    spherical_mesh = pv.read(str(sphere_mesh or Path(__file__).resolve().with_name("regular_sphere.obj")))
    som = train_spherical_som(data_for_som, spherical_mesh, radius, n_rings,
                              init_neuron_size, lr, seed, n_epochs, lr_end)
    
    #Predict labels
    bmu_labels, raw_labels, raw_labels_count, separated_region_labels = cluster_facet_mesh(
        obj_mesh, face_adjacency, som)
    print("SOM clustering: there are {} clusters".format(raw_labels_count))
    obj_mesh.cell_data["raw_labels"] = raw_labels # Assign cluster labels to each face
    
    # The power-based region merge is now driven by the power_thr slider in
    # subplot(1, 1) (see render_segmentation below), so it is not run here.

    end_time = time.time()
    running_time = end_time - start_time
    print(f"Total running time: {running_time:.2f} seconds")

    # Start plotting, Create a 2x3 grid plotter ##################################################
    plotter = pv.Plotter(shape=(2, 3), title= "Facet segmentation by S-SOM")

    plotter.subplot(0, 0) #-----------------------------------------------
    plotter.add_mesh(obj_mesh, color='grey', show_edges=True, edge_opacity=0.2)      

    plotter.subplot(0, 1) #-----------------------------------------------
    plotter.add_mesh(pv.Sphere(radius=1.0), color='white', opacity=0.15)

    # Per-node colors. Winning neurons (BMU of >=1 face) get their label color
    # (matching the segmentation plots); neurons that win no data are black.
    neuron_points = som.get_weights()
    n_nodes = len(neuron_points)
    winners = np.unique(bmu_labels)                       # neurons that are a BMU (node ids)
    # raw_labels = remap_labels(bmu_labels): each winning node maps to a 0-based
    # label (sorted by node id). Rebuild that same mapping for the colors.
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

    # 3) The trained SOM neurons, colored as computed above.
    neuron_glyphs = pv.PolyData(neuron_points).glyph(scale=False, geom=pv.Sphere(radius=0.03))
    points_per_neuron = neuron_glyphs.n_points // n_nodes   # divides evenly: same geom per node
    neuron_glyphs.point_data["colors"] = np.repeat(neuron_rgb, points_per_neuron, axis=0)
    plotter.add_mesh(neuron_glyphs, scalars="colors", rgb=True, label="SOM neurons")
    #plotter.add_legend()

    plotter.subplot(1, 0) #-----------------------------------------------
    plot(plotter, obj_mesh, "raw_labels", raw_labels_count) 

    plotter.subplot(1, 1) #-----------------------------------------------
    obj_mesh01 = obj_mesh.copy()
    plot(plotter,obj_mesh01, "separated_region_labels",  len(np.unique(separated_region_labels)))   
    
    plotter.subplot(1, 2) #-----------------------------------------------
    # Live power_thr control: re-run ONLY the region merge (SOM training is not
    # repeated) whenever the slider is released, and redraw this subplot.
    obj_mesh12 = obj_mesh.copy()

    # Holds the most recently rendered merge so it can be saved after the
    # interactive session closes (reflects the final power_thr the user explored).
    latest = {"labels": None}

    def render_segmentation(power_thr_val):
        labels, count = merge_facet_regions(
            obj_mesh, separated_region_labels, face_adjacency, power_thr_val, max_merges)
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

    # Save the final (slider-tuned) segmentation: one face label per line, to a
    # .seg file next to the input mesh.
    merged_similar_region_labels = latest["labels"]
    if merged_similar_region_labels is not None:
        seg_path = input.replace('.obj', '.seg')
        with open(seg_path, 'w') as f:
            for segment_index in merged_similar_region_labels:
                f.write(f'{segment_index}\n')
        print("segmentation saved to ", seg_path)

    #plotter.subplot(1, 2) #-----------------------------------------------
     #End plotting               ############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spherical SOM surface segmentation.")
    parser.add_argument("--input", type=str, default = "./datasets/3DPuzzle/brick_part01.obj",  help="Path to the OBJ file")
    parser.add_argument("--radius", type=float, default=0.1, help="Neighborhood radius for SOM")
    parser.add_argument("--n_rings", type=int, default=0, help="Number of rings in the neighborhood for SOM training")
    parser.add_argument("--init_neu_size", type=float, default=2, help="Initial distance of neurons to the origin")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate for SOM training")
    parser.add_argument("--power_thr", type=float, default=0.39, help="stop when best remaining power < this value")
    parser.add_argument("--seed", type=int, default=0, help="Training seed shared with seg_facet_planar.py")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr-end", type=float, default=0.01)
    parser.add_argument("--smooth-iters", type=int, default=5)
    parser.add_argument("--max-merges", type=int, default=1000)
    parser.add_argument("--sphere-mesh", type=Path, default=Path(__file__).resolve().with_name("regular_sphere.obj"))

    args = parser.parse_args()
    main(args.input, args.radius, args.n_rings, args.init_neu_size, args.lr, args.power_thr,
         args.max_merges, args.seed, args.epochs, args.lr_end, args.smooth_iters, args.sphere_mesh)
