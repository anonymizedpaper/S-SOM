import numpy as np
import pyvista as pv
from helper import *
from SSOM2D import *
import argparse
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pymeshlab
import os
import glob

def get_color_map(class_count):
    cmap = plt.get_cmap('jet')
    colors = []
    i = 0
    while len(colors) < class_count:
        c = cmap(i / class_count)[:3]  # Use n+1 to avoid endpoints
        if not np.allclose(c, (0, 0, 0)):  # Exclude black
            colors.append(mcolors.to_hex(c))
        i += 1
    return colors
def plot(plotter, mesh, scalar_name, class_count):
    assert(class_count > 0)
    colors = get_color_map(class_count)
    if(class_count <20):
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": class_count})
    else:
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": 20})
def plot_with_title(plotter, mesh, scalar_name, class_count, title):
    assert(class_count > 0)
    colors = get_color_map(class_count)
    if(class_count <20):
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": class_count, "title": title})
    else:
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": 20, "title": title})

def comp_cur(mesh):
    # Compute mean curvature (default per vertex)
    curvature_point = mesh.curvature(curv_type='Mean')

    # Assign it as a point data array
    mesh.point_data["Mean Curvature (Points)"] = curvature_point
    # Apply Gaussian smoothing to the curvature values
    from scipy.ndimage import gaussian_filter

    # Reshape curvature values to match mesh dimensions
    curvature_point_smoothed = gaussian_filter(curvature_point, sigma=2)

    # Assign smoothed curvature as point data
    mesh.point_data["Smoothed Mean Curvature"] = curvature_point_smoothed


    # Approximate face-based curvature: average curvature at the face's vertices
    face_curvature = np.zeros(mesh.n_cells)

    for i, cell in enumerate(mesh.faces.reshape((-1, 4))):  # assuming triangle mesh (3 vertices + 1 size)
        ids = cell[1:]
        face_curvature[i] = np.mean(curvature_point[ids])

    # Apply sigmoid normalization
    percentile = 99
    threshold = np.percentile(face_curvature, percentile)

    a = threshold  # controls steepness
    b = np.mean(face_curvature)  # center of sigmoid
    # Clip values to prevent overflow in exp
    clipped_input = np.clip(a * (face_curvature - b), -500, 500)
    sigmoid_normalized = 1 / (1 + np.exp(-clipped_input))

    return sigmoid_normalized

def normalize_features_by_max_norm(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a feature matrix by dividing ALL features by the maximum L2 norm
    across samples (rows). This keeps relative magnitudes but caps the largest
    sample norm at 1.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    norms = np.linalg.norm(features, axis=1)
    max_norm = float(np.max(norms)) if norms.size else 0.0
    if max_norm < eps:
        return features
    return features / max_norm


def clamp_percentiles(x, pct):
    """Clamp to the [pct, 100-pct] percentile band and rescale it to [0, 1].

    Values are first clipped to the pct-th / (100-pct)-th percentiles, then the
    clipped band is linearly stretched to [0, 1]: out-of-band values land exactly
    at 0.0 (low) or 1.0 (high), and inlier values spread across the full range.
    This removes the skew a few extreme SDF faces would otherwise impose on the
    /max normalization and the downstream feature distance.

    `pct` is in [0, 50); pct<=0 is a no-op. Returns a rescaled copy.
    """
    x = np.asarray(x, dtype=float)
    if pct <= 0:
        return x
    lo = np.percentile(x, pct)
    hi = np.percentile(x, 100.0 - pct)
    if hi - lo < 1e-12:                       # degenerate band -> just clip
        return np.clip(x, lo, hi)
    return (np.clip(x, lo, hi) - lo) / (hi - lo)


def smooth_face_scalar(mesh, face_adjacency, values, n_iter=5, sigma=None):
    """Gaussian-smooth a per-face scalar field over the face-adjacency graph.

    Each iteration replaces every face value with a Gaussian-weighted average of
    itself and its 1-ring face neighbors. A neighbor j of face i is weighted by
    w_ij = exp(-d_ij^2 / (2 sigma^2)), where d_ij is the distance between the two
    face centroids (face i itself has distance 0 -> weight 1, the largest). The
    result is renormalized by the total weight. Repeating n_iter times widens the
    effective kernel (diffusion), giving smoother SDF with less per-face noise.

    sigma defaults to the median centroid spacing between adjacent faces. Returns
    a 1-D array of length mesh.n_cells. n_iter<=0 returns a copy unchanged.
    """
    values = np.asarray(values, dtype=float).ravel()
    n_faces = mesh.n_cells
    if n_iter <= 0:
        return values.copy()

    centroids = np.asarray(mesh.cell_centers().points, dtype=float)   # (F, 3)

    # Padded (F, K) neighbor-index matrix + validity mask (padding slots -> face 0,
    # masked out so they contribute zero weight), same pattern as smooth_normals.
    nbr_lists = [list(face_adjacency.get(f, ())) for f in range(n_faces)]
    max_deg = max((len(nb) for nb in nbr_lists), default=0)
    if max_deg == 0:
        return values.copy()
    nbr_idx = np.zeros((n_faces, max_deg), dtype=np.intp)
    nbr_mask = np.zeros((n_faces, max_deg), dtype=bool)
    for f, nb in enumerate(nbr_lists):
        if nb:
            nbr_idx[f, :len(nb)] = nb
            nbr_mask[f, :len(nb)] = True

    # Centroid distance face -> each neighbor, then Gaussian weights (0 for padding).
    d = np.linalg.norm(centroids[:, None, :] - centroids[nbr_idx], axis=2)   # (F, K)
    if sigma is None:
        valid = d[nbr_mask]
        sigma = float(np.median(valid)) if valid.size else 1.0
    sigma = max(float(sigma), 1e-12)
    w = np.exp(-(d ** 2) / (2.0 * sigma ** 2)) * nbr_mask    # (F, K)
    wsum = w.sum(axis=1)                                     # neighbor weight total

    out = values.copy()
    for _ in range(n_iter):
        nbr_sum = (w * out[nbr_idx]).sum(axis=1)             # weighted neighbor sum
        out = (out + nbr_sum) / (1.0 + wsum)                 # include self (weight 1)
    return out


def make_feature_distribution_figure(features, feature_name, labels=None, figsize=(4, 3)):
    """Build and return a matplotlib Figure showing all per-face feature points.

    - 2-D features (e.g. 'sdf_cur'): scatter of column 0 (x) vs column 1 (y).
    - 1-D features (e.g. 'sdf'):     histogram of the single feature value.
    If `labels` is given, 2-D points are colored by cluster/segment id, so you
    can see how the SOM clusters separate in feature space.

    Returns the Figure (does not show it) so it can be embedded in a PyVista
    subplot via pv.ChartMPL, or shown standalone with plt.show().
    """
    features = np.asarray(features)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    n, d = features.shape

    fig, ax = plt.subplots(figsize=figsize)
    if d >= 2:
        if labels is not None:
            sc = ax.scatter(features[:, 0], features[:, 1], c=labels, cmap='jet',
                            s=8, alpha=0.6, edgecolors='none')
            fig.colorbar(sc, ax=ax, label='cluster')
        else:
            ax.scatter(features[:, 0], features[:, 1], s=8, alpha=0.6, edgecolors='none')
        ax.set_xlabel('feature[0]  (SDF)')
        ax.set_ylabel('feature[1]  (curvature)')
    else:
        ax.hist(features[:, 0], bins=60, color='steelblue', alpha=0.85)
        ax.set_xlabel('feature value (SDF)')
        ax.set_ylabel('count')
    ax.set_title(f"Feature distribution ('{feature_name}', {n} faces, dim={d})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def main(input, radius, n_rings, init_neuron_size, lr,  power_thr=0.15, max_merges=10000, feature_name = "sdf", visualize = True, sdf_cap_pct=1, sdf_smooth_iter=5):
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
        raise ValueError(f"Unsupported features option '{features}'. Use 'sdf' or 'sdf_curvature'.")

    obj_mesh.cell_data['features'] = features

    # # Load mesh 
    spherical_mesh = pv.read("regular_sphere.obj")
    som = SphereSOM(spherical_mesh)
    som.train(features, n_epochs=1000, lr=lr, radius=radius )
    
    #Predict labels
    raw_labels = som.predict(features)
    print("raw_labels", raw_labels)

    raw_labels, raw_labels_count = remap_labels(raw_labels)  # Convert to face labels 0-based indices
    print("SOM clustering: there are {} clusters".format(raw_labels_count))
    obj_mesh.cell_data["raw_labels"] = raw_labels # Assign cluster labels to each face  
    
    #Separate disconnected components
    separated_region_labels = separate_disconnected_components(obj_mesh, face_adjacency, raw_labels)
    separated_region_labels = merge_zero_area_regions(obj_mesh, separated_region_labels, face_adjacency)
    separated_region_labels, _ = remap_labels(separated_region_labels, mesh=obj_mesh)
    obj_mesh.cell_data["separated_region_labels"] = separated_region_labels

    # Merge region based on power
    merged_region_label = separated_region_labels.copy()
    merged_region_label = merge_region_based_on_power(obj_mesh, merged_region_label, face_adjacency,  power_thr=power_thr, max_merges=max_merges, target_n_regs=None, feature_name = "features")
    
    merged_similar_region_labels, _ = remap_labels(merged_region_label)  # Convert to face labels 0-based indices
    obj_mesh.cell_data["merged_similar_region_labels"] = merged_similar_region_labels

    print("merged_similar_region_labels", merged_similar_region_labels)
    with open(input.replace('.obj', '.seg'), 'w') as f: 
        for segment_index in merged_similar_region_labels:
            f.write(f'{segment_index}\n')
        print("segmentation saved to ", input.replace('.obj', '.seg'))

    # Create a 2x3 grid plotter #################################################################
    if visualize:
        plotter = pv.Plotter(shape=(2, 3), title= "Part segmentation by S-SOM")
        
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

        # Live controls: re-run ONLY the region merge (line 174) whenever a slider
        # is released, and redraw this subplot. SOM/SDF stay fixed. Shared state
        # holds the current power_thr and alpha (beta = 1 - alpha).
        ctrl = {"power_thr": float(power_thr), "alpha": 3/4}

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
    parser = argparse.ArgumentParser(description="Spherical SOM surface segmentation.")    
    parser.add_argument("--radius", type=float, default=1, help="Neighborhood radius for SOM")
    parser.add_argument("--input", type=str, default = "./datasets/Princeton/1.obj",  help="Path to the OBJ file")
    parser.add_argument("--n_rings", type=int, default=0, help="Number of rings in the neighborhood for SOM training")
    parser.add_argument("--init_neu_size", type=float, default=2, help="Initial distance of neurons to the origin")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SOM training")
    parser.add_argument("--power_thr", type=float, default=0.2, help="stop when best remaining power < this value")
    parser.add_argument("--fea", type=str, default="sdf_cur", help="Features set to use")
    parser.add_argument("--sdf_cap_pct", type=float, default=1.0, help="Winsorize: clip the lowest/highest this %% of SDF values (0 disables)")
    parser.add_argument("--sdf_smooth_iter", type=int, default=5, help="Gaussian-smoothing iterations for per-face SDF over neighbors (0 disables)")

    args = parser.parse_args()
    if os.path.isdir(args.input):
        # Find all .obj files in the directory
        print("Process all *.obj in ", input)
        obj_pattern = os.path.join(input, "*.obj")
        obj_files = glob.glob(obj_pattern)
        
        if not obj_files:
            print(f"No .obj files found in directory: {input}")
        else:
            print(f"Found {len(obj_files)} .obj file(s) in directory: {input}")
        
        # Process each .obj file without visualization
        # for obj_path in obj_files:
        #     main(obj_path, fea, lr, radius, min_region_face_count, threshold_similarity_merge, visualize=False)
    else:# Process single file with visualization
        main(args.input, args.radius,  args.n_rings, args.init_neu_size, args.lr, args.power_thr, feature_name=args.fea, sdf_cap_pct=args.sdf_cap_pct, sdf_smooth_iter=args.sdf_smooth_iter)
