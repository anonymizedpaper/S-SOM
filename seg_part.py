import numpy as np
import pyvista as pv
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

def process_obj_file(obj_file, feature_name, lr, radius, min_region_face_count, threshold_similarity_merge, visualize=True):   
    obj_mesh = load_obj_with_face_normals(obj_file)
    face_adjacency = build_face_adjacency(obj_mesh)
    
    cur = comp_cur(obj_mesh) 
    if feature_name == "sdf": # Select feature set based on CLI argument        
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(obj_file)
        ms.compute_scalar_by_shape_diameter_function_per_vertex_gpu(coneangle = 120, onprimitive = 'On Faces', removeoutliers = True, numberrays = 180)
        sdf = ms.current_mesh().face_scalar_array() 
        sdf = sdf/np.max(sdf)       
        features = np.array(sdf).reshape(-1, 1)
    elif feature_name == "sdf_cur":
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(obj_file)
        ms.compute_scalar_by_shape_diameter_function_per_vertex_gpu(coneangle = 120, onprimitive = 'On Faces', removeoutliers = True, numberrays = 180)
        sdf = ms.current_mesh().face_scalar_array()
        sdf = sdf/np.max(sdf)
        cur_normalized = cur/np.max(cur)
        features = np.concatenate((sdf.reshape(-1, 1), cur_normalized.reshape(-1, 1)), axis=1)

        norms = np.linalg.norm(features, axis=1)
        max_norm = float(np.max(norms))
        features = features/max_norm
    else:
        raise ValueError(f"Unsupported features option '{features}'. Use 'sdf' or 'sdf_curvature'.")

    obj_mesh.cell_data['features'] = features

    # # Load mesh and assign 6D features, train SOM
    #data_for_som = np.concatenate((translated_xyz*(1-normal_weight), obj_mesh.cell_data['Normals']*normal_weight), axis=1)
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
    obj_mesh.cell_data["separated_region_labels"] = separated_region_labels 
    
    #Merge small regions to their biggest neighbor
    region_labels = merge_small_regions(obj_mesh, separated_region_labels, face_adjacency, min_region_face_count)
    merged_small_region_labels, merged_region_labels_count = remap_labels(region_labels)  # Convert to face labels 0-based indices
    obj_mesh.cell_data["merged_small_region_labels"] = merged_small_region_labels
    
    # Merge similar regions based features
    merged_region_label_temp = merged_small_region_labels.copy()
    merged_region_label_temp = merge_similar_neighbor_regions(obj_mesh, merged_region_label_temp, face_adjacency, threshold_similarity_merge)
    merged_similar_region_labels, merged_similar_region_labels_count = remap_labels(merged_region_label_temp)  # Convert to face labels 0-based indices
    obj_mesh.cell_data["merged_similar_region_labels"] = merged_similar_region_labels

    print("merged_similar_region_labels", merged_similar_region_labels)
    with open(obj_file.replace('.obj', '.seg'), 'w') as f: #suppose there is *.sdf file in the same folder
        for segment_index in merged_similar_region_labels:
            f.write(f'{segment_index}\n')

    # Create a 2x3 grid plotter #################################################################
    if visualize:
        plotter = pv.Plotter(shape=(2, 3))

        plotter.subplot(0, 0)
        plotter.add_text("Initial SOM Clustering", font_size=12)
        plot(plotter, obj_mesh, "raw_labels", raw_labels_count)

        plotter.subplot(0, 1)
        plotter.add_text("After Disconnected Component Separation", font_size=12)
        obj_mesh01 = obj_mesh.copy()
        plot(plotter,obj_mesh01, "separated_region_labels", merged_region_labels_count)
        
        plotter.subplot(0, 2)
        plotter.add_text("After Small Region Merging", font_size=12)
        obj_mesh10 = obj_mesh.copy()
        plot(plotter,obj_mesh10, "merged_small_region_labels", merged_region_labels_count)
        
        plotter.subplot(1, 0)
        plotter.add_text("Final Segmentation", font_size=12)
        obj_mesh11 = obj_mesh.copy()
        plot_with_title(plotter, obj_mesh11, "merged_similar_region_labels", merged_similar_region_labels_count, "Segment ID")

        # Add SDF visualization
        plotter.subplot(1, 1)
        plotter.add_text("SDF Values", font_size=12)
        obj_mesh_sdf = obj_mesh.copy()
        obj_mesh_sdf["sdf"] = sdf
        plotter.add_mesh(obj_mesh_sdf, scalars='sdf', cmap = "jet", show_scalar_bar=True, show_edges=True, edge_opacity=0.2)

        # Add curvature visualization
        plotter.subplot(1, 2)
        plotter.add_text("Curvature Values", font_size=12)
        obj_mesh_curv = obj_mesh.copy()
        plotter.add_mesh(obj_mesh_curv, scalars='Smoothed Mean Curvature',  cmap = "jet", show_scalar_bar=True, show_edges=True, edge_opacity=0.2)

        plotter.show()


def main(input, fea, lr, radius, min_region_face_count, threshold_similarity_merge):
    # Check if obj_file is a directory
    if os.path.isdir(input):
        # Find all .obj files in the directory
        print("Process all *.obj in ", input)
        obj_pattern = os.path.join(input, "*.obj")
        obj_files = glob.glob(obj_pattern)
        
        if not obj_files:
            print(f"No .obj files found in directory: {input}")
            return
        
        print(f"Found {len(obj_files)} .obj file(s) in directory: {input}")
        
        # Process each .obj file without visualization
        for obj_path in obj_files:
            process_obj_file(obj_path, fea, lr, radius, min_region_face_count, threshold_similarity_merge, visualize=False)
    else:
        # Process single file with visualization
        process_obj_file(input, fea, lr, radius, min_region_face_count, threshold_similarity_merge, visualize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spherical SOM surface segmentation.")    
    parser.add_argument("--input", type=str, required=True, help="Path to the OBJ file(s)")
    parser.add_argument("--fea", type=str, default="sdf", help="Features set to use")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SOM training")
    parser.add_argument("--radius", type=float, default=1, help="Neighborhood radius for SOM")    
    parser.add_argument("--min_region_face_count", type=float, default=0.01, help="Minimum proportion of faces per region")
    parser.add_argument("--threshold_similarity_merge", type=float, default=0.2, help="Similarity threshold for region merging")
    args = parser.parse_args()
    main(args.input, args.fea, args.lr, args.radius, args.min_region_face_count, args.threshold_similarity_merge)
