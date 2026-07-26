

# 3D Mesh Segmentation using Spherical SOM

This project implements an unsupervised 3D surface segmentation method based on **Spherical Self-Organizing Maps (S-SOM)**.
It clusters face normals and geometric information from a 3D mesh using a spherical topology, then refines the segmentation through post-processing steps.
Input: 3D model file (*.obj)

### Facet segmentation via normmal vectors
<img width="3840" height="2088" alt="Seg_facet" src="https://github.com/user-attachments/assets/f43124b4-79de-4c4b-80d7-5c4eacc20f79" />

### Part segmentation via SDF (and/or) curvatures
<img width="3835" height="2299" alt="Seg_part_interactive" src="https://github.com/user-attachments/assets/e737af08-94b7-4f98-82fb-45066f4fd29e" />
 
## Features

- Load a 3D mesh (.obj) and compute face normals
- Train Spherical SOM in an unsupervised manner to segment the surface based on surface descriptors (normals,sdf, curvature)
- Post-process to:
  - Separate disconnected regions
  - Merge similar regions to avoid oversegmentation using a power score function 
- Visualize segmentation results using `pyvista`

## Requirements
- Python 3.13.5
- Important libs:
 - numpy
 - pyvista
 - matplotlib
 - scikit-learn
 - pymeshlab

Install dependencies via bash:
pip install -r requirements.txt

## Run
- Facet segmentation using normal vector as feature descriptor:
python seg_facet.py --obj_file=./datasets/3DPuzzle/brick_part01.obj

- Part segmentation: --fea can be either "sdf_cur", "sdf", or "cur"
python seg_part.py --input=./datasets/Princeton/30.obj --fea="sdf_cur" 

## Illustration of interactive adjustment of the segment merging threshold
User can adjust the slider interactively to see the segmentation result 

![Interactive](https://github.com/user-attachments/assets/4c04e266-3e20-4031-beb0-73d5600f33d7)


## Ablation: Spherical Topology

We conduct an ablation study replacing the spherical lattice of S-SOM with a traditional planar SOM, keeping the descriptor, the number of neurons, the neighborhood kernel and learning-rate schedule, the iteration budget, and the entire downstream post-processing pipeline unchanged. Since the proposed enclosing-sphere initialization is defined only for a spherical lattice, a planar SOM cannot be initialized identically. We therefore evaluate the planar baseline under both of its standard initialization schemes, namely PCA initialization, which places the neurons on the plane spanned by the two leading principal directions of the input normals, and random initialization, which draws weights from an isotropic Gaussian, and compare S-SOM against the stronger of the two. Any remaining difference is thus attributable to the lattice topology together with the initialization it admits, and cannot be explained by a disadvantageous choice of planar initialization.



Face normals are unit vectors, so the input to the SOM lies on the closed sphere $S^2$, and each planar facet of the mesh corresponds to one tight cluster of normals on that sphere. A planar SOM, however, arranges its neurons on an open two-dimensional grid, which cannot wrap around a closed surface no matter how it is initialized. With the PCA (linear) initialization, the grid forms a flat sheet through the data mean and thus inevitably slices through the sphere of normals. Clusters near the slice are then surrounded by several grid neurons at nearly equal distances, while clusters far from the sheet are close to none. With random initialization, the neurons are scattered at arbitrary directions and radii, so their proximity to a given cluster is a matter of chance rather than orientation. In both cases, the Euclidean best-matching-unit competition begins with several nearly equidistant neurons per normal cluster. Each of them wins a share of the cluster's samples and is pulled into it, and because the updates are purely attractive, the redundant winners are never eliminated. As a result, a single planar facet is quantized by several near-duplicate neurons and appears in the segmentation as interleaved fragments, which is precisely the observed oversegmentation. These co-winning neurons also tend to occupy distant positions on the grid, either because the cluster straddles the sheet's cut or boundary in the PCA case, or because recruitment ignores grid position entirely in the random case. Consequently, one coherent planar region is torn across far-apart parts of the map. The spherical lattice of S-SOM avoids both problems by construction. Its closed lattice keeps every neuron equidistant from the data sphere, so the competition depends only on orientation, each cluster of normals is captured by a single neuron, and neighboring normals remain neighbors on the lattice.
