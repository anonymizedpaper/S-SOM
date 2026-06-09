

# 3D Mesh Segmentation using Spherical SOM

This project implements an unsupervised 3D surface segmentation method based on **Spherical Self-Organizing Maps (S-SOM)**.
It clusters face normals and geometric information from a 3D mesh using a spherical topology, then refines the segmentation through post-processing steps.
Input: 3D model file (*.obj)
<img width="3840" height="2088" alt="Seg_facet" src="https://github.com/user-attachments/assets/f43124b4-79de-4c4b-80d7-5c4eacc20f79" />
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


