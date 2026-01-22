# 3D Mesh Segmentation using Spherical SOM

This project implements an unsupervised 3D surface segmentation method based on **Spherical Self-Organizing Maps (S-SOM)**.
It clusters face normals and geometric information from a 3D mesh using a spherical topology, then refines the segmentation through post-processing steps.
Input: 3D model file (*.obj), and (*.sdf) in case using SDF feature.
 
## Features

- Load a 3D mesh (.obj) and compute face normals
- Train Spherical SOM in an unsupervised manner to segment the surface based on surface descriptors (normals,sdf, curvature)
- Post-process to:
  - Separate disconnected regions
  - Merge small regions
  - Merge similar regions based on surface orientation
- Visualize segmentation results using `pyvista`

## Requirements
- Python 3.7+
- `numpy`
- `pyvista`
- `matplotlib`
- scikit-learn

Install dependencies via bash
pip install -r requirements.txt

## Run
### Facet segmentation using normal vector as feature descriptor
python seg_facet.py --obj_file=./3DPuzzle/brick_part01.obj

### Part segmentation
python seg_part_int.py --input=.\Princeton\30.obj --fea="sdf" --thr=0.135

## Illustration of interactive adjustment of the segment merging threshold
User can adjust the slider interactively to see the segmentation result 

![Interactive](https://github.com/user-attachments/assets/4c04e266-3e20-4031-beb0-73d5600f33d7)


