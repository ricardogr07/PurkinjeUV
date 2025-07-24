# PurkinjeUV

`PurkinjeUV` is a modular Python package for constructing, simulating, and exporting fractal-based Purkinje networks over anatomical or idealized cardiac surface meshes. It offers a flexible architecture for working with geometries via OBJ, VTK, and GMSH, and supports UV mapping, eikonal solvers, and export utilities.

---

## Features

- Fractal-based generation of Purkinje networks constrained to 3D surface meshes
- Fast Iterative Method (FIM)-based eikonal solver for activation time simulation
- Surface processing tools, including Laplacian-based UV mapping
- VTK and IGB utilities for visualization and interoperability with scientific tools
- Fully modular and scriptable, suitable for both research and reproducible simulation pipelines

---

## Installation

Install the latest release from PyPI:

```bash
pip install purkinje-uv
```

---

## Getting Started

Example to generate and visualize a basic fractal Purkinje tree on a sphere mesh:

```python
from purkinje_uv import Parameters
from example_fractal_tree_3d import Fractal_Tree_3D

param = Parameters()
param.meshfile = "data/sphere.obj"
param.save = True
param.filename = "output/fractal_tree"

branches, nodes = Fractal_Tree_3D(param)
```

Visualization:

```python
import pyvista as pv
tree = pv.read("output/fractal_tree.vtu")
tree.plot()
```

You can find runnable notebooks under:

- `examples/demo_obj/demo_obj_fractal_tree.ipynb`
- `examples/demo_gmsh/demo_fractal_tree_biventricular.ipynb`

---

## Repository Structure

```
purkinje_uv/               # Core module
examples/
  demo_obj/                # OBJ-based toy examples
  demo_gmsh/               # Realistic biventricular geometries via gmsh
data/                      # Sample meshes
```

---

## Requirements

- Python ≥ 3.8
- Optional:
  - `pyvista` for visualization
  - `cupy` for GPU acceleration (used in FIM solver)
  - `gmsh` and `cardiac-geometries` for realistic biventricular geometries

---

## Attributions and Credits

This package is based on the work by [Francisco Sahli](https://github.com/fsahli), and specifically the repository:

- https://github.com/fsahli/fractal-tree  
  Which implements the methodology from:

> Sahli Costabal, F., Yao, J., & Kuhl, E. (2016). Predicting the cardiac toxicity of drugs using a hybrid multiscale model of the heart. *Journal of the Mechanical Behavior of Biomedical Materials*, 62, 217–231.  
> DOI: 10.1016/j.jmbbm.2016.05.004

This project also incorporates structural ideas and processing utilities from the later repository:

- https://github.com/fsahli/purkinje-learning

This package does not attempt to replicate or replace either of the original works. Instead, it adapts and modularizes key components for clarity, reproducibility, and ease of extension, particularly in the context of modern cardiac simulation workflows and integration with UV-mapped meshes.

All intellectual credit for the core algorithms and the geometric methodology belongs to the original authors.


---

## Maintainer and Modifications

This current repository is maintained by Ricardo García Ramírez as of July 2025. It introduces the following:

- Modular and pip-installable architecture (`purkinje-uv`)
- Improved demo scripts and Jupyter notebooks
- PyVista-based visualization integration
- Clean separation of mesh handling, branching logic, and exporting
- Explicit support for OBJ and GMSH-based geometries

---

## Citation

If you use this library in your research or publications, please cite the original article as well as the repository:

```bibtex
@article{sahli2016hybrid,
  title={Predicting the cardiac toxicity of drugs using a hybrid multiscale model of the heart},
  author={Sahli Costabal, Feras and Yao, Jiajian and Kuhl, Ellen},
  journal={Journal of the Mechanical Behavior of Biomedical Materials},
  volume={62},
  pages={217--231},
  year={2016},
  publisher={Elsevier}
}

@misc{purkinjeuv2025,
  author = {Ricardo García Ramírez},
  title = {PurkinjeUV: Modular Fractal Purkinje Generator on Surface Meshes},
  year = {2025},
  howpublished = {\url{https://github.com/ricardogr07/PurkinjeUV}}
}
```

---

## License

This repository is released under the MIT License. See the [LICENSE](LICENSE) file for full terms.
