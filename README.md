# PurkinjeUV

[![PyPI version](https://badge.fury.io/py/purkinje-uv.svg)](https://badge.fury.io/py/purkinje-uv) [![CI](https://github.com/ricardogr07/purkinje-uv/actions/workflows/ci.yml/badge.svg)](https://github.com/ricardogr07/PurkinjeUV/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

PurkinjeUV is a modular Python package for constructing, simulating, and exporting fractal-based Purkinje networks over anatomical or idealized cardiac surface meshes. It offers a flexible architecture for working with geometries via OBJ, VTK, and GMSH, and supports UV mapping, eikonal solvers, and export utilities.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Requirements](#requirements)
- [Attributions and Credits](#attributions-and-credits)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)

## Features

- Fractal-based generation of Purkinje networks constrained to 3D surface meshes
- Fast Iterative Method (FIM)-based eikonal solver for activation time simulation
- Surface processing tools, including Laplacian-based UV mapping
- VTK and IGB utilities for visualization and interoperability with scientific tools
- Fully modular and scriptable, suitable for both research and reproducible simulation pipelines

## Installation

Install the latest release from PyPI:

```bash
pip install purkinje-uv
```

## Getting Started

```python
from purkinje_uv import Parameters
from purkinje_uv.example_fractal_tree_3d import Fractal_Tree_3D

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

Runnable notebooks:

- `examples/demo_obj/demo_obj_fractal_tree.ipynb`
- `examples/demo_gmsh/demo_fractal_tree_biventricular.ipynb`

## Requirements

- Python ≥ 3.10
- Optional:
  - `pyvista` for visualization
  - `cupy` for GPU acceleration (used in FIM solver)
  - `gmsh` and `cardiac-geometries` for realistic biventricular geometries

## Attributions and Credits

Based on the work by [Francisco Sahli](https://github.com/fsahli):
- fractal-tree: https://github.com/fsahli/fractal-tree  
- purkinje-learning: https://github.com/fsahli/purkinje-learning  

**References:**
Sahli Costabal, F., Yao, J., & Kuhl, E. (2016). Predicting the cardiac toxicity of drugs using a hybrid multiscale model of the heart. *Journal of the Mechanical Behavior of Biomedical Materials*, 62, 217–231. DOI: 10.1016/j.jmbbm.2016.05.004

Maintained by Ricardo García Ramírez (July 2025)

## Citation

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
  howpublished = {\url{https://github.com/ricardogr07/purkinje-uv}}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
