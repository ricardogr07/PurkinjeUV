Reproducibility
===============

Deterministic runs
------------------

Set seeds **before** importing/constructing components that sample randomness.

.. code-block:: python

   import os, random, numpy as np
   os.environ["PYTHONHASHSEED"] = "0"
   random.seed(42)
   np.random.seed(42)

   from purkinje_uv import FractalTreeParameters, Mesh, FractalTree

   p = FractalTreeParameters()
   # p.length, p.branch_angle, p.w, p.l_segment, p.N_it ...

   mesh = Mesh.from_file("endocardium.vtu")
   mesh.detect_boundary()
   mesh.compute_uvscaling()

   ft = FractalTree(mesh=mesh, params=p, mesh_uv=mesh)
   ft.grow_tree()

Version pinning
---------------

- **Python** and **OS**: record versions alongside runs.
- **Dependencies**: pin versions via a lock or env export:

  .. code-block:: bash

     python -m pip freeze > requirements-lock.txt

- **Config**: save the full ``FractalTreeParameters`` object used for generation next to outputs.

Run manifest (recommended)
--------------------------

- Project commit SHA / run id
- Python & package versions
- Mesh file path & checksum
- FractalTreeParameters (JSON dump)
- Random seeds
- Output file paths
