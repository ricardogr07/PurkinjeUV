Quickstart
==========

This minimal example shows the end-to-end flow: mesh → growth → Purkinje tree → activation → save.
Adjust file paths and parameters as needed.

.. code-block:: python

   from purkinje_uv import FractalTreeParameters, FractalTree, PurkinjeTree

   # 1) Parameters (frozen dataclass). FractalTree will load the mesh from p.meshfile.
   p = FractalTreeParameters(
       meshfile="path/to/endocardium.vtu",  # required by current FractalTree
       init_node_id=0,
       second_node_id=1,
       init_length=0.1,
       N_it=10,
       length=0.1,
       branch_angle=0.15,  # radians
       w=0.1,
       l_segment=0.01,
       # Optional fascicles
       # fascicles_angles=[-0.4, 0.5],
       # fascicles_length=[0.2, 0.4],
   )

   # 2) Grow fractal tree on the surface
   ft = FractalTree(params=p)  # internally loads mesh and computes UV scaling
   ft.grow_tree()

   # 3) Wrap into a PurkinjeTree and activate
   purk = PurkinjeTree(
       nodes=ft.nodes_xyz,
       connectivity=ft.connectivity,
       end_nodes=ft.end_nodes,
   )
   purk.extract_pmj_np_unique()  # select PMJs (if applicable)
   purk.activate_fim()           # run activation model (set stimuli as needed)

   # 4) Persist
   purk.save()         # native format
   purk.save_meshio()  # export to mesh-friendly formats
   purk.save_pmjs()    # save PMJ list if needed

Notes
-----

- `FractalTree` loads the mesh from ``p.meshfile`` and computes UV scaling internally.
- The generation loop follows :numref:`alg-flow`. Single-step calls are in :numref:`alg-seq`.
- See :doc:`/theory/projection_surface` for how 2D growth is projected to the surface.
- For anatomical seeds, see :doc:`/seeding`.

Backend selection (CPU/GPU)
===========================

This package can run on CPU (NumPy) or GPU (CuPy). By default, if CuPy is
*not* installed, everything runs on CPU. If CuPy is installed and available,
you can choose the backend explicitly.

Install
-------

.. code-block:: bash

   # CPU-only
   pip install -U purkinje-uv

   # GPU (Linux/Colab with CUDA 12)
   pip install -U "purkinje-uv[gpu]"

Select backend
--------------

Use the context manager to force CPU or GPU:

.. code-block:: python

   from purkinje_uv.config import use, is_gpu, backend_name
   from purkinje_uv import PurkinjeTree

   # Assume you already built: nodes, connectivity, pmj_ids, x0, x0_vals
   tree = PurkinjeTree(nodes=nodes, connectivity=connectivity, end_nodes=pmj_ids)

   # --- Force CPU (NumPy) ---
   with use("cpu"):
       assert not is_gpu()
       print("Backend:", backend_name())  # e.g., 'numpy'
       act_pmj = tree.activate_fim(x0, x0_vals, return_only_pmj=True)

   # --- Force GPU (CuPy) ---
   # Requires CuPy installed and a visible CUDA device.
   with use("cuda"):
       assert is_gpu()
       print("Backend:", backend_name())  # e.g., 'cupy'
       act_pmj = tree.activate_fim(x0, x0_vals, return_only_pmj=True)

How it routes devices
---------------------

``PurkinjeTree.activate_fim`` checks the active backend via ``is_gpu()`` and
selects the FIM solver device accordingly:

- CPU backend → ``device="cpu"``
- GPU backend → ``device="cuda"``

So switching the context with ``use("cpu")`` / ``use("cuda")`` is enough. No
extra flags are needed at the call site.

Sanity checks
-------------

.. code-block:: python

   from purkinje_uv.config import use, is_gpu, backend_name

   with use("cpu"):
       assert not is_gpu()
       print("CPU OK:", backend_name())

   try:
       with use("cuda"):
           assert is_gpu(), "CuPy/CUDA not available"
           print("GPU OK:", backend_name())
   except Exception as exc:
       print("GPU unavailable, staying on CPU:", exc)

Notes
-----

- On Colab, set **Runtime → Change runtime type → GPU** to use T4/L4/A100.
- The GPU extra installs ``cupy-cuda12x`` on Linux (matches Colab’s CUDA 12).
- If GPU execution fails at runtime, the code falls back to CPU automatically.
