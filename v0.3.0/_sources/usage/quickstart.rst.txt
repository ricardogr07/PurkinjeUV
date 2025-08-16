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
