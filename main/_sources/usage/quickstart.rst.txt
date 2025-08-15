Quickstart
==========

This minimal example shows the end-to-end flow: mesh → growth → Purkinje tree → activation → save.
Adjust file paths and parameters as needed.

.. code-block:: python

   from purkinje_uv import Parameters, Mesh, FractalTree, PurkinjeTree

   # 1) Parameters (paper-aligned knobs)
   p = Parameters()
   # p.length        # median branch length
   # p.branch_angle  # branching angle (rad)
   # p.w             # repulsion weight
   # p.l_segment     # step size (≈ length / N_s)
   # p.N_it          # generations

   # 2) Load/prepare mesh
   mesh = Mesh.from_file("path/to/endocardium.vtu")  # or construct from your data
   mesh.detect_boundary()
   mesh.compute_uvscaling()  # optional if UVs are present

   # 3) Grow fractal tree on the surface
   ft = FractalTree(mesh=mesh, params=p, mesh_uv=mesh)
   ft.grow_tree()

   # 4) Wrap into a PurkinjeTree and activate
   purk = PurkinjeTree(
       nodes=ft.nodes_xyz,
       connectivity=ft.connectivity,
       end_nodes=ft.end_nodes
   )
   purk.extract_pmj_np_unique()  # select PMJs (if applicable)
   purk.activate_fim()           # run activation model

   # 5) Persist
   purk.save()         # native format
   purk.save_meshio()  # export to mesh-friendly formats
   purk.save_pmjs()    # save PMJ list if needed

Notes
-----

- The generation loop follows :numref:`alg-flow`. Single-step calls are in :numref:`alg-seq`.
- See :doc:`/theory/projection_surface` for how 2D growth is projected to the surface.
- For anatomical seeds, see :doc:`/seeding`.
