Anatomical Seeding
==================

Goal
----

Provide initial attachment points for the Purkinje network that respect anatomy, so that
growth starts in physiologically meaningful locations and covers the intended chambers.

Interfaces
----------

- ``Parameters.init_node_id`` – index of the first mesh node (root).
- ``Parameters.second_node_id`` – optional helper to define an initial direction.
- ``Parameters.fascicles_angles`` / ``Parameters.fascicles_length`` – optional hints for
  adding early fascicles with specified angles/lengths (if used in your workflow).

Common strategies
-----------------

1. **Label-driven**
   If the endocardial mesh has region labels (e.g., LV septum, anterior/posterior
   fascicles, RV septum), pick the nearest mesh node to each target label centroid.

2. **Geometric rules**
   Without labels, derive landmarks from geometry:
   - septal band: nodes with minimal distance to the interventricular plane,
   - apex base axis: principal component direction to distinguish basal/apical halves,
   - anterior/posterior split: plane orthogonal to the apex-base axis.

3. **Manual seed**
   For controlled experiments, choose seed indices by inspecting the mesh in ParaView
   and reading the point ids under the cursor.

Mapping to parameters
---------------------

- Set ``init_node_id`` to the chosen root node id.
- If you have a second node id along the initial direction, set ``second_node_id``.
  Otherwise, the initial direction is inferred from local surface geometry.
- Optionally, add fascicle entries to ``fascicles_angles`` and ``fascicles_length`` if your
  protocol initializes multiple early branches.

Example
-------

.. code-block:: python

   import numpy as np
   from purkinje_uv import Parameters, Mesh, FractalTree

   mesh = Mesh.from_file("endocardium.vtu")
   mesh.detect_boundary()

   # Example: pick the mesh node closest to a user-provided landmark (x0)
   x0 = np.array([10.2, -3.7, 22.1])  # mm, example landmark in world coordinates
   root_id = int(mesh.tree.query(x0)[1])  # nearest vertex id using the KD-tree

   p = Parameters()
   p.init_node_id = root_id
   # Optional: define second node to bias the first direction
   # p.second_node_id = ...

   ft = FractalTree(mesh=mesh, params=p, mesh_uv=mesh)
   ft.grow_tree()

Validation checklist
--------------------

- The root node lies on the intended endocardial chamber.
- The initial direction points into the chamber (not out of it).
- After the first few generations, endpoints distribute across the intended regions.
