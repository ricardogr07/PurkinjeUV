Architecture
============

This page gives a high-level view of how the library is structured and how the core classes interact, aligned with the paper’s Section 2.1.

Class diagram
-------------

.. figure:: ../images/uml_class_diagram.png
   :alt: UML class diagram for purkinje-uv
   :align: center
   :width: 90%

   High-level UML for the generation and activation stack.

Relationships (at a glance)
---------------------------

- **PurkinjeTree** ◼→ **FractalTree**: composition `1 → 1`
- **FractalTree** ◼→ **FractalTreeParameters**: composition `1 → 1`
- **FractalTree** ◼→ **Mesh** (role: ``mesh``): composition `1 → 1`
- **FractalTree** ◼→ **Mesh** (role: ``mesh_uv``): composition `1 → 1`
- **FractalTree** ◇→ **Edge**: aggregation `1 → 0..*`
- **Branch** → **Mesh**: association `1 → 1`
- **Branch** → **Nodes**: association `1 → 1`

Workflow overview
-----------------

.. figure:: ../images/workflow_drawio.png
   :alt: Create → Activate → Save workflow
   :align: center
   :width: 90%

   End-to-end workflow: parameterize → generate (FractalTree) → build (PurkinjeTree) → activate → save.

Object lifecycle (short)
------------------------

1. Initialize ``FractalTreeParameters`` and load ``Mesh``.
2. Create ``FractalTree(params)`` and call ``grow_tree()``:
   - internally creates/uses ``Branch`` objects,
   - accumulates ``Edge`` entries and 3D ``nodes_xyz`` + ``connectivity``,
   - maintains ``end_nodes``.
3. Build ``PurkinjeTree(nodes=..., connectivity=..., end_nodes=...)``.
4. (Optional) Extract PMJs, run activation (e.g., ``activate_fim()``).
5. Persist with ``save()``, ``save_meshio()``, and ``save_pmjs()``.

Assumptions & invariants
------------------------

See :doc:`/dev/data_model_invariants` for the formal checklist (acyclic connectivity, degree-1 terminals, valid node indices, no self-edges, etc.).

API reference
-------------

For the full public surface (methods, parameters, and module layout), see :doc:`/api/modules`.
