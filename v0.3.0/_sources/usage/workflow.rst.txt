Workflow
========

End-to-end steps
----------------

1. **Parameters** – set growth/spacing/branching knobs (``FractalTreeParameters``).
2. **Mesh prep** – load endocardium, run ``detect_boundary()``, optionally ``compute_uvscaling()``.
3. **Generate** – instantiate ``FractalTree`` and call ``grow_tree()`` (projection+collision inside).
4. **Build** – create ``PurkinjeTree`` from nodes/connectivity/end nodes.
5. **Activate** – compute activation (e.g., ``activate_fim()``) and PMJs as needed.
6. **Save/Export** – write network, mesh outputs, and PMJ files.

Flow figure
-----------

.. figure:: ../images/workflow_drawio.png
   :alt: Workflow for Purkinje generation, activation, and export
   :align: center
   :width: 85%

   Create → Activate → Save workflow (see :numref:`alg-flow` for the algorithm loop).
