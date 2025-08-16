2D Fundamentals
=======================

Overview
--------

We treat the Purkinje network as a **fractal tree**. Each branch grows by adding short
segments at its tip; the direction is steered by a repulsion field that keeps new points
away from the already-built tree and preserves spacing. The page summarizes the growth
rule used in the paper :cite:`AlvarezBarrientos2023` and maps it to the library API.

Discrete growth rule
--------------------

Let :math:`\mathbf{x}^i` be the end point of the *i*-th segment of a branch and
:math:`\mathbf{d}^i` the **unit** direction of that segment.
Growth proceeds by computing a new direction and then advancing a fixed step:

.. math::
   :label: eq:dir_update

   \mathbf{d}^{\,i+1}
   \;=\;
   \frac{\mathbf{d}^{\,i} + w\,\nabla d_{\mathrm{CP}}(\mathbf{x}^i)}
        {\left\lVert \mathbf{d}^{\,i} + w\,\nabla d_{\mathrm{CP}}(\mathbf{x}^i)\right\rVert},
   \qquad i>1,

.. math::
   :label: eq:pos_update

   \mathbf{x}^{\,i+1}
   \;=\;
   \mathbf{x}^{\,i} + \frac{\ell_b}{N_s}\,\mathbf{d}^{\,i+1}.

Where

- :math:`d_{\mathrm{CP}}(\mathbf{x})` is the **distance to the closest point** already in the tree,
  and :math:`\nabla d_{\mathrm{CP}}` points in a *repulsive* direction.
- :math:`w` is the **repulsion weight** controlling how strongly tips avoid existing branches.
- :math:`\ell_b` is the **branch length** for the current generation.
- :math:`N_s` is the **number of segments per branch**; the step size is :math:`\ell_b/N_s`.

**Initial direction at bifurcation.** When a new branch is spawned, its first direction
:math:`\mathbf{d}^1` is obtained by rotating the parent’s last direction by
the **branching angle** :math:`\pm \alpha_b` within the local tangent plane.
For subsequent segments, use (:eq:`eq:dir_update`)–(:eq:`eq:pos_update`).

Parameter mapping (paper → library)
-----------------------------------

- :math:`\ell_b`  → ``FractalTreeParameters.length`` (median branch length).
- :math:`\alpha_b` → ``FractalTreeParameters.branch_angle`` (branching angle).
- :math:`w`       → ``FractalTreeParameters.w`` (repulsion weight).
- :math:`N_s`     → implicit via the **step length**:
  ``FractalTreeParameters.l_segment`` controls the step; typically
  :math:`N_s \approx \lceil \ell_b / \texttt{l\_segment} \rceil`.
- Segment spacing / collisions → ``Nodes.collision()``,
  with KD-tree updates via ``Nodes.update_collision_tree()``.

Implementation notes
--------------------

- The direction update is **renormalized** every step (unit vector).
- Multiple branches may grow “in parallel” (one step at each active tip) until the
  current generation reaches :math:`N_s` segments.
- The geometric update is performed in a **parameter space** (2D chart) and projected
  back to the endocardial surface; see :doc:`projection_surface`.

Figure
--------------

.. _fig-paper-2:

.. figure:: ../images/fig_paper_2.png
   :alt: Schematic of branch extension and bifurcation (paper Figure 2)
   :align: center
   :width: 70%

   Paper schematic of the growth process, showing the role of :math:`\ell_b`, :math:`\alpha_b`,
   and :math:`w`. See :cite:`AlvarezBarrientos2023` §2.1.

API cross-reference
-------------------

- :class:`purkinje_uv.fractal_tree_uv.FractalTree` – orchestrates growth (``grow_tree``),
  maintains ``edges``, ``connectivity``, ``end_nodes``.
- :class:`purkinje_uv.branch.Branch` – computes the next step and requests projection/collision checks.
- :class:`purkinje_uv.mesh.Mesh` – ``project_new_point``, ``gradient``, ``detect_boundary``,
  ``compute_uvscaling`` (used during growth).
- :class:`purkinje_uv.nodes.Nodes` – spacing control: ``collision``, ``update_collision_tree``.

Algorithm flow (generation loop)
--------------------------------

.. mermaid::
   :caption: Algorithm flow for the fractal-tree generation loop.
   :name: alg-flow

   flowchart TD
     A([Start]) --> B[FractalTree.grow_tree()]
     B --> C[Init nodes/edges and seeds]
     C --> D{More generations?}
     D -->|Yes| E{Active tips left?}
     D -->|No| Z[Finalize: build PurkinjeTree(nodes, connectivity, end_nodes)]

     E -->|Pick tip t| F[Branch.step()]
     F --> G[Update direction d_next = normalize(d + w * grad dCP)]
     G --> H[Propose 2D step: l_b / N_s]
     H --> I[Map 2D to 3D candidate]
     I --> J[Mesh.project_new_point(candidate) -> x_proj]
     J --> K{Nodes.collision(x_proj)?}

     K -->|Too close| L{Reduce step?}
     L -->|Yes| H
     L -->|No| M[Terminate tip; mark end node]
     M --> E

     K -->|OK| N[Append node and Edge(n_prev, new)]
     N --> O{Reached N_s segments this generation?}
     O -->|No| E
     O -->|Yes| P{Bifurcate here?}
     P -->|Yes| Q[Spawn child tips with +/- branch_angle]
     Q --> E
     P -->|No| E

     Z --> END([Done])


Single step: class interaction
------------------------------

.. mermaid::
   :caption: One growth step — who calls whom.
   :name: alg-seq

   sequenceDiagram
     participant FT as FractalTree
     participant BR as Branch
     participant ME as Mesh
     participant NO as Nodes

     FT->>BR: step()
     BR->>BR: compute d_next (w · grad d_CP)
     BR->>ME: project_new_point(candidate)
     ME-->>BR: x_proj
     BR->>NO: collision(x_proj)
     NO-->>BR: ok / too close
     alt ok
       BR-->>FT: append node & Edge(n_prev, new)
     else too close
       BR-->>FT: reduce step or terminate tip
     end

References
----------

- :cite:`AlvarezBarrientos2023` — Section 2.1 (*Purkinje network generation*), Figure 2.
