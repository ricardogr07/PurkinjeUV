Performance
===========

Hot spots
---------

- **Nearest-triangle search / projection**: use a persistent cell locator.
- **Collision checks**: batch inserts and rebuild the KD-tree periodically.
- **Step size**: large steps cause retries; too small steps waste iterations.

Practical tips
--------------

- **Reuse locators**: create the VTK cell locator once per mesh, reuse in all steps.
- **Batch updates**: call ``Nodes.update_collision_tree()`` after a batch of nodes, not every node.
- **Tune step**: choose ``l_segment`` so that ``length / l_segment`` is ~10–30 segments.
- **Avoid tiny angles**: very small ``branch_angle`` increases projection/collision retries.
- **Vectorize**: where possible, operate on arrays; avoid Python loops in tight paths.

Scaling out
-----------

- Separate runs with different seeds/params across processes.
- Persist intermediate outputs (nodes/connectivity) if activation is separate.

Diagnostics
-----------

- Log: number of retries due to collisions, number of branch terminations, average segments per branch.
- Track: max nearest-node distance over time (surface coverage proxy).
