Data Model Invariants
=====================

Why invariants?
---------------

They ensure a valid, simulation-ready Purkinje graph: geometry is on the surface, indices are valid,
and the connectivity is a tree without self-loops.

Checklist
---------

- **On-surface nodes**: every node lies on the endocardial surface within tolerance.
- **Valid indices**: all connectivity indices are in ``[0, len(nodes)-1]``.
- **No self-edges**: no edge connects a node to itself.
- **No duplicate edges**: undirected edges appear once (or consistently ordered).
- **Acyclic**: the graph is a tree (``|E| = |V| - C`` where C is number of components).
- **Terminals**: all end nodes in ``end_nodes`` have degree = 1.
- **No isolated nodes**: every node appears in at least one edge (except permissible root cases).
- **Finite geometry**: node coordinates are finite (no NaN/Inf).
- **PMJs** (if present): are a subset of graph nodes.

Minimal validator (example)
---------------------------

.. code-block:: python

   import math

   def validate_tree(nodes, connectivity, end_nodes):
       n = len(nodes)
       # Indices
       for (u, v) in connectivity:
           assert 0 <= u < n and 0 <= v < n, "out-of-range index"
           assert u != v, "self-edge found"
       # Duplicates (undirected)
       seen = set()
       for (u, v) in connectivity:
           e = (u, v) if u < v else (v, u)
           assert e not in seen, "duplicate edge"
           seen.add(e)
       # Degrees
       deg = [0] * n
       for (u, v) in connectivity:
           deg[u] += 1; deg[v] += 1
       for k in end_nodes:
           assert deg[k] == 1, f"endpoint degree != 1 at {k}"
       # Components and acyclicity via union-find
       parent = list(range(n))
       def find(x):
           while x != parent[x]:
               parent[x] = parent[parent[x]]
               x = parent[x]
           return x
       def unite(a, b):
           ra, rb = find(a), find(b)
           if ra == rb:
               return False
           parent[rb] = ra; return True
       merges = 0
       for (u, v) in seen:
           merges += 1 if unite(u, v) else 0
       components = len({find(i) for i in range(n) if deg[i] > 0})
       assert len(seen) == sum(deg[i] > 0 for i in range(n)) - components, "not a forest"
       # Geometry finite
       for x in nodes:
           assert all(math.isfinite(float(c)) for c in x), "non-finite coordinate"

Operational tips
----------------

- Validate after growth and before activation/export.
- If you reindex or merge nodes, re-validate.
- Keep tolerances centralized (distance thresholds, collision spacing).
