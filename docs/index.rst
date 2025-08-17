Welcome to the purkinje-uv documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   welcome
   dev/architecture
   theory/index
   usage/index
   seeding
   dev/index
   references
   api/modules

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

Introduction
------------

``purkinje-uv`` is a Python library for generating Purkinje networks using fractal trees mapped onto cardiac mesh surfaces.
It includes geometric tools, network generation logic, and activation solvers.

Mermaid Example
---------------

.. mermaid::

   graph TD
       Start --> Mesh[Create Mesh]
       Mesh --> Tree[Generate FractalTree]
       Tree --> Network[Build PurkinjeTree]
       Network --> Sim[Run Activation Solver]
