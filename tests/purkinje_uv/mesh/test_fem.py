"""Unit tests for FEM-related methods in Mesh class."""

import numpy as np
import scipy.sparse as sp


def test_bmatrix_and_jacobian(simple_triangle_mesh):
    mesh = simple_triangle_mesh
    B, J = mesh.Bmatrix(0)

    assert B.shape == (2, 3)
    assert isinstance(J, float)
    assert J > 0


def test_stiffness_matrix(simple_triangle_mesh):
    mesh = simple_triangle_mesh
    B, J = mesh.Bmatrix(0)
    K = mesh.StiffnessMatrix(B, J)

    assert K.shape == (3, 3)
    assert np.allclose(K, K.T, atol=1e-8)  # Should be symmetric

    eigvals = np.linalg.eigvals(K)
    assert np.all(eigvals >= -1e-12)  # Allow small numerical noise


def test_mass_matrix(simple_triangle_mesh):
    mesh = simple_triangle_mesh
    _, J = mesh.Bmatrix(0)
    M = mesh.MassMatrix(J)

    assert M.shape == (3, 3)
    assert np.allclose(M, M.T, atol=1e-8)  # Should be symmetric
    assert np.all(np.linalg.eigvals(M) >= 0)  # Positive semi-definite


def test_gradient(simple_triangle_mesh):
    mesh = simple_triangle_mesh
    u = np.array([1.0, 2.0, 3.0])  # Linear field
    grad = mesh.gradient(0, u)

    assert grad.shape == (3,)
    assert not np.allclose(grad, 0.0)


def test_force_vector(simple_triangle_mesh):
    mesh = simple_triangle_mesh
    B, J = mesh.Bmatrix(0)

    X = np.array([1.0, 1.0])
    f = mesh.ForceVector(B, J, X)

    assert f.shape == (3,)
    assert not np.allclose(f, 0.0)


def test_compute_geodesic(simple_triangle_mesh):
    """
    Test that `computeGeodesic` returns valid geodesic distances and gradients
    given fixed boundary conditions.

    This uses a minimal mesh with one fixed node and simulates a heat-based geodesic solve.
    """
    mesh = simple_triangle_mesh

    nodes = [0]  # Fix one node
    nodeVals = [0.0]  # Geodesic distance = 0 at that node

    distances, gradients = mesh.computeGeodesic(nodes=nodes, nodeVals=nodeVals)

    assert distances.shape == (3,)
    assert gradients.shape == (1, 3)  # one triangle → one gradient vector
    assert np.isclose(distances[0], 0.0)
    assert np.all(np.isfinite(distances))
    assert np.all(np.isfinite(gradients))


def test_compute_laplace(simple_triangle_mesh):
    """
    Test that `computeLaplace` returns a valid scalar field with correct Dirichlet conditions.

    Fixes values at a node and verifies the output is finite and matches at boundaries.
    """
    mesh = simple_triangle_mesh

    nodes = [0]  # Fix node 0
    nodeVals = [42.0]  # Arbitrary scalar value

    result = mesh.computeLaplace(nodes=nodes, nodeVals=nodeVals)

    assert result.shape == (3,)
    assert np.isclose(result[0], 42.0)
    assert np.all(np.isfinite(result))


def test_compute_laplacian_assembly(simple_triangle_mesh):
    """
    Test that `computeLaplacian` returns correctly shaped stiffness and mass matrices.

    This verifies that the method assembles the global FEM matrices for the mesh.
    """
    mesh = simple_triangle_mesh

    K, M = mesh.computeLaplacian()

    n_nodes = mesh.verts.shape[0]

    # Ensure matrices are square and of correct size
    assert K.shape == (n_nodes, n_nodes)
    assert M.shape == (n_nodes, n_nodes)

    # Ensure they are sparse matrices
    assert sp.issparse(K)
    assert sp.issparse(M)

    # Sanity checks for non-trivial entries
    assert K.nnz > 0
    assert M.nnz > 0


def test_uv_map_generation(simple_triangle_mesh):
    """
    Verify that uvmap() produces valid UV coordinates when uv_bc() returns a
    closed boundary loop but BC arrays are provided for the UNIQUE boundary
    nodes (i.e., len(around_nodes) - 1).
    """
    mesh = simple_triangle_mesh

    # Explicitly define the boundary for the single triangle: 0→1→2→0
    mesh.boundary_edges = [(0, 1), (1, 2), (2, 0)]

    # Monkey-patch uv_bc to return a closed loop in around_nodes
    def fake_uv_bc():
        around_nodes = [0, 1, 2, 0]  # closed loop
        bc_u = np.array([0.0, 1.0, 0.5], dtype=float)  # len == 3
        bc_v = np.array([0.0, 0.0, 1.0], dtype=float)  # len == 3
        return around_nodes, bc_u, bc_v

    mesh.uv_bc = fake_uv_bc  # Override with our well-formed BC provider

    # Execute UV mapping
    mesh.uvmap()

    # Basic shape/finite checks
    assert hasattr(mesh, "uv"), "mesh.uv was not created by uvmap()"
    assert mesh.uv.shape == (3, 2)
    assert np.isfinite(mesh.uv).all()

    # Since all 3 nodes are boundary nodes and we provided Dirichlet BCs for each,
    # the UVs at those nodes should match our BCs exactly.
    expected_uv = np.array(
        [
            [0.0, 0.0],  # node 0
            [1.0, 0.0],  # node 1
            [0.5, 1.0],  # node 2
        ],
        dtype=float,
    )
    np.testing.assert_allclose(mesh.uv[[0, 1, 2]], expected_uv, rtol=1e-7, atol=1e-8)

    # If uv scaling is produced, it should be finite and non-negative
    if hasattr(mesh, "uvscaling") and mesh.uvscaling is not None:
        assert np.all(np.isfinite(mesh.uvscaling))
        assert np.all(mesh.uvscaling >= 0.0)
