from purkinje_uv.fractal_tree_parameters import Parameters

def test_parameters_defaults():
    params = Parameters()

    assert params.meshfile is None
    assert params.init_node_id == 0
    assert params.second_node_id == 1
    assert params.init_length == 0.1
    assert params.N_it == 10
    assert params.length == 0.1
    assert params.branch_angle == 0.15
    assert params.w == 0.1
    assert params.l_segment == 0.01

    assert isinstance(params.fascicles_angles, list)
    assert isinstance(params.fascicles_length, list)
    assert len(params.fascicles_angles) == 0
    assert len(params.fascicles_length) == 0

def test_parameters_mutability():
    params = Parameters()

    params.meshfile = "sample.obj"
    params.init_node_id = 5
    params.fascicles_angles = [0.1, 0.2]
    params.fascicles_length = [0.03, 0.04]

    assert params.meshfile == "sample.obj"
    assert params.init_node_id == 5
    assert params.fascicles_angles == [0.1, 0.2]
    assert params.fascicles_length == [0.03, 0.04]
