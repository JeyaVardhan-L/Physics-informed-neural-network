import torch

from pinn.network import MLP, column


def test_parameter_count():
    # 1*32+32, then 32*32+32 twice, then 32*1+1
    assert MLP(width=32, depth=3).n_parameters() == 64 + 1056 + 1056 + 33


def test_same_seed_same_weights():
    t = column([0.0, 0.5, 1.0])
    assert torch.equal(MLP(seed=7)(t), MLP(seed=7)(t))
    assert not torch.equal(MLP(seed=7)(t), MLP(seed=8)(t))


def test_building_a_net_leaves_the_global_seed_alone():
    torch.manual_seed(0)
    expected = torch.rand(1)
    torch.manual_seed(0)
    MLP(seed=5)
    assert torch.equal(torch.rand(1), expected)


def test_double_precision():
    assert MLP()(column([0.1])).dtype == torch.float64
