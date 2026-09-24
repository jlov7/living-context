from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from living_context.replacement_study import (
    assert_prefix_only_training,
    cross_entropy_from_logits,
    target_prediction_slice,
)


@pytest.mark.mlx
def test_mlx_cross_entropy_uses_the_independently_computed_causal_slice() -> None:
    prefix_length = 3
    sequence_length = 5
    target_start = 2
    # Correct positions are 4, 5, 6. Adjacent positions deliberately favor
    # wrong classes, so an off-by-one target shift cannot pass this check.
    logits = mx.array(
        [
            [[9.0, 0.0, 0.0]] * 4
            + [[0.0, 0.0, 9.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0]]
            + [[0.0, 0.0, 9.0]]
        ]
    )
    vectors = [[0.0, 0.0, 9.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0]]
    prediction_slice = target_prediction_slice(prefix_length, sequence_length, target_start)
    assert prediction_slice == slice(4, 7)
    targets = mx.array([[2, 0, 1]])
    actual = nn.losses.cross_entropy(
        logits[:, prediction_slice, :], targets, reduction="mean"
    ).item()
    expected = (
        sum(
            cross_entropy_from_logits(vector, target)
            for vector, target in zip(vectors, [2, 0, 1], strict=True)
        )
        / 3
    )
    assert actual == pytest.approx(expected, abs=1e-6)
    shifted = nn.losses.cross_entropy(logits[:, 3:6, :], targets, reduction="mean").item()
    assert shifted > actual + 2.0


@pytest.mark.mlx
def test_gradient_surface_contains_only_prefix_weights_after_base_freeze() -> None:
    from mlx.utils import tree_flatten

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = mx.array([2.0])

    class Prefix(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weights = mx.array([0.5])

    base = Base()
    base.freeze()
    prefix = Prefix()

    def loss_fn():
        return mx.sum((prefix.weights - 1.0) ** 2) + 0.0 * mx.sum(base.weight)

    _loss, gradients = nn.value_and_grad(prefix, loss_fn)()
    base_names = sorted(name for name, _ in tree_flatten(base.trainable_parameters()))
    gradient_names = sorted(name for name, _ in tree_flatten(gradients))
    assert_prefix_only_training(base_names, gradient_names)
