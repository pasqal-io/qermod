from __future__ import annotations

import pytest
import torch

from qermod import (
    CorrelatedReadout,
    IndependentReadout,
    Noise,
    PrimitiveNoise,
    deserialize,
    serialize,
)


def test_noise_instance_model_validation() -> None:

    with pytest.raises(ValueError):
        IndependentReadout(error_definition=-0.1)


@pytest.mark.parametrize(
    "initial_noise",
    [
        IndependentReadout(error_definition=0.1),
        CorrelatedReadout(error_definition=torch.rand((4, 4))),
    ],
)
def test_serialization(initial_noise: PrimitiveNoise) -> None:
    assert initial_noise == deserialize(serialize(initial_noise))


@pytest.mark.parametrize(
    "noise_config",
    [
        [Noise.READOUT.INDEPENDENT],
        [Noise.DIGITAL.BITFLIP],
        [Noise.DIGITAL.BITFLIP, Noise.DIGITAL.PHASEFLIP],
    ],
)
@pytest.mark.parametrize(
    "initial_noise",
    [
        IndependentReadout(error_definition=0.1),
        CorrelatedReadout(error_definition=torch.rand((4, 4))),
    ],
)
def test_append(initial_noise: PrimitiveNoise, noise_config: list[Noise]) -> None:
    for c in noise_config:
        with pytest.raises(ValueError):
            initial_noise | PrimitiveNoise(protocol=c, error_definition=0.1)
    with pytest.raises(ValueError):
        initial_noise | IndependentReadout(error_definition=0.1)

    with pytest.raises(ValueError):
        initial_noise | CorrelatedReadout(error_definition=torch.rand((4, 4)))
