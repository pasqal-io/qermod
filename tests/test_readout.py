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


def test_serialization(readout_noise: PrimitiveNoise) -> None:
    assert readout_noise == deserialize(serialize(readout_noise))


def test_filter(readout_noise: PrimitiveNoise) -> None:
    assert readout_noise.filter(Noise.READOUT)
    assert not readout_noise.filter(Noise.ANALOG)
    assert not readout_noise.filter(Noise.DIGITAL)

    assert readout_noise.filter(readout_noise.protocol)


@pytest.mark.parametrize(
    "noise_config",
    [
        [Noise.READOUT.INDEPENDENT],
        [Noise.DIGITAL.BITFLIP],
        [Noise.DIGITAL.BITFLIP, Noise.DIGITAL.PHASEFLIP],
    ],
)
def test_append(readout_noise: PrimitiveNoise, noise_config: list[Noise]) -> None:
    for c in noise_config:
        with pytest.raises(ValueError):
            readout_noise | PrimitiveNoise(protocol=c, error_definition=0.1)
    with pytest.raises(ValueError):
        readout_noise | IndependentReadout(error_definition=0.1)

    with pytest.raises(ValueError):
        readout_noise | CorrelatedReadout(error_definition=torch.rand((4, 4)))
