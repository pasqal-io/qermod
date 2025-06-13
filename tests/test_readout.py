from __future__ import annotations

import pytest
import torch

from qermod import CorrelatedReadout, IndependentReadout, NoiseCategory, PrimitiveNoise


def test_noise_instance_model_validation() -> None:

    with pytest.raises(ValueError):
        IndependentReadout(error_definition=-0.1)


@pytest.mark.parametrize(
    "noise_config",
    [
        [NoiseCategory.READOUT.INDEPENDENT],
        [NoiseCategory.DIGITAL.BITFLIP],
        [NoiseCategory.DIGITAL.BITFLIP, NoiseCategory.DIGITAL.PHASEFLIP],
    ],
)
@pytest.mark.parametrize(
    "initial_noise",
    [
        IndependentReadout(error_definition=0.1),
        CorrelatedReadout(error_definition=torch.rand((4, 4))),
    ],
)
def test_append(initial_noise: PrimitiveNoise, noise_config: list[NoiseCategory]) -> None:
    for c in noise_config:
        with pytest.raises(ValueError):
            initial_noise + PrimitiveNoise(protocol=c, error_definition=0.1)
    with pytest.raises(ValueError):
        initial_noise + IndependentReadout(error_definition=0.1)

    with pytest.raises(ValueError):
        initial_noise + CorrelatedReadout(error_definition=torch.rand((4, 4)))
