from __future__ import annotations

import pytest
import torch

from qermod import Noise, NoiseCategory, NoiseInstance


def test_noise_instance_model_validation() -> None:

    with pytest.raises(ValueError):
        NoiseInstance(protocol=NoiseCategory.READOUT.INDEPENDENT, error_rate=-0.1)


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
        Noise(configs=[NoiseInstance(protocol=NoiseCategory.READOUT.INDEPENDENT, error_rate=0.1)]),
        Noise(
            configs=[
                NoiseInstance(
                    protocol=NoiseCategory.READOUT.CORRELATED, error_rate=torch.rand((4, 4))
                )
            ]
        ),
    ],
)
def test_append(initial_noise: Noise, noise_config: list[NoiseCategory]) -> None:
    with pytest.raises(ValueError):
        initial_noise.append(
            Noise(configs=[NoiseInstance(protocol=c, error_rate=0.1) for c in noise_config])
        )
    with pytest.raises(ValueError):
        initial_noise.readout_independent(error_rate=0.1)

    with pytest.raises(ValueError):
        initial_noise.readout_correlated(error_rate=torch.rand(4, 4))
