from __future__ import annotations

import pytest

from qermod import Noise, NoiseCategory, NoiseInstance

list_noises = [noise for noise in NoiseCategory.DIGITAL]


def test_noise_instance_model_validation() -> None:

    with pytest.raises(ValueError):
        NoiseInstance(protocol=NoiseCategory.DIGITAL.BITFLIP, error_rate=0.1, seed=0)

    with pytest.raises(ValueError):
        NoiseInstance(protocol=NoiseCategory.DIGITAL.BITFLIP, error_rate=-0.1)


@pytest.mark.parametrize(
    "noise_config",
    [
        [NoiseCategory.READOUT.INDEPENDENT],
        [NoiseCategory.DIGITAL.BITFLIP],
        [NoiseCategory.DIGITAL.BITFLIP, NoiseCategory.DIGITAL.PHASEFLIP],
    ],
)
def test_append(noise_config: list[NoiseCategory]) -> None:
    noise = Noise(configs=[NoiseInstance(protocol=NoiseCategory.DIGITAL.BITFLIP, error_rate=0.1)])

    len_noise_config = len(noise_config)
    noise.append(Noise(configs=[NoiseInstance(protocol=p, error_rate=0.1) for p in noise_config]))

    assert len(noise.configs) == (len_noise_config + 1)


def test_equality() -> None:
    noise = Noise(configs=[NoiseInstance(protocol=NoiseCategory.DIGITAL.BITFLIP, error_rate=0.1)])
    noise.append(
        Noise(configs=[NoiseInstance(protocol=NoiseCategory.DIGITAL.BITFLIP, error_rate=0.1)])
    )

    noise2 = Noise(configs=[NoiseInstance(protocol=NoiseCategory.DIGITAL.BITFLIP, error_rate=0.1)])
    noise2.bitflip(error_rate=0.1)

    assert noise == noise2
