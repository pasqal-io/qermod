from __future__ import annotations

import pytest

from qermod import (
    AnalogDepolarizing,
    Bitflip,
    NoiseCategory,
    PrimitiveNoise,
    deserialize,
    serialize,
)

list_noises = [noise for noise in NoiseCategory.DIGITAL]


def test_serialization() -> None:
    noise = Bitflip(error_definition=0.1)
    noise_serial = deserialize(serialize(noise))

    assert noise == noise_serial

    noise = PrimitiveNoise(protocol=NoiseCategory.DIGITAL.BITFLIP, error_definition=0.1)
    noise_serial = deserialize(serialize(noise))

    assert noise == noise_serial


def test_noise_instance_model_validation() -> None:

    with pytest.raises(ValueError):
        Bitflip(error_definition=0.1, seed=0)

    with pytest.raises(ValueError):
        Bitflip(error_definition=-0.1)


@pytest.mark.parametrize(
    "noise_config",
    [
        [NoiseCategory.READOUT.INDEPENDENT],
        [NoiseCategory.DIGITAL.BITFLIP],
        [NoiseCategory.DIGITAL.BITFLIP, NoiseCategory.DIGITAL.PHASEFLIP],
    ],
)
def test_append(noise_config: list[NoiseCategory]) -> None:
    noise = Bitflip(error_definition=0.1)

    len_noise_config = len(noise_config)
    for p in noise_config:
        noise |= PrimitiveNoise(protocol=p, error_definition=0.1)

    assert len(noise) == (len_noise_config + 1)

    with pytest.raises(ValueError):
        noise | AnalogDepolarizing(error_definition=0.1)


def test_equality() -> None:
    noise = Bitflip(error_definition=0.1)
    noise |= Bitflip(error_definition=0.1)

    noise2 = Bitflip(error_definition=0.1) | Bitflip(error_definition=0.1)

    assert noise == noise2
