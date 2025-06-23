from __future__ import annotations

import pytest
from qadence import X, Y

from qermod import (
    AnalogDepolarizing,
    Bitflip,
    Noise,
    PrimitiveNoise,
    deserialize,
    serialize,
)

digital_noises = Noise.DIGITAL.list()


def test_initialization_target_gates() -> None:
    noise = Bitflip(error_definition=0.1)
    assert not noise.target

    noise = Bitflip(error_definition=0.1, target=0)
    assert noise.target == 0

    targets = [0, 1]
    noise = Bitflip(error_definition=0.1, target=targets)
    for i in range(2):
        assert noise.target[i] == targets[i]  # type:ignore[index]

    noise = Bitflip(error_definition=0.1, target=X)
    assert noise.target == X

    noise = Bitflip(error_definition=0.1, target=Y(0))
    assert noise.target == Y(0)

    targets = [X, Y(0)]
    noise = Bitflip(error_definition=0.1, target=targets)
    for i in range(2):
        assert noise.target[i] == targets[i]  # type:ignore[index]


@pytest.mark.parametrize("noise_config", digital_noises)
def test_serialization(noise_config: Noise.DIGITAL) -> None:
    noise = PrimitiveNoise(protocol=noise_config, error_definition=0.1)
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
        [Noise.READOUT.INDEPENDENT],
        [Noise.DIGITAL.BITFLIP],
        [Noise.DIGITAL.BITFLIP, Noise.DIGITAL.PHASEFLIP],
    ],
)
def test_append(noise_config: list[Noise]) -> None:
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
