from __future__ import annotations

import pytest

from qermod import NoiseHandler, NoiseProtocol

list_noises = [noise for noise in NoiseProtocol.DIGITAL]


def test_serialization() -> None:
    noise = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, {"error_probability": 0.2})
    serialized_noise = NoiseHandler._from_dict(noise._to_dict())
    assert noise == serialized_noise


@pytest.mark.parametrize(
    "noise_config",
    [
        NoiseProtocol.READOUT,
        NoiseProtocol.DIGITAL.BITFLIP,
        [NoiseProtocol.DIGITAL.BITFLIP, NoiseProtocol.DIGITAL.PHASEFLIP],
    ],
)
def test_append(noise_config: NoiseProtocol | list[NoiseProtocol]) -> None:
    options = {"error_probability": 0.1}
    noise = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, options)

    len_noise_config = len(noise_config) if isinstance(noise_config, list) else 1
    noise.append(NoiseHandler(noise_config, options))

    assert len(noise.protocol) == (len_noise_config + 1)


def test_equality() -> None:
    options = {"error_probability": 0.1}
    noise = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, options)
    noise.append(NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, options))

    noise2 = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, options)
    noise2.bitflip(options)

    assert noise == noise2
