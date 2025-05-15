from __future__ import annotations

import pytest
import torch

from qermod import NoiseHandler, NoiseProtocol


def test_serialization() -> None:
    noise = NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT)
    serialized_noise = NoiseHandler._from_dict(noise._to_dict())
    assert noise == serialized_noise

    rand_confusion = torch.rand(4, 4)
    rand_confusion = rand_confusion / rand_confusion.sum(dim=1, keepdim=True)
    noise = NoiseHandler(
        protocol=NoiseProtocol.READOUT.CORRELATED,
        options={"seed": 0, "confusion_matrix": rand_confusion},
    )
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
@pytest.mark.parametrize(
    "initial_noise",
    [
        NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT),
        NoiseHandler(protocol=NoiseProtocol.READOUT.CORRELATED, options=torch.rand((4, 4))),
    ],
)
def test_append(
    initial_noise: NoiseHandler, noise_config: NoiseProtocol | list[NoiseProtocol]
) -> None:
    options = {"error_probability": 0.1}
    with pytest.raises(ValueError):
        initial_noise.append(NoiseHandler(noise_config, options))
    with pytest.raises(ValueError):
        initial_noise.readout_independent(options)

    with pytest.raises(ValueError):
        initial_noise.readout_correlated({"confusion_matrix": torch.rand(4, 4)})
