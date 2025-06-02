from __future__ import annotations

import pytest

from qermod import Noise, NoiseCategory, NoiseInstance


def test_noise_instance_model_validation() -> None:

    with pytest.raises(ValueError):
        NoiseInstance(protocol=NoiseCategory.ANALOG.DEPHASING, error_rate=0.1, seed=0)

    with pytest.raises(ValueError):
        NoiseInstance(protocol=NoiseCategory.ANALOG.DEPHASING, error_rate=-0.1)


def test_error_append() -> None:
    noise = Noise(configs=[NoiseInstance(protocol=NoiseCategory.ANALOG.DEPHASING, error_rate=0.1)])
    with pytest.raises(ValueError):

        noise.append(
            Noise(
                configs=[NoiseInstance(protocol=NoiseCategory.ANALOG.DEPOLARIZING, error_rate=0.1)]
            )
        )
    with pytest.raises(ValueError):
        noise.digital_depolarizing(error_rate=0.1)


def test_equality() -> None:
    noise = Noise(configs=[NoiseInstance(protocol=NoiseCategory.ANALOG.DEPHASING, error_rate=0.1)])

    noise2 = Noise(
        configs=[NoiseInstance(protocol=NoiseCategory.ANALOG.DEPOLARIZING, error_rate=0.1)]
    )

    assert noise != noise2
