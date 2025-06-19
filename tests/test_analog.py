from __future__ import annotations

import pytest

from qermod import (
    AnalogDepolarizing,
    Dephasing,
    DigitalDepolarizing,
    PrimitiveNoise,
    deserialize,
    serialize,
)


def test_serialization(analog_noise: PrimitiveNoise) -> None:
    assert analog_noise == deserialize(serialize(analog_noise))


def test_noise_instance_model_validation() -> None:

    with pytest.raises(ValueError):
        Dephasing(error_definition=0.1, seed=0)

    with pytest.raises(ValueError):
        Dephasing(error_definition=-0.1)


def test_error_append() -> None:
    noise = Dephasing(error_definition=0.1)
    with pytest.raises(ValueError):

        noise | AnalogDepolarizing(error_definition=0.1)
    with pytest.raises(ValueError):
        noise | DigitalDepolarizing(error_definition=0.1)


def test_equality() -> None:
    noise = Dephasing(error_definition=0.1)
    noise2 = AnalogDepolarizing(error_definition=0.1)

    assert noise != noise2
    assert noise == Dephasing(error_definition=0.1)
