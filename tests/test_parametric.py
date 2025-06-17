from __future__ import annotations

from qadence.parameters import Parameter

from qermod import Bitflip, IndependentReadout


def test_parametric_digital() -> None:

    noise = Bitflip(error_definition=Parameter("p"))
    assert not noise.error_definition.is_number  # type: ignore[union-attr]

    noise = Bitflip(error_definition=Parameter(0.1))
    assert noise.error_definition == 0.1

    noise = Bitflip(error_definition="p")
    assert isinstance(noise.error_definition, Parameter)


def test_parametric_readout() -> None:

    noise = IndependentReadout(error_definition=Parameter("p"))
    assert not noise.error_definition.is_number  # type: ignore[union-attr]

    noise = IndependentReadout(error_definition=Parameter(0.1))
    assert noise.error_definition == 0.1

    noise = IndependentReadout(error_definition="p")
    assert isinstance(noise.error_definition, Parameter)
