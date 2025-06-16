from __future__ import annotations

import pytest

from qermod import Bitflip
from qadence.parameters import Parameter

def test_parametric() -> None:

    noise = Bitflip(error_definition=Parameter('p'))
    assert not noise.error_definition.is_number

    noise = Bitflip(error_definition=Parameter(0.1))
    assert noise.error_definition.is_number