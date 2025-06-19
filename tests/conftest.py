from __future__ import annotations

import torch
from qermod import PrimitiveNoise, IndependentReadout, CorrelatedReadout
import pytest

@pytest.fixture(
    params=[
        IndependentReadout(error_definition=0.1),
        CorrelatedReadout(error_definition=torch.rand((4, 4))),
    ],
)
def readout_noise(
    request: pytest.Fixture,
) -> PrimitiveNoise:
    return request.param

