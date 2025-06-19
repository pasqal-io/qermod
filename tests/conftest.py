from __future__ import annotations

from typing import cast

import pytest
import torch

from qermod import CorrelatedReadout, IndependentReadout, PrimitiveNoise


@pytest.fixture(
    params=[
        IndependentReadout(error_definition=0.1),
        CorrelatedReadout(error_definition=torch.rand((4, 4))),
    ],
)
def readout_noise(
    request: pytest.Fixture,
) -> PrimitiveNoise:
    return cast(PrimitiveNoise, request.param)
