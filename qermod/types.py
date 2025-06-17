from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Union

from pydantic import Field
from pyqtorch.noise import DigitalNoiseType as DigitalNoise
from qadence.parameters import Parameter
from qadence.types import TParameter
from qadence_commons import StrEnum

probaType = Annotated[float, Field(strict=True, gt=0, lt=1.0)]


class AnalogNoise(StrEnum):
    """Type of noise protocol."""

    DEPOLARIZING = "Depolarizing"
    DEPHASING = "Dephasing"


class ReadoutNoise(StrEnum):
    """Type of readout protocol."""

    INDEPENDENT = "Independent Readout"
    """Simple readout protocols where each qubit is corrupted independently."""
    CORRELATED = "Correlated Readout"
    """Using a confusion matrix (2**n, 2**n) for corrupting bitstrings values."""


@dataclass
class NoiseCategory:
    """Type of noise protocol."""

    ANALOG = AnalogNoise
    """Noise applied in analog blocks."""
    READOUT = ReadoutNoise
    """Noise applied on outputs of quantum programs."""
    DIGITAL = DigitalNoise
    """Noise applied to digital blocks."""


NoiseCategoryEnum = Union[DigitalNoise, AnalogNoise, ReadoutNoise]
ERROR_TYPE = Union[Parameter, TParameter]
