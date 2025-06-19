from __future__ import annotations

from typing import Optional

from pydantic import Field
from pyqtorch.noise.readout import WhiteNoise

from qermod.noise import PrimitiveNoise
from qermod.types import Noise, NoiseEnum


class Bitflip(PrimitiveNoise):
    """The Bitflip noise."""

    protocol: NoiseEnum = Field(default=Noise.DIGITAL.BITFLIP, frozen=True)


class Phaseflip(PrimitiveNoise):
    """The Phaseflip noise."""

    protocol: NoiseEnum = Field(default=Noise.DIGITAL.PHASEFLIP, frozen=True)


class PauliChannel(PrimitiveNoise):
    """The PauliChannel noise."""

    protocol: NoiseEnum = Field(default=Noise.DIGITAL.PAULI_CHANNEL, frozen=True)


class AmplitudeDamping(PrimitiveNoise):
    """The AmplitudeDamping noise."""

    protocol: NoiseEnum = Field(default=Noise.DIGITAL.AMPLITUDE_DAMPING, frozen=True)


class PhaseDamping(PrimitiveNoise):
    """The PhaseDamping noise."""

    protocol: NoiseEnum = Field(default=Noise.DIGITAL.PHASE_DAMPING, frozen=True)


class DigitalDepolarizing(PrimitiveNoise):
    """The DigitalDepolarizing noise."""

    protocol: NoiseEnum = Field(default=Noise.DIGITAL.DEPOLARIZING, frozen=True)


class GeneralizedAmplitudeDamping(PrimitiveNoise):
    """The GeneralizedAmplitudeDamping noise."""

    protocol: NoiseEnum = Field(default=Noise.DIGITAL.GENERALIZED_AMPLITUDE_DAMPING, frozen=True)


class AnalogDepolarizing(PrimitiveNoise):
    """The AnalogDepolarizing noise."""

    protocol: NoiseEnum = Field(default=Noise.ANALOG.DEPOLARIZING, frozen=True)


class Dephasing(PrimitiveNoise):
    """The Dephasing noise."""

    protocol: NoiseEnum = Field(default=Noise.ANALOG.DEPHASING, frozen=True)


class IndependentReadout(PrimitiveNoise):
    """The IndependentReadout noise.

    Note we can pass a confusion matrix via the `error_definition` argument.
    """

    protocol: NoiseEnum = Field(default=Noise.READOUT.INDEPENDENT, frozen=True)
    seed: int | None = None
    noise_distribution: Optional[WhiteNoise] = None


class CorrelatedReadout(PrimitiveNoise):
    """The CorrelatedReadout noise.

    Note a confusion matrix should be passed via the `error_definition` argument.
    """

    protocol: NoiseEnum = Field(default=Noise.READOUT.CORRELATED, frozen=True)
    seed: Optional[int] = None
