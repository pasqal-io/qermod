from __future__ import annotations

from typing import Optional

from pydantic import Field
from pyqtorch.noise.readout import WhiteNoise

from qermod.noise import PrimitiveNoise
from qermod.types import NoiseCategory, NoiseCategoryEnum


class Bitflip(PrimitiveNoise):
    """The Bitflip noise."""

    protocol: NoiseCategoryEnum = Field(default=NoiseCategory.DIGITAL.BITFLIP, frozen=True)


class Phaseflip(PrimitiveNoise):
    """The Phaseflip noise."""

    protocol: NoiseCategoryEnum = Field(default=NoiseCategory.DIGITAL.PHASEFLIP, frozen=True)


class PauliChannel(PrimitiveNoise):
    """The PauliChannel noise."""

    protocol: NoiseCategoryEnum = Field(default=NoiseCategory.DIGITAL.PAULI_CHANNEL, frozen=True)


class AmplitudeDamping(PrimitiveNoise):
    """The AmplitudeDamping noise."""

    protocol: NoiseCategoryEnum = Field(
        default=NoiseCategory.DIGITAL.AMPLITUDE_DAMPING, frozen=True
    )


class PhaseDamping(PrimitiveNoise):
    """The PhaseDamping noise."""

    protocol: NoiseCategoryEnum = Field(default=NoiseCategory.DIGITAL.PHASE_DAMPING, frozen=True)


class DigitalDepolarizing(PrimitiveNoise):
    """The DigitalDepolarizing noise."""

    protocol: NoiseCategoryEnum = Field(default=NoiseCategory.DIGITAL.DEPOLARIZING, frozen=True)


class GeneralizedAmplitudeDamping(PrimitiveNoise):
    """The GeneralizedAmplitudeDamping noise."""

    protocol: NoiseCategoryEnum = Field(
        default=NoiseCategory.DIGITAL.GENERALIZED_AMPLITUDE_DAMPING, frozen=True
    )


class AnalogDepolarizing(PrimitiveNoise):
    """The AnalogDepolarizing noise."""

    protocol: NoiseCategoryEnum = Field(default=NoiseCategory.ANALOG.DEPOLARIZING, frozen=True)


class Dephasing(PrimitiveNoise):
    """The Dephasing noise."""

    protocol: NoiseCategoryEnum = Field(default=NoiseCategory.ANALOG.DEPHASING, frozen=True)


class IndependentReadout(PrimitiveNoise):
    """The IndependentReadout noise.

    Note we can pass a confusion matrix via the `error_definition` argument.
    """

    protocol: NoiseCategoryEnum = Field(default=NoiseCategory.READOUT.INDEPENDENT, frozen=True)
    seed: Optional[int] = None
    noise_distribution: WhiteNoise | None = None


class CorrelatedReadout(PrimitiveNoise):
    """The CorrelatedReadout noise.

    Note a confusion matrix should be passed via the `error_definition` argument.
    """

    protocol: NoiseCategoryEnum = Field(default=NoiseCategory.READOUT.CORRELATED, frozen=True)
    seed: Optional[int] = None
