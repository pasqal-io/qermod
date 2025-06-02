from __future__ import annotations

from itertools import compress
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from pyqtorch.noise.readout import WhiteNoise

from qermod.types import NoiseCategory, NoiseCategoryEnum

# to handle torch Tensor
BaseModel.model_config["arbitrary_types_allowed"] = True


class NoiseInstance(BaseModel):
    """A container for one source of noise.

    Args:
        protocol (NoiseCategoryEnum): Type of noise protocol.
        error_rate ()

    """

    protocol: NoiseCategoryEnum
    error_rate: float | list[float] | torch.Tensor
    seed: int | None = None
    noise_distribution: WhiteNoise | None = None
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_consistency(self) -> NoiseInstance:
        rates = [self.error_rate] if isinstance(self.error_rate, float) else self.error_rate
        if isinstance(rates, torch.Tensor):
            if (rates < 0).any() or (rates > 1.0).any():
                raise ValueError("`error_rate` can only be defined on [0,1]")
        else:
            for p in rates:
                if p < 0 or p > 1.0:
                    raise ValueError("`error_rate` can only be defined on [0,1]")
        if self.protocol not in NoiseCategory.READOUT.list():
            if self.noise_distribution is not None or self.seed is not None:
                raise ValueError(
                    "`noise_distribution` and `seed` can only be set for READOUT noise."
                )
        return self

    def __repr__(self) -> str:
        options: dict = {"error_rate": self.error_rate}
        if self.protocol in NoiseCategory.READOUT.list():
            options = options | {
                "noise_distribution": self.noise_distribution,
                "seed": self.seed,
                "confusion_matrix": self.confusion_matrix,
            }
        return f"NoiseInstance({self.protocol}, {str(options)})"


class Noise(BaseModel):
    """A container for multiple sources of noise.

    Note `NoiseCategory.ANALOG` and `NoiseCategory.DIGITAL` sources cannot be both present.
    Also `NoiseCategory.READOUT` can only be present once as the last noise sources, and only
    exclusively with `NoiseCategory.DIGITAL` sources.

    Args:
        configs: The config(s) applied. To be defined as a list of `NoiseInstance`.

    Examples:
    ```
        from qermod import NoiseCategory, Noise, NoiseInstance

        analog_options = {"error_probability": 0.1}
        digital_options = {"error_probability": 0.1}
        readout_options = {"error_probability": 0.1, "seed": 0}

        # single noise sources
        analog_noise = Noise(configs=[NoiseInstance(protocol=NoiseCategory.ANALOG.DEPOLARIZING,
            **analog_options)])
        digital_depo_noise = Noise(configs=[NoiseInstance(protocol=NoiseCategory.DIGITAL.DEPOLARIZING,
            **digital_options)])
        readout_noise = Noise(configs=[NoiseInstance(protocol=NoiseCategory.READOUT, **readout_options)])

        # init from multiple sources
        protocols: list = [NoiseCategory.DIGITAL.DEPOLARIZING, NoiseCategory.READOUT]
        options: list = [digital_options, readout_noise]
        noise_combination = Noise(configs=[NoiseInstance(protocol=p, **opts) for p, opts zip(protocols, options)])

        # Appending noise sources
        noise_combination = Noise(configs=[NoiseInstance(protocol=NoiseCategory.DIGITAL.BITFLIP, **digital_options)])
        noise_combination.append([digital_depo_noise, readout_noise])
    ```
    """

    configs: list[NoiseInstance]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def verify_all_protocols(self) -> Noise:
        """Make sure all protocols are correct in terms and their combination too."""

        types = [type(p.protocol) for p in self.configs]
        unique_types = set(types)
        if NoiseCategory.DIGITAL in unique_types and NoiseCategory.ANALOG in unique_types:
            raise ValueError("Cannot define a config with both Digital and Analog noises.")

        if NoiseCategory.ANALOG in unique_types:
            if NoiseCategory.READOUT in unique_types:
                raise ValueError("Cannot define a config with both READOUT and Analog noises.")
            if types.count(NoiseCategory.ANALOG) > 1:
                raise ValueError("Multiple Analog Noises are not supported yet.")

        if NoiseCategory.READOUT in unique_types:
            if (
                self.configs[-1].protocol not in NoiseCategory.READOUT.list()
                or types.count(NoiseCategory.READOUT) > 1
            ):
                raise ValueError("Only define a Noise with one READOUT as the last Noise.")
        return self

    def __repr__(self) -> str:
        return "\n".join([str(c) for c in self.configs])

    def append(self, other: Noise | list[Noise]) -> None:
        """Append noises.

        Args:
            other (Noise | list[Noise]): The noises to add.
        """
        # To avoid overwriting the noise_sources list if an error is raised, make a copy
        other_list = other if isinstance(other, list) else [other]
        configs = self.configs[:]

        for noise in other_list:
            configs += noise.configs

        # init may raise an error
        temp_handler = Noise(configs=configs)
        # if verify passes, replace configs
        self.configs = temp_handler.configs

    def filter(self, protocol: NoiseCategoryEnum) -> Noise | None:
        protocol_matches: list = [isinstance(c.protocol, protocol) for c in self.configs]  # type: ignore[arg-type]

        # if we have at least a match
        if True in protocol_matches:
            return Noise(
                configs=list(compress(self.configs, protocol_matches)),
            )
        return None

    def bitflip(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(configs=[NoiseInstance(protocol=NoiseCategory.DIGITAL.BITFLIP, *args, **kwargs)])
        )
        return self

    def phaseflip(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(
                configs=[NoiseInstance(protocol=NoiseCategory.DIGITAL.PHASEFLIP, *args, **kwargs)]
            )
        )
        return self

    def digital_depolarizing(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(
                configs=[
                    NoiseInstance(protocol=NoiseCategory.DIGITAL.DEPOLARIZING, *args, **kwargs)
                ]
            )
        )
        return self

    def pauli_channel(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(
                configs=[
                    NoiseInstance(protocol=NoiseCategory.DIGITAL.PAULI_CHANNEL, *args, **kwargs)
                ]
            )
        )
        return self

    def amplitude_damping(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(
                [NoiseInstance(protocol=NoiseCategory.DIGITAL.AMPLITUDE_DAMPING, *args, **kwargs)]
            )
        )
        return self

    def phase_damping(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(
                configs=[
                    NoiseInstance(protocol=NoiseCategory.DIGITAL.PHASE_DAMPING, *args, **kwargs)
                ]
            )
        )
        return self

    def generalized_amplitude_damping(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(
                [
                    NoiseInstance(
                        NoiseCategory.DIGITAL.GENERALIZED_AMPLITUDE_DAMPING, *args, **kwargs
                    )
                ]
            )
        )
        return self

    def analog_depolarizing(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(
                configs=[NoiseInstance(protocol=NoiseCategory.ANALOG.DEPOLARIZING, *args, **kwargs)]
            )
        )
        return self

    def dephasing(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(configs=[NoiseInstance(protocol=NoiseCategory.ANALOG.DEPHASING, *args, **kwargs)])
        )
        return self

    def readout_independent(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(
                configs=[NoiseInstance(protocol=NoiseCategory.READOUT.INDEPENDENT, *args, **kwargs)]
            )
        )
        return self

    def readout_correlated(self, *args: Any, **kwargs: Any) -> Noise:
        self.append(
            Noise(
                configs=[NoiseInstance(protocol=NoiseCategory.READOUT.CORRELATED, *args, **kwargs)]
            )
        )
        return self
