from __future__ import annotations

from typing import Iterable, List, Union

from pydantic import field_validator
from qadence import AbstractBlock
from qadence.parameters import Parameter

from qermod.noise.abstract import AbstractNoise
from qermod.types import ERROR_TYPE, NoiseCategoryEnum


class PrimitiveNoise(AbstractNoise):
    """
    Primitive noise represent elementary noise operations.
    """

    protocol: NoiseCategoryEnum
    error_definition: ERROR_TYPE
    target_gates: List[Union[AbstractBlock, type[AbstractBlock]]] = list()

    @field_validator("error_definition", mode="before")
    @classmethod
    def _normalize_error_definition(cls, val: ERROR_TYPE) -> Parameter:
        param = val if isinstance(val, Parameter) else Parameter(val)
        if param.is_number:
            if param < 0 or param > 1:
                raise ValueError("`error_definition` should be bound between 0 and 1")
        return param

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterable:
        yield self

    def flatten(self) -> PrimitiveNoise:
        return self
