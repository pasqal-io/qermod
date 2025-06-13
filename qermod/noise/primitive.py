from __future__ import annotations

from typing import Iterable

from qermod.noise.abstract import AbstractNoise
from qermod.types import NoiseCategoryEnum, ERROR_TYPE

class PrimitiveNoise(AbstractNoise):
    """
    Primitive noise represent elementary noise operations.
    """
    protocol: NoiseCategoryEnum
    error_definition: ERROR_TYPE

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterable:
        yield self
    
    def flatten(self) -> PrimitiveNoise:
        return self