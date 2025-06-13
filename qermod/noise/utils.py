from __future__ import annotations

from logging import getLogger
from typing import Generator, List, Type, TypeVar, Union

from qermod.noise import (
    AbstractNoise,
    CompositeNoise,
    PrimitiveNoise,
)

logger = getLogger(__name__)

TPrimitiveNoise = TypeVar("TPrimitiveNoise", bound=PrimitiveNoise)
TCompositeNoise = TypeVar("TCompositeNoise", bound=CompositeNoise)


def _construct(
    Block: Type[TCompositeNoise],
    args: tuple[Union[AbstractNoise, Generator, List[AbstractNoise]], ...],
) -> TCompositeNoise:
    if len(args) == 1 and isinstance(args[0], Generator):
        args = tuple(args[0])
    return Block(blocks=(b for b in args))  # type: ignore [arg-type]


def chain(*args: Union[AbstractNoise, Generator, List[AbstractNoise]]) -> CompositeNoise:
    """Chain noise blocks sequentially.

    Arguments:
        *args: Noise blocks to chain. Can also be a generator.

    Returns:
        CompositeNoise
    """
    return _construct(CompositeNoise, args)
