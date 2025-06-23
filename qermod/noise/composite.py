from __future__ import annotations

from pydantic import field_serializer, model_validator

from qermod.types import Noise, NoiseType

from .abstract import AbstractNoise


class CompositeNoise(AbstractNoise):
    """Noise which composes multiple instance into one larger noise (which can again be composed).

    Composite blocks are constructed via [`chain`][qermod.noise.utils.chain].
    """

    blocks: tuple[AbstractNoise, ...]

    @model_validator(mode="after")
    def verify_all_protocols(self) -> CompositeNoise:
        """Make sure all protocols are correct in terms and their combination too."""

        primitives = self.flatten()
        types = [type(p.protocol) for p in primitives]
        unique_types = set(types)
        if Noise.DIGITAL in unique_types and Noise.ANALOG in unique_types:
            raise ValueError("Cannot define a config with both DIGITAL and ANALOG noises.")

        if Noise.ANALOG in unique_types:
            if Noise.READOUT in unique_types:
                raise ValueError("Cannot define a config with both READOUT and ANALOG noises.")
            if types.count(Noise.ANALOG) > 1:
                raise ValueError("Multiple ANALOG Noises are not supported yet.")

        if Noise.READOUT in unique_types:
            if (
                primitives[-1].protocol not in Noise.READOUT.list()
                or types.count(Noise.READOUT) > 1
            ):
                raise ValueError("Only define a Noise with one READOUT as the last instance.")
        return self

    @field_serializer("blocks")
    def serialize_blocks(self, blocks: tuple[AbstractNoise, ...]) -> dict:
        return {str(i): blocks[i].model_dump() for i in range(len(blocks))}

    def __iter__(self) -> CompositeNoise:
        self._iterator = iter(self.blocks)
        return self

    def flatten(self) -> list[AbstractNoise]:
        all_ops = list()
        for b in self.blocks:
            if isinstance(b, CompositeNoise):
                all_ops += b.flatten()
            else:
                all_ops += [b]
        return all_ops

    def __next__(self) -> AbstractNoise:
        return next(self._iterator)

    def __getitem__(self, item: int) -> AbstractNoise:
        return self.blocks[item]

    def __len__(self) -> int:
        return len(self.blocks)

    def __contains__(self, other: object) -> bool:
        # Check containment by instance.
        if isinstance(other, AbstractNoise):
            for b in self.blocks:
                if isinstance(b, CompositeNoise) and other in b:
                    return True
                elif b == other:
                    return True
        # Check containment by type.
        elif isinstance(other, type):
            for b in self.blocks:
                if isinstance(b, CompositeNoise) and other in b:
                    return True
                elif type(b) is other:
                    return True
        else:
            raise TypeError(
                f"Can not check for containment between {type(self)} and {type(other)}."
            )
        return False

    def filter(self, noise_type: type[Noise] | NoiseType) -> list[AbstractNoise]:

        blocks = [block.filter(noise_type) for block in self.blocks]
        if blocks != [None] * len(blocks):
            return [b[0] for b in blocks if len(b) == 1]  # type: ignore [return-value]
        return list()
