from __future__ import annotations

from qermod.noise import AbstractNoise, PrimitiveNoise, CompositeNoise

def serialize(noise: AbstractNoise) -> dict:
    """Serialize noise.

    Args:
        noise (AbstractNoise): A noise configuration

    Returns:
        dict: Dictionaruy for serialization.
    """
    return noise.model_dump()

def deserialize(noise: dict) -> AbstractNoise:
    """Deserialize the noise dictionary back to an instance.

    Args:
        noise (dict): Dictionary of noise specifications.

    Returns:
        AbstractNoise: Instance.
    """
    if 'blocks' in noise:
        blocks = tuple()
        nb_blocks = len(noise['blocks'])
        for i in range(nb_blocks):
            blocks += (PrimitiveNoise(**noise['blocks'][str(i)]),)
        return CompositeNoise(blocks=blocks)
    else:
        return PrimitiveNoise(**noise)