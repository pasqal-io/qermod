# `Qermod`

Running programs on NISQ devices often leads to partially useful results due to the presence of noise.
In order to perform realistic simulations, a number of noise models are defined in `Qermod` (for digital or analog operations and simulated readout errors) are supported in `Qadence` through their implementation in backends and
corresponding error mitigation techniques whenever possible.

# Noise

## Readout errors

State Preparation and Measurement (SPAM) in the hardware is a major source of noise in the execution of
quantum programs. They are typically described using confusion matrices of the form:

$$
T(x|x')=\delta_{xx'}
$$

Two types of readout protocols are available:

- `NoiseCategory.READOUT.INDEPENDENT` where each bit can be corrupted independently of each other.
- `NoiseCategory.READOUT.CORRELATED` where we can define of confusion matrix of corruption between each
possible bitstrings.


## Analog noisy simulation

At the moment, analog noisy simulations are only compatible with the Pulser backend.

## Digital noisy simulation

When dealing with programs involving only digital operations, several options are made available from [PyQTorch](https://pasqal-io.github.io/pyqtorch/latest/noise/) via the `NoiseCategory.DIGITAL`.

# Implementation

## PrimitiveNoise

A primitive Noise models can be defined via the `PrimitiveNoise`. It contains a noise configuration
defined by a `NoiseProtocol` type and an `error_definition` argument. Several predefined types are available in `qermod.protocols`.

```python exec="on" source="material-block" session="noise" result="json"
from qermod import PrimitiveNoise
from qermod import protocols
from qadence.types import NoiseProtocol

analog_noise = protocols.AnalogDepolarizing(error_definition=0.1)
digital_noise = protocols.Bitflip(error_definition=0.1)
readout_noise = protocols.IndependentReadout(error_definition=0.1)

simple_primitive = PrimitiveNoise(protocol=NoiseProtocol.DIGITAL.BITFLIP, error_definition=0.1)
```

## Chaining

One can also compose noise configurations via the `chain` method, or by using the `+` or `+=` operator.

```python exec="on" source="material-block" session="noise" result="json"
from qermod import chain

digital_readout = digital_noise + readout_noise
print(digital_readout)

digital_readout = chain(digital_noise, readout_noise)
print(digital_readout)
```

!!! warning "Noise scope"
    Note it is not possible to define a noise configuration with both digital and analog noises, both readout and analog noises, several analog noises, several readout noises, or a readout noise that is not the last defined protocol in a sequence.

# Implement parametric noise

Noise definition can be made parametric via `qadence.parameters.Parameter`:


```python exec="on" source="material-block" session="noise" result="json"
from qadence.parameters import Parameter
digital_noise = protocols.Bitflip(error_definition=Parameter('p', trainable=True))

print(digital_noise)
```

# Serialization

Regarding serialization, we can use `qermod.serialize` and `qermod.deserialize`:

```python exec="on" source="material-block" session="noise" result="json"
from qermod import serialize, deserialize, Bitflip
noise = Bitflip(error_definition=0.1)
noise_serial = deserialize(serialize(noise))
assert noise == noise_serial
```

# Implement parametric noise

Noise definition can be made parametric via `qadence.parameters.Parameter`:


```python exec="on" source="material-block" session="noise" result="json"
from qadence.parameters import Parameter
digital_noise = protocols.Bitflip(error_definition=Parameter('p', trainable=True))

print(digital_noise)
```
