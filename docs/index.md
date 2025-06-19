# Noisy Simulation

Running programs on NISQ devices often leads to imperfect results due to the presence of noise. In order to perform realistic simulations, a number of noise models (for digital operations, analog operations and simulated readout errors) are supported in `Qadence`.

Noisy simulations shift the quantum paradigm from a close-system (noiseless case) to an open-system (noisy case) where a quantum system is represented by a probabilistic combination $p_i$ of possible pure states $|\psi_i \rangle$. Thus, the system is described by a density matrix $\rho$ (and computation modify the density matrix) defined as follows:

$$
\rho = \sum_i p_i |\psi_i\rangle \langle \psi_i|
$$

The noise protocols applicable in `Qermod` are classified into three types: digital (for digital operations), analog (for analog operations), and readout error (for measurements).

# Available noise models

## Digital noisy simulation

Digital noise refer to unintended changes occurring with reference to the application of a noiseless digital gate operation. Several noise models
are made available via the[`PyQTorch` backend](https://pasqal-io.github.io/pyqtorch/latest/noise/).
Given an `error_definition` user-defined input, we support the following digital noise models:

- `Bitflip`: flips between |0⟩ and |1⟩ with `error_definition`
- `Phaseflip`: flips the phase of a qubit by applying a Z gate with `error_definition`
- `DigitalDepolarizing`: randomizes the state of a qubit by applying I, X, Y, or Z gates with equal `error_definition`
- `PauliChannel`: applies the Pauli operators (X, Y, Z) to a qubit with specified probabilities (via `error_definition`)
- `AmplitudeDamping`: models the asymmetric process through which the qubit state |1⟩ irreversibly decays into the state |0⟩ with `error_definition`
- `PhaseDamping`: similar to AMPLITUDE_DAMPING but concerning the phase
- `GeneralizedAmplitudeDamping`: extends amplitude damping; the first float is `error_definition` of amplitude damping, and second float is the `damping_rate`.

## Readout errors

Readout errors are linked to the incorrect measurement outcomes from the system.
They are typically described using confusion matrices of the form:

$$
T(x|x')=\delta_{xx'}
$$

Two types of readout protocols are available:

- `Independent` where each bit can be corrupted independently of each other.
- `Correlated` where we can define of confusion matrix of corruption between each
possible bitstrings.


## Analog noisy simulation

Analog noise can be set for analog operations.
At the moment, analog noisy simulations are only compatible with the `Pulser` backend, and we support the following models:

- `AnalogDepolarizing`: evolves to the maximally mixed state with probabilities defined with `error_definition`
- `Dephasing`: induces the loss of phase coherence without affecting the population of computational basis states

# Implementation

Several predefined noise models are available in `Qermod`.

```python exec="on" source="material-block" session="noise" result="json"
from qermod import AnalogDepolarizing, Bitflip, IndependentReadout

analog_noise = AnalogDepolarizing(error_definition=0.1)
digital_noise = Bitflip(error_definition=0.1)
readout_noise = IndependentReadout(error_definition=0.1)
```

## Chaining

One can also compose noise configurations via the `chain` method, or by using the `|` or `|=` operator.

```python exec="on" source="material-block" session="noise" result="json"
from qermod import chain

digital_readout = digital_noise | readout_noise
print(digital_readout)

digital_readout = chain(digital_noise, readout_noise)
print(digital_readout)
```

!!! warning "Noise scope"
    Note it is not possible to define a noise configuration with both digital and analog noises, both readout and analog noises, several analog noises, several readout noises, or a readout noise that is not the last defined protocol in a sequence.

## Implement parametric noise

Noise definition can be made parametric via `qadence.parameters.Parameter`:


```python exec="on" source="material-block" session="noise" result="json"
from qadence.parameters import Parameter
digital_noise = protocols.Bitflip(error_definition=Parameter('p', trainable=True))
```

## Serialization

Serialization is enabled via the `qermod.serialize` and `qermod.deserialize` functions:

```python exec="on" source="material-block" session="noise" result="json"
from qermod import serialize, deserialize, Bitflip
noise = Bitflip(error_definition=0.1)
noise_serial = deserialize(serialize(noise))
assert noise == noise_serial
```


## Specifying target gates

To specify a gate can only act on a type of gates or set of gates, we need to specify the `target_gates` attribute:

```python exec="on" source="material-block" session="noise" result="json"
from qadence import X, Y
noise = Bitflip(error_definition=0.1, target_gates=[X, Y(0)]) # any gate of type X or any Y applied on qubit 0
```

## Filtering

We can filter by noise type via the `filter` method:

```python exec="on" source="material-block" session="noise" result="json"
from qermod import Noise
filtered_noise = digital_readout.filter(Noise.DIGITAL)
print(filtered_noise) # markdown-exec: hide
```
