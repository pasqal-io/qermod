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
