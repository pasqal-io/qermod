from __future__ import annotations

import pytest
import torch
from qadence import DiffMode, QuantumModel
from qadence.backends import backend_factory
from qadence.blocks import chain
from qadence.circuit import QuantumCircuit
from qadence.operations import AnalogRX, AnalogRZ, Z
from qadence.types import PI, BackendName

from qermod import NoiseHandler, NoiseProtocol


@pytest.mark.xfail(reason="Should fix type checking with qermod.")
def test_batched_noisy_simulations() -> None:
    noiseless_pulser_sim = torch.tensor([[0.3597]])
    batched_noisy_pulser_sim = torch.tensor([[0.4160, 0.4356, 0.4548, 0.4737]])

    analog_block = chain(AnalogRX(PI / 2.0), AnalogRZ(PI))
    observable = [Z(0) + Z(1)]
    circuit = QuantumCircuit(2, analog_block)
    model_noiseless = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    noiseless_expectation = model_noiseless.expectation()

    options = {"noise_probs": [0.1, 0.2, 0.3, 0.4]}
    noise = NoiseHandler(protocol=NoiseProtocol.ANALOG.DEPHASING, options=options)
    model_noisy = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PULSER,
        diff_mode=DiffMode.GPSR,
        noise=noise,
    )
    batched_noisy_expectation = model_noisy.expectation()
    assert torch.allclose(noiseless_expectation, noiseless_pulser_sim, atol=1.0e-3)
    assert torch.allclose(batched_noisy_expectation, batched_noisy_pulser_sim, atol=1.0e-3)

    # test backend itself
    backend = backend_factory(backend=BackendName.PULSER, diff_mode=DiffMode.GPSR)
    (pulser_circ, pulser_obs, embed, params) = backend.convert(circuit, observable)
    batched_native_expectation = backend.expectation(
        pulser_circ, pulser_obs, embed(params, {}), noise=noise
    )
    assert torch.allclose(batched_native_expectation, batched_noisy_pulser_sim, atol=1.0e-3)
