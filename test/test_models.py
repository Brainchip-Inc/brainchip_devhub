"""
Model tests for the CI, parametrized over the pretrained models discovered in the repository
(see conftest.py and discover_models.py).

- float models (*.h5): quantize and run one forward pass on random inputs. No accuracy, no
  hardware — only guards against float models that can no longer be quantized with the
  pinned toolchain.
- quantized models (*_qat.h5): convert to Akida, map on the device and verify that software
  and hardware backends produce identical outputs.
- akida models (*.fbz): load, map on the device and run inference on random inputs. The
  stored .fbz artifacts are what end users deploy, so this guards against files that no
  longer load or map with the pinned toolchain and driver.

The hardware tests carry the `hardware` marker: the float workflow deselects them with
`-m "not hardware"`, the hardware workflow selects them with `-m hardware`.
"""

import numpy as np
import pytest

import akida
import cnn2snn
import quantizeml

from akida_models.model_io import load_model


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def get_device_or_skip(arch):
    """Return the first Akida device matching the architecture, or skip.

    Args:
        arch: "v1" or "v2" (from the model's location, akida1/ or akida2/).
    """
    devices = akida.devices()
    if not devices:
        pytest.skip("No Akida hardware device detected")
    target = akida.IpVersion.v1 if arch == "v1" else akida.IpVersion.v2
    for device in devices:
        if device.ip_version == target:
            return device
    pytest.skip(f"No Akida {arch} device detected (found: {[d.desc for d in devices]})")


def load_keras_model(spec):
    """Load a float or quantized Keras model according to its architecture."""
    if spec.arch == "v1":
        return cnn2snn.load_quantized_model(str(spec.abs_path))
    return load_model(str(spec.abs_path))


def akida_version_context(arch):
    """Return the cnn2snn set_akida_version context manager for an arch."""
    return cnn2snn.set_akida_version(
        cnn2snn.AkidaVersion.v1 if arch == "v1" else cnn2snn.AkidaVersion.v2)


def get_random_inputs(model, seed, num_inputs=1):
    """Generate random inputs matching an akida.Model input layer.
    """
    rng = np.random.default_rng(seed)
    input_params = model.layers[0].parameters
    high = None
    if input_params.layer_type == akida.LayerType.Quantizer:
        # Depending on the output_signed, we modify the ranges of the inputs.
        input_variables = model.layers[0].variables
        if input_params.output_signed == 0:
            # Quantizer outputs are uint8, therefore inputs must have the following ranges.
            low = -(input_variables["zero_points"] / input_variables["scales"])
            high = (255. - input_variables["zero_points"]) / input_variables["scales"]
        else:
            # Quantizer outputs are int8, therefore inputs must have the following ranges.
            low = (-128. - input_variables["zero_points"]) / input_variables["scales"]
            high = (127. - input_variables["zero_points"]) / input_variables["scales"]
        if input_params.channels_first:
            low = low.reshape((-1, 1, 1))
            high = high.reshape((-1, 1, 1))
        return rng.uniform(low, high, (num_inputs, *model.input_shape)).astype(np.float32)
    if input_params.layer_type == akida.LayerType.InputData:
        if input_params.input_signed:
            type = np.int8 if input_params.input_bits <= 8 else np.int16
        else:
            # 1.0 specific case with input_bits = 4
            type = np.uint8
            high = 2**(input_params.input_bits) - 1
    else:
        type = np.uint8
    inputs = rng.integers(low=np.iinfo(type).min,
                          high=high or np.iinfo(type).max + 1,
                          size=(num_inputs, *model.input_shape),
                          dtype=type)
    return inputs


def skip_if_akd1500_memory_exceeded(device, model_hw):
    """Skip the test when the mapped program exceeds AKD1500 memory.
    """
    desc = device.desc
    prefix = 'PCIe/AKD1500/'
    if prefix in desc:
        if 'MB' not in desc:
            # No host memory support, default is 1MB internal memory
            mem_size = 1 * 1024 * 1024
        else:
            desc = desc[len(prefix):]
            mem_size = int(desc[:desc.find('MB')]) * 1024 * 1024
        for seq in model_hw.sequences:
            program = seq.program
            if program:
                model_size = len(program)
                if model_size > mem_size:
                    pytest.skip(f"Not enough extended memory, requires {model_size} but got "
                                f"{mem_size}.")


def map_and_assert_hw(model, device):
    """Map an akida.Model on the device and assert it is HW-supported.

    Returns the mapped hardware model (a fresh akida.Model sharing the layers,
    so the caller's model can keep running on the software backend).
    """
    model_hw = akida.Model(model.layers)
    model_hw.map(device, hw_only=False)
    hw_seq = sum(1 for seq in model_hw.sequences if seq.backend == akida.BackendType.Hardware)
    assert hw_seq > 0, "Model not HW supported but expected supported."
    skip_if_akd1500_memory_exceeded(device, model_hw)
    return model_hw


def assert_sw_hw_outputs_match(model_sw, model_hw):
    """Run 5 forwards of 2 random samples on both backends and compare exactly.
    """
    output_sw, output_hw = None, None
    for _ in range(5):
        input_sample = get_random_inputs(model_sw, seed=20260827, num_inputs=2)
        output_sample = model_sw.forward(input_sample)
        output_sw = output_sample if output_sw is None else np.concatenate([output_sw,
                                                                            output_sample])
        output_sample = model_hw.forward(input_sample)
        output_hw = output_sample if output_hw is None else np.concatenate([output_hw,
                                                                            output_sample])
    np.testing.assert_array_equal(output_sw, output_hw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_float_model_quantize_sanity(float_spec):
    num_samples = 2
    model = load_keras_model(float_spec)
    if float_spec.arch == "v1":
        quantized = cnn2snn.quantize(model,
                                     input_weight_quantization=8,
                                     weight_quantization=4,
                                     activ_quantization=4)
    else:
        quantized = quantizeml.models.quantize(model, num_samples=num_samples)

    rng = np.random.default_rng(0)
    inputs = rng.uniform(0, 255, (num_samples, *model.input_shape[1:])).astype(np.float32)
    outputs = quantized.predict(inputs, verbose=0)

    assert outputs.shape[0] == num_samples
    assert np.all(np.isfinite(outputs)), "Quantized model produced non-finite outputs"


@pytest.mark.hardware
@pytest.mark.flaky(reruns=3, rerun_delay=2)
def test_quantized_model_sw_hw_match(quantized_spec):
    device = get_device_or_skip(quantized_spec.arch)
    model_keras = load_keras_model(quantized_spec)
    with akida_version_context(quantized_spec.arch):
        model_sw = cnn2snn.convert(model_keras)

    model_hw = map_and_assert_hw(model_sw, device)
    assert_sw_hw_outputs_match(model_sw, model_hw)


@pytest.mark.hardware
@pytest.mark.flaky(reruns=3, rerun_delay=2)
def test_akida_model_load_map_inference(akida_spec):
    num_samples = 2
    device = get_device_or_skip(akida_spec.arch)
    model = akida.Model(str(akida_spec.abs_path))
    model_hw = map_and_assert_hw(model, device)

    inputs = get_random_inputs(model_hw, seed=20260827, num_inputs=num_samples)
    outputs = model_hw.forward(inputs)
    assert outputs.shape[0] == num_samples
