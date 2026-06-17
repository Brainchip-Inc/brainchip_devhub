import binascii
import struct
import time
import multiprocessing as mp

import akida
import numpy as np
from cnn2snn import set_akida_version, AkidaVersion

try:
    from pyftdi.i2c import I2cController
    _PYFTDI_AVAILABLE = True
except ImportError:
    _PYFTDI_AVAILABLE = False

_AKD1500_I2C_URL = 'ftdi://ftdi:ft2232h/1'
_AKD1500_INA_CONFIGS = [(0x40, 0.002), (0x41, -0.05)]


def get_mapping_stats(ak_model):
    """Return mapping statistics for a mapped akida.Model.

    Args:
        ak_model: akida.Model that has already been mapped to a hardware or
                  virtual device.

    Returns:
        Tuple of (num_nps, num_passes, num_sequences).
    """
    total_nps = 0
    total_passes = 0
    total_sequences = len(ak_model.sequences)
    for seq in ak_model.sequences:
        total_passes += len(seq.passes)
        for hwpass in seq.passes:
            for layer in hwpass.layers:
                total_nps += len(layer.mapping.nps)
    return total_nps, total_passes, total_sequences


def get_akida_device(target_version=None):
    """Return the first available Akida hardware device, or None if unavailable.

    Args:
        target_version: Optional akida.IpVersion to match. If None, returns the
                        first detected device regardless of version.

    Returns:
        akida.Device if a matching device is found, otherwise None (inference
        falls back to the software backend).
    """
    devices = akida.devices()
    if len(devices)==0:
        print('No connected Akida hardware device detected.')
        print('Calls to akida will run on the software backend.')
        return None
    else:
        if target_version is None:
            if len(devices)>0:
                print(str(len(devices)) + ' Akida devices found. Using the first device detected.')
            return devices[0]
        else:
            for dd in akida.devices():
                if dd.ip_version == target_version:
                    print('Target Akida device found')
                    return dd
                print('Connected Akida Device does not match the requested IPVersion.')
                print('Calls to akida will run on the software backend.')
                return None

#----------------------------------------------------------------------------------
#
# Per-layer benchmarking
#
#----------------------------------------------------------------------------------

def _silence_output_layer(ak_model):
    """Set the output layer's threshold/act_step so it never fires.

    Used during per-layer benchmarking to prevent the truncated sub-model's
    final layer from producing output events whose copy-off could skew 
    timing.
    """
    layer = ak_model.layers[-1]
    if len(layer.get_variable_names()) > 0:
        th = layer.get_variable('threshold')
        layer.set_variable('threshold', np.ones_like(th) * 524287)
        act = layer.get_variable('act_step')
        layer.set_variable('act_step', np.ones_like(act)*65535)
    return ak_model


def _remove_final_maxpool(ak_model):
    """Strip the max-pool from the final convolutional layer if present.

    Sub-models built for per-layer benchmarking may end on a layer that includes
    maxpooling. Hardware constraints actually force max-pooling to occur between
    two layers, and so such sub-models would fail. This sanitizer function 
    removes the max-pooling op.
    """
    model_dict = ak_model.to_dict()
    layer_dict = model_dict['layers'][-1]
    if layer_dict['parameters']['layer_type'] in ['Convolutional', 'SeparableConvolutional']:
        if layer_dict['parameters']['pooling_height'] == 2:
            layer_dict['parameters']['pooling_height'] = -1
            layer_dict['parameters']['pooling_width'] = -1
            layer_dict['parameters']['pooling_stride_x'] = -1
            layer_dict['parameters']['pooling_stride_y'] = -1
            layer_dict['parameters']['pool_type'] = 0
            out_h = layer_dict['input_shape'][0]
            out_w = layer_dict['input_shape'][1]
            out_c = layer_dict['output_shape'][2]
            layer_dict['output_shape'] = [out_h, out_w, out_c]
            ak_model = akida.Model.from_dict(model_dict)
    return ak_model


def per_layer_benchmark(ak_model, device, samples, repeats=100, clock_freq=400e6):
    """Run a cumulative sub-model benchmark and return per-layer timing.

    Args:
        ak_model: akida.Model to benchmark.
        device: Akida hardware device (from get_akida_device).
        sample: Single input sample, shape (1, H, W, C), dtype uint8.
        repeats: Number of forward passes per sub-model layer.
        clock_freq: Hardware clock frequency in Hz (400 MHz for AKD1500).

    Returns:
        dict with keys:
            layer_names      – list of layer name strings
            per_layer_clocks – ndarray (num_layers,) mean clock cycles per layer
            per_layer_ms     – ndarray (num_layers,) mean latency in ms per layer
            cumulative_clocks – ndarray (num_layers, repeats) raw cumulative clocks
            cumulative_times  – ndarray (num_layers, repeats) raw wall-clock ns
    """
    num_layers = len(ak_model.layers)
    cumulative_clocks = np.zeros((num_layers, repeats))
    cumulative_times = np.zeros((num_layers, repeats))
    layer_names = []

    for ll in range(num_layers):
        cut_model = akida.Model(ak_model.layers[:ll + 1])
        cut_model = _silence_output_layer(cut_model)
        cut_model = _remove_final_maxpool(cut_model)
        cut_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)
        layer_names.append(cut_model.layers[-1].name)

        # Run a priming frame
        cut_model.forward(samples[0:1,])

        for rr in range(repeats):
            start_t = time.perf_counter_ns()
            cut_model.forward(samples[rr:rr+1,])
            stop_t = time.perf_counter_ns()
            cumulative_times[ll, rr] = stop_t - start_t
            cumulative_clocks[ll, rr] = cut_model.metrics['inference_clk']

    per_layer_clocks = np.copy(cumulative_clocks)
    per_layer_clocks[1:] = cumulative_clocks[1:] - cumulative_clocks[:-1]
    per_layer_ms = np.mean(per_layer_clocks / clock_freq * 1000, axis=1)

    # Print results
    col_w = max(len(n) for n in layer_names) + 2
    print(f"\n{'Layer':<{col_w}} {'Latency (ms)':>14} {'Clocks':>12}")
    print('-' * (col_w + 20))
    for name, ms in zip(layer_names, per_layer_ms):
        print(f'{name:<{col_w}} {ms:>14.4f}')
    print('-' * (col_w + 20))
    print(f"{'Total':<{col_w}} {per_layer_ms.sum():>14.4f}")


    return {
        'layer_names': layer_names,
        'per_layer_clocks': np.mean(per_layer_clocks, axis=1),
        'per_layer_ms': per_layer_ms,
        'cumulative_clocks': cumulative_clocks,
        'cumulative_times': cumulative_times,
    }

#----------------------------------------------------------------------------------
#
# AKD1500 Power + Latency Benchmarking
#
#----------------------------------------------------------------------------------

class InaFtdi:
    """I2C interface to an INA219 current/voltage sensor via FTDI."""

    def __init__(self, port, shunt, debug=False):
        """Initialise and reset the INA219 sensor over I2C.

        Args:
            port:  PyFTDI I2cPort for the sensor's I2C address.
            shunt: Shunt resistor value in ohms, used to convert shunt voltage
                   to current.
            debug: If True, print raw register bytes for each read/write.

        Raises:
            RuntimeError: If the configuration register readback does not match.
        """
        self.port = port
        self.debug = debug
        self.shunt = shunt
        port.flush()
        port.write_to(0, b'\x80\x00')
        time.sleep(0.1)
        port.write_to(0, b'\x45\x27')
        readback = port.read_from(0, 2)
        if debug:
            print(binascii.hexlify(readback))
        if readback != b'\x45\x27':
            raise RuntimeError(f'INA init failed: unexpected readback {readback!r}')

    def read_voltage(self):
        """Read the bus voltage from register 2. Returns volts (float)."""
        raw = self.port.read_from(2, 2)
        if self.debug:
            print(binascii.hexlify(raw))
        (vbus,) = struct.unpack('>H', raw)
        return vbus * 1.25 / 1000  # V

    def read_current(self):
        """Read the shunt voltage from register 1 and convert to current. Returns mA (float)."""
        raw = self.port.read_from(1, 2)
        if self.debug:
            print(binascii.hexlify(raw))
        (vshunt,) = struct.unpack('>h', raw)
        return vshunt * 2.5 / self.shunt / 1000  # mA


def _power_process_fn(stop_event, ready_event, power_queue, i2c_url, ina_configs):
    """Subprocess worker that streams INA219 power readings into a queue.

    Initialises the I2C bus inside the subprocess (avoids sharing the FTDI USB
    file descriptor across processes), then loops reading current and voltage
    until stop_event is set, putting (perf_counter, voltage_V, current_mA)
    tuples onto power_queue. Sets ready_event once the sensor is initialised.

    Args:
        stop_event:  multiprocessing.Event — set by the parent to stop sampling.
        ready_event: multiprocessing.Event — set here once the sensor is ready.
        power_queue: multiprocessing.Queue — receives (t, V, mA) tuples.
        i2c_url:     PyFTDI I2C bus URL string.
        ina_configs: List of (i2c_address, shunt_ohm) tuples; index 0 is used.
    """
    # Subprocess worker: initialise I2C inside the subprocess to avoid sharing
    # the USB file descriptor, then stream (timestamp, voltage, current) tuples.
    while True:
        try:
            i2c = I2cController()
            i2c.configure(i2c_url, frequency=400000)
            ina = InaFtdi(i2c.get_port(ina_configs[0][0]), ina_configs[0][1])
            time.sleep(0.1)
            break
        except Exception:
            pass

    # One-time voltage read as a sanity check; reuse as constant thereafter
    try:
        voltage = ina.read_voltage()
        print(f'  Supply voltage: {voltage:.3f} V')
    except Exception:
        voltage = 0.8  # nominal AKD1500 core voltage

    ready_event.set()

    n_ok = n_err = 0
    while not stop_event.is_set():
        try:
            current = ina.read_current()
            power_queue.put((time.perf_counter(), voltage, current))
            n_ok += 1
        except Exception:
            n_err += 1
    print(f'  Power process: {n_ok} ok, {n_err} errors')


def _check_power_available(i2c_url, ina_configs, retries=3, retry_delay=0.5):
    """Return True if pyftdi is installed and the I2C INA sensor is reachable."""
    if not _PYFTDI_AVAILABLE:
        return False
    for attempt in range(retries):
        i2c = None
        try:
            i2c = I2cController()
            i2c.configure(i2c_url, frequency=400000)
            ina = InaFtdi(i2c.get_port(ina_configs[0][0]), ina_configs[0][1])
            ina.read_current()
            return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(retry_delay)
        finally:
            if i2c is not None:
                try:
                    i2c.close()
                except Exception:
                    pass
    return False


def _analyze_power(readings, repeat_meta):
    """Compute power statistics from per-repeat floor/inference metadata.

    Args:
        readings:     list of (perf_counter_t, voltage_V, current_mA) tuples
                      from the continuously-running power subprocess.
        repeat_meta:  list of dicts, one per repeat, each with keys:
                        'floor_pre_start'  - perf_counter timestamp before the
                                             pre-inference floor sleep
                        'inf_timestamps'   - list of n_samples+1 perf_counter
                                             timestamps (first before inference,
                                             then one after each forward call)
                        'floor_post_end'   - perf_counter timestamp after the
                                             post-inference floor sleep
    Returns:
        dict with aggregate power statistics and the raw data needed for plotting.
    """
    readings_arr = np.array(readings)
    abs_times = readings_arr[:, 0]
    voltages  = readings_arr[:, 1]
    currents  = readings_arr[:, 2]
    powers    = voltages * currents  # mW

    per_repeat = []
    for meta in repeat_meta:
        t_pre  = meta['floor_pre_start']
        t_i0   = meta['inf_timestamps'][0]
        t_i1   = meta['inf_timestamps'][-1]
        t_post = meta['floor_post_end']

        floor_mask = (((abs_times >= t_pre)  & (abs_times < t_i0)) |
                      ((abs_times >  t_i1)   & (abs_times <= t_post)))
        inf_mask   =  (abs_times >= t_i0)   & (abs_times <= t_i1)

        avg_floor = float(np.mean(powers[floor_mask])) if np.any(floor_mask) else 0.0
        avg_total = float(np.mean(powers[inf_mask]))   if np.any(inf_mask)   else 0.0
        avg_dyn   = avg_total - avg_floor

        n_inf          = len(meta['inf_timestamps']) - 1
        inf_duration_s = t_i1 - t_i0
        per_repeat.append({
            'avg_floor_mw':          avg_floor,
            'avg_total_mw':          avg_total,
            'avg_dynamic_mw':        avg_dyn,
            'avg_energy_mj':         avg_total * inf_duration_s / n_inf,
            'avg_dynamic_energy_mj': avg_dyn   * inf_duration_s / n_inf,
        })

    def _mean(key):
        return float(np.mean([r[key] for r in per_repeat]))

    return {
        'readings':               readings,
        'repeat_meta':            repeat_meta,
        'avg_floor_mw':           _mean('avg_floor_mw'),
        'avg_total_mw':           _mean('avg_total_mw'),
        'avg_dynamic_mw':         _mean('avg_dynamic_mw'),
        'avg_energy_mj':          _mean('avg_energy_mj'),
        'avg_dynamic_energy_mj':  _mean('avg_dynamic_energy_mj'),
    }


def full_model_benchmark(
    ak_model,
    device,
    samples,
    map_mode=akida.MapMode.Minimal,
    repeats=10,
    batch_size=1,
    clock_freq=400e6,
    floor_duration=1.0,
    i2c_url=_AKD1500_I2C_URL,
    ina_configs=None,
):
    """Benchmark end-to-end inference latency, with optional power measurement.

    Power measurement requires an AKD1500 device, pyftdi installed, and the
    FTDI I2C hardware connected. If any of these are absent the function falls
    back silently to timing-only mode.

    Args:
        ak_model:       akida.Model to benchmark.
        device:         Akida hardware device (from get_akida_device).
        samples:        Input samples array, shape (N, H, W, C), dtype uint8.
        map_mode:       akida.MapMode to use when mapping the model.
        repeats:        Number of full passes through the samples array.
        batch_size:     Inference batch size.
        clock_freq:     Hardware clock frequency in Hz (400 MHz for AKD1500).
        floor_duration: Seconds of idle measurement before and after inference
                        (used to establish the power floor). Ignored when power
                        measurement is unavailable.
        i2c_url:        PyFTDI I2C bus URL for the INA sensor.
        ina_configs:    List of (i2c_address, shunt_ohm) tuples.

    Returns:
        dict with keys:
            map_mode        - akida.MapMode used
            mean_inf_clk    - mean clock cycles per forward call
            std_inf_clk     - std dev clock cycles
            mean_inf_ms     - mean wall-clock latency per forward call (ms)
            std_inf_ms      - std dev latency (ms)
            n_inferences    - total number of forward calls made
            power           - power analysis dict, or None if not measured
    """
    if ina_configs is None:
        ina_configs = _AKD1500_INA_CONFIGS

    # -------------------------------------------------------------------------
    # Detect power measurement availability
    # -------------------------------------------------------------------------
    measure_power = False
    if device.ip_version != akida.IpVersion.v1:
        print('Power measurement not available: only supported on AKD1500 (IpVersion.v1).')
    elif not _PYFTDI_AVAILABLE:
        print('Power measurement not available: pyftdi not installed.')
    else:
        print('Checking I2C power measurement hardware...')
        if _check_power_available(i2c_url, ina_configs):
            measure_power = True
            print('  I2C INA sensor detected — power measurement enabled.')
        else:
            print('  I2C INA sensor not found — running timing-only benchmark.')

    # -------------------------------------------------------------------------
    # Map model
    # -------------------------------------------------------------------------
    try:
        ak_model.map(device, mode=map_mode, hw_only=True)
    except:
        print('This model does not map fully to hardware. Quitting benchmarking...')
        return None

    # Priming pass
    ak_model.forward(samples[0:batch_size], batch_size=batch_size)

    # -------------------------------------------------------------------------
    # Inference loop (with or without power measurement)
    # -------------------------------------------------------------------------
    # Each repeat follows the structure: floor -> inference x n_samples -> floor.
    # This keeps the floor baseline adjacent to the inference window in every
    # repeat, making per-repeat power statistics self-contained.
    n_samples = len(samples)

    if measure_power:
        ctx = mp.get_context('fork')
        stop_event  = ctx.Event()
        ready_event = ctx.Event()
        power_queue = ctx.Queue()

        power_proc = ctx.Process(
            target=_power_process_fn,
            args=(stop_event, ready_event, power_queue, i2c_url, ina_configs),
            daemon=True,
        )
        power_proc.start()
        ready_event.wait()
        print('  Power measurement active.')

    inf_clks    = []
    repeat_meta = []

    for rr in range(repeats):
        # Pre-inference floor
        floor_pre_start = time.perf_counter()
        if measure_power:
            print(f'  Repeat {rr + 1}/{repeats}: recording floor ({floor_duration:.1f} s)...')
            time.sleep(floor_duration)

        # Inference pass
        inf_timestamps = [time.perf_counter()]
        for ii in range(0, n_samples, batch_size):
            ak_model.forward(samples[ii:ii + batch_size], batch_size=batch_size)
            inf_clks.append(ak_model.metrics['inference_clk'])
            inf_timestamps.append(time.perf_counter())

        # Post-inference floor
        if measure_power:
            time.sleep(floor_duration)
        floor_post_end = time.perf_counter()

        repeat_meta.append({
            'floor_pre_start': floor_pre_start,
            'inf_timestamps':  inf_timestamps,
            'floor_post_end':  floor_post_end,
        })
        print(f'  Repeat {rr + 1}/{repeats} done.')

    if measure_power:
        stop_event.set()
        readings = []
        while power_proc.is_alive():
            try:
                readings.append(power_queue.get(timeout=0.05))
            except Exception:
                pass
        power_proc.join()
        while not power_queue.empty():
            try:
                readings.append(power_queue.get_nowait())
            except Exception:
                break

    # -------------------------------------------------------------------------
    # Post-process timing
    # -------------------------------------------------------------------------
    inf_clks = np.array(inf_clks)
    # Compute per-call wall-clock times within each repeat (excludes the gap
    # between repeats caused by the floor sleep).
    per_call_ms = np.concatenate([
        np.diff(meta['inf_timestamps']) * 1000 for meta in repeat_meta
    ])

    # Prepare return values
    mean_inf_clk = float(np.mean(inf_clks))
    std_inf_clk = float(np.std(inf_clks))
    mean_clk_ms = mean_inf_clk / clock_freq * 1000
    mean_inf_ms = float(np.mean(per_call_ms))
    std_inf_ms = float(np.std(per_call_ms))
    n_inferences = len(inf_clks)
    power_results = _analyze_power(readings, repeat_meta) if measure_power else None

    # Print key results
    print(f'\n  Mean inference time:    {mean_inf_ms:.3f} ms  '
        f'(σ={std_inf_ms:.3f} ms)')
    print(f'  Mean on-chip time:      {mean_clk_ms:.3f} ms  '
        f'({mean_inf_clk:.0f} clocks)')
    print(f'  Total inferences run:   {n_inferences}')
    if power_results is not None:
        print(f'  Avg dynamic power:      {power_results["avg_dynamic_mw"]:.1f} mW')
        print(f'  Avg dynamic energy:     {power_results["avg_dynamic_energy_mj"]:.3f} mJ/inf')


    return {
        'map_mode':      map_mode,
        'mean_inf_clk':  mean_inf_clk,
        'std_inf_clk':   std_inf_clk,
        'mean_clk_ms':   mean_clk_ms,
        'mean_inf_ms':   mean_inf_ms,
        'std_inf_ms':    std_inf_ms,
        'n_inferences':  n_inferences,
        'power':         power_results,
    }
