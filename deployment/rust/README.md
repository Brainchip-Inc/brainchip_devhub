# Rust Deployment — Pure Rust Akida Driver

**No Python. No C++ SDK. No MetaTF. No kernel module required.**

Deploy trained Akida models to hardware using a pure Rust stack. Works on
any Linux kernel (5.15+). VFIO primary backend — no kernel module in the data path.

| | Python SDK | C++ Package | **Rust (this)** |
|---|---|---|---|
| Runtime deps | Python 3.10+ | libakida.so | **None** |
| Kernel module | akida_pcie.ko | akida_pcie.ko | **None (VFIO)** |
| Linux > 6.9 | ⚠️ driver rebuild | ⚠️ driver rebuild | **✅ works** |
| Binary size | ~200 MB env | ~50 MB | **~4 MB** |
| Inference latency | SDK overhead | SDK overhead | **54 µs direct** |

## Quick Start

```bash
# 1. Clone rustChip (the Rust driver)
git clone https://github.com/syntheticChemistry/rustChip.git
cd rustChip

# 2. Verify — 376 tests, ~2 seconds, no hardware required
cargo test --workspace

# 3. Build
cargo build --release

# 4. One-time VFIO setup (no root needed after this)
sudo ./scripts/bind-akida-vfio.sh

# 5. Enumerate devices
cargo run --release --bin akida -- enumerate
```

## Load and Run a Model

```rust
use akida_driver::{DeviceManager, InferenceExecutor, InferenceConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Discover hardware
    let mgr = DeviceManager::discover()?;
    let caps = mgr.devices()[0].capabilities();
    println!("Found: {:?} — {} NPs, {} MB SRAM",
             caps.chip_version, caps.npu_count, caps.memory_mb);

    // Load model and run inference
    let model_bytes = std::fs::read("model.fbz")?;
    let mut dev = mgr.open_first()?;
    dev.write(&model_bytes)?;

    let input = vec![0u8; 224 * 224 * 3]; // your preprocessed image
    let mut output = vec![0u8; 1024];
    dev.read(&mut output)?;

    println!("Output: {:?}", &output[..10]);
    Ok(())
}
```

## Parse Any .fbz Model

rustChip parses all models from the BrainChip model zoo — v1 and v2:

```rust
use akida_models::Model;

let model = Model::load("plant_village_mobilenetv1_0.5_224_i8_w8_a8.fbz")?;
println!("Layers: {}, Parameters: {}", model.layer_count(), model.param_count());
```

The `.fbz` format is Snappy-compressed FlatBuffers. rustChip handles the
decompression and parsing natively — no Python, no SDK.

## Backends

| Backend | Requirements | Performance | When to use |
|---------|-------------|-------------|-------------|
| **VFIO** (primary) | IOMMU enabled | Full DMA, 37 MB/s | Production (recommended) |
| **Kernel** (fallback) | akida_pcie.ko loaded | /dev/akida* R/W | Legacy systems |
| **Software** (testing) | None | CPU emulation | CI, development |

### VFIO Setup (one-time)

```bash
# Bind Akida device to vfio-pci
sudo ./scripts/bind-akida-vfio.sh

# Install udev rule for user-level access (no sudo at runtime)
sudo cp scripts/99-akida-vfio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

After setup, all operations are userspace — no root required.

## Measured Performance (AKD1000, PCIe x1 Gen2)

| Metric | Value |
|--------|-------|
| Single inference | 54 µs / 18,500 Hz |
| Batch=8 | 390 µs/sample / 20,700 /s |
| DMA throughput | 37 MB/s sustained |
| Energy per inference | 1.4 µJ |
| Weight swap (`set_variable()`) | 86 µs |
| 24-hour production run | 5,978 calls, 0 errors |

## Supported Hardware

| Chip | Status | Notes |
|------|--------|-------|
| AKD1000 | ✅ Full support | PCIe x1 Gen2, validated in production |
| AKD1500 | ✅ Full support | Same Akida 1.0 IP, PCIe x2 Gen2 |
| AKD2500 | 🔲 Placeholder | Akida 2.0, TSMC 12nm, tape-out Dec 2026 |
| Akida Pico | 🔲 Placeholder | µW NPU, AXI/SPI (not PCIe) |

## Advanced: Multi-Tenancy

Run 7 independent models simultaneously on one AKD1000:

```rust
use akida_driver::tenancy::MultiTenantDevice;

let mut mt = MultiTenantDevice::new(device)?;
mt.load_at_offset("classifier.fbz", 0x0000)?;    // 179 NPs
mt.load_at_offset("detector.fbz",   0x00B3)?;     // 134 NPs
mt.load_at_offset("kws.fbz",        0x0139)?;     // 220 NPs
// ... up to 7 programs, 814/1000 NPs used
```

## Advanced: Hybrid ESN (tanh fix)

The AKD1000's bounded ReLU breaks reservoir computing. `HybridEsn` fixes it:

```rust
use akida_driver::HybridEsn;

// Use your existing tanh-trained weights — no retraining needed
let mut esn = HybridEsn::from_weights(&w_in, &w_res, &w_out, 0.3)?;
let prediction = esn.step(&features)?;  // 18,500 Hz, 1.4 µJ
```

Hardware does the matrix multiply (int4, 54 µs), host applies tanh (< 1 µs).
Full accuracy at hardware speed. No MetaTF required.

## Links

- **Source**: [syntheticChemistry/rustChip](https://github.com/syntheticChemistry/rustChip)
- **Hardware discoveries**: [BEYOND_SDK.md](https://github.com/syntheticChemistry/rustChip/blob/main/docs/BEYOND_SDK.md)
- **Integration guide**: [INTEGRATION_GUIDE.md](https://github.com/syntheticChemistry/rustChip/blob/main/specs/INTEGRATION_GUIDE.md)
- **PR to BrainChip**: [akida_dw_edma #20](https://github.com/Brainchip-Inc/akida_dw_edma/pull/20)

## License

rustChip is AGPL-3.0-or-later. This deployment guide is CC-BY-SA 4.0.

A permissive license exception is available to BrainChip and any NPU vendor
willing to engage on reciprocal terms. Contact: ecoPrimal@pm.me
(Kevin Mok, ORCID 0009-0004-2141-0321).
