//! Enumerate all Akida devices on the system.
//!
//! No hardware required — this will report "no devices found" gracefully
//! if no AKD1000/AKD1500 is present.
//!
//! ```bash
//! cargo run --example enumerate
//! ```

fn main() {
    println!("🧠 Akida Device Enumeration — Pure Rust\n");

    #[cfg(target_os = "linux")]
    {
        match akida_driver::DeviceManager::discover() {
            Ok(mgr) => {
                let devices = mgr.devices();
                if devices.is_empty() {
                    println!("No Akida devices found.");
                    println!("\nTo use VFIO backend:");
                    println!("  1. Ensure IOMMU is enabled (intel_iommu=on or amd_iommu=on)");
                    println!("  2. Run: sudo ./scripts/bind-akida-vfio.sh");
                    println!("\nTo use kernel backend:");
                    println!("  1. Load the module: sudo insmod akida-pcie.ko");
                    return;
                }

                println!("Found {} device(s):\n", devices.len());
                for (i, dev) in devices.iter().enumerate() {
                    let caps = dev.capabilities();
                    println!("Device {i}:");
                    println!("  Chip:      {:?}", caps.chip_version);
                    println!("  NPs:       {}", caps.npu_count);
                    println!("  SRAM:      {} MB", caps.memory_mb);
                    println!("  PCIe:      Gen{} x{}", caps.pcie.generation, caps.pcie.lanes);
                    println!();
                }
            }
            Err(e) => {
                println!("Discovery failed: {e}");
                println!("\nThis is expected if no Akida hardware is present.");
                println!("The software backend works without hardware — see parse_model example.");
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        println!("Hardware discovery requires Linux (VFIO/sysfs).");
        println!("The software backend and model parser work on any platform.");
        println!("\nTry: cargo run --example parse_model -- path/to/model.fbz");
    }
}
