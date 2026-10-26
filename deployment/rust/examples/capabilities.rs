//! Display chip capabilities from the silicon model.
//!
//! No hardware required — reads from the compiled-in chip spec.
//!
//! ```bash
//! cargo run --example capabilities
//! ```

use akida_driver::pcie_ids::{self, ChipVariant};

fn main() {
    println!("🧠 Akida Chip Family — Silicon Model\n");

    let variants = [
        ("AKD1000", ChipVariant::Akd1000),
        ("AKD1500", ChipVariant::Akd1500),
        ("AKD2500", ChipVariant::Akd2500),
        ("Pico",    ChipVariant::Pico),
    ];

    for (name, variant) in &variants {
        println!("{name}:");
        println!("  Generation:   {:?}", variant.generation());
        println!("  NP count:     {}", variant.np_count());
        println!("  SRAM:         {} MB", variant.sram_mb());
        println!("  Local RAM:    {} KB", variant.local_ram_kb());
        println!("  PCIe lanes:   {}{}", variant.pcie_lanes(),
                 if !variant.is_pcie() { " (not PCIe — AXI/SPI)" } else { "" });
        println!("  Weight bits:  {:?}", variant.weight_bits());
        println!("  Max clock:    {} MHz", variant.max_clock_mhz());
        println!("  Production:   {}", if variant.is_production() { "✅" } else { "🔲 pre-silicon" });
        println!();
    }

    println!("Vendor ID: 0x{:04X}", pcie_ids::BRAINCHIP_VENDOR_ID);
    println!("lspci:     {}", pcie_ids::lspci_filter());
    println!("\nAll device IDs: {:?}", pcie_ids::ALL_DEVICE_IDS);
    println!("Future IDs:    {:?}", pcie_ids::ALL_DEVICE_IDS);
}
