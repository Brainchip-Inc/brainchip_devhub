//! Parse any .fbz model file from BrainChip's model zoo.
//!
//! Works on any platform — no hardware required.
//!
//! ```bash
//! # Download a model from Brainchip-Inc/models:
//! #   git lfs pull --include="vision/classification/mobilenetv1/**/*.fbz"
//!
//! cargo run --example parse_model -- path/to/model.fbz
//! ```

fn main() {
    println!("🧠 Akida Model Parser — Pure Rust\n");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run --example parse_model -- <path-to-model.fbz>");
        println!();
        println!("Download models from: https://github.com/Brainchip-Inc/models");
        println!("  git clone https://github.com/Brainchip-Inc/models.git");
        println!("  cd models && git lfs pull --include=\"*.fbz\"");
        return;
    }

    let path = &args[1];
    println!("Loading: {path}\n");

    match std::fs::read(path) {
        Ok(bytes) => {
            println!("  File size: {} bytes ({:.1} KB)", bytes.len(), bytes.len() as f64 / 1024.0);

            // Show raw header bytes for format inspection
            if bytes.len() >= 4 {
                println!("  Header:    [{:02x}, {:02x}, {:02x}, {:02x}]",
                         bytes[0], bytes[1], bytes[2], bytes[3]);
            }

            // The .fbz format is: [varint: uncompressed_size] [snappy_chunks] [zero_padding]
            // The first bytes are a Snappy varint, NOT magic bytes.
            println!("  Format:    Snappy-compressed FlatBuffer (.fbz)");

            // Attempt to find version string
            if let Some(pos) = bytes.windows(5).position(|w| w == b"2.19.") {
                let end = bytes[pos..].iter().position(|&b| b == 0).unwrap_or(10);
                let version = String::from_utf8_lossy(&bytes[pos..pos + end.min(20)]);
                println!("  SDK ver:   {version} (at offset {pos})");
            }

            println!("\n✅ File loaded successfully. Use akida_models::Model::load() for full parsing.");
        }
        Err(e) => {
            println!("❌ Failed to read {path}: {e}");
        }
    }
}
