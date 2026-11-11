//! Export the reference classifier as a runtime graph file.
//!
//!   cargo run -p verinoo-model --bin export -- \
//!     --model-id acme-support-v1 --out models/acme-support-v1.graph.json
//!
//! The exported graph runs through `verinoo-runtime` and produces byte-identical
//! scores to the compiled-in classifier, which makes it a faithful example of
//! how a customer's own trained weights are loaded.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_id = verinoo_model::MODEL_ID.to_string();
    let mut out = format!("models/{model_id}.graph.json");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-id" | "--model" if i + 1 < args.len() => {
                model_id = args[i + 1].clone();
                i += 1;
            }
            "--out" if i + 1 < args.len() => {
                out = args[i + 1].clone();
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let graph = verinoo_model::export_graph(&model_id);
    if let Some(parent) = Path::new(&out).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut text = serde_json::to_string_pretty(&graph)?;
    text.push('\n');
    std::fs::write(&out, &text)?;

    let params = graph["layers"][0]["weights"]
        .as_array()
        .map(|w| w.len())
        .unwrap_or(0)
        + verinoo_model::CLASSES;
    println!("wrote {out} (modelId={model_id}, {params} parameters)");
    println!("sha256: {}", verinoo_model::sha256_hex(&text));
    Ok(())
}
