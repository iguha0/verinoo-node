//! `onnx-import` — convert an ONNX classifier into a runtime graph file.
//!
//! ```text
//! onnx-import --onnx model.onnx --model-id acme-support-v2 \
//!   --labels billing,refund,shipping --out models/acme-support-v2.graph.json
//! ```
//!
//! Parity against the float graph runs by default (`--check N` samples, 0
//! disables it) because a conversion that changes answers is worse than no
//! conversion.

use std::path::Path;

use verinoo_onnx::check::{compare, sample_features};
use verinoo_onnx::convert::op_chain;
use verinoo_onnx::{convert, parse_model, ImportOptions};

const USAGE: &str = "\
onnx-import — convert an ONNX classifier into a verinoo-runtime graph

USAGE:
  onnx-import --onnx <file.onnx> --model-id <id> --labels <a,b,c> [options]

OPTIONS:
  --onnx <path>          ONNX model to read (required)
  --model-id <id>        identifier tasks will name in targetModel (required)
  --labels <a,b,c>       class names, in the model's output order (required)
  --out <path>           graph file to write (default: models/<id>.graph.json)
  --features <n>         featuriser width; only needed if the ONNX input is dynamic
  --description <text>   human-readable description stored in the graph
  --architecture <id>    architecture label (default: derived from layer widths)
  --check <n>            parity samples to run (default 32, 0 disables)
";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn next(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut onnx_path: Option<String> = None;
    let mut model_id: Option<String> = None;
    let mut labels: Option<Vec<String>> = None;
    let mut out: Option<String> = None;
    let mut features: Option<usize> = None;
    let mut description = String::new();
    let mut architecture: Option<String> = None;
    let mut checks: usize = 32;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--onnx" => onnx_path = Some(next(&args, &mut i, "--onnx")?),
            "--model-id" => model_id = Some(next(&args, &mut i, "--model-id")?),
            "--labels" => {
                let raw = next(&args, &mut i, "--labels")?;
                let parsed: Vec<String> = raw
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                labels = Some(parsed);
            }
            "--out" => out = Some(next(&args, &mut i, "--out")?),
            "--features" => {
                let raw = next(&args, &mut i, "--features")?;
                features = Some(
                    raw.parse()
                        .map_err(|_| format!("--features expects a number, got `{raw}`"))?,
                );
            }
            "--description" => description = next(&args, &mut i, "--description")?,
            "--architecture" => architecture = Some(next(&args, &mut i, "--architecture")?),
            "--check" => {
                let raw = next(&args, &mut i, "--check")?;
                checks = raw
                    .parse()
                    .map_err(|_| format!("--check expects a number, got `{raw}`"))?;
            }
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}` (try --help)")),
        }
        i += 1;
    }

    let onnx_path = onnx_path.ok_or("--onnx is required (try --help)")?;
    let model_id = model_id.ok_or("--model-id is required (try --help)")?;
    let labels = labels.ok_or("--labels is required (try --help)")?;
    if labels.is_empty() {
        return Err("--labels must name at least one class".into());
    }
    let out = out.unwrap_or_else(|| format!("models/{model_id}.graph.json"));

    let bytes = std::fs::read(&onnx_path).map_err(|e| format!("reading {onnx_path}: {e}"))?;
    let model = parse_model(&bytes)?;
    println!(
        "read {onnx_path}: ir_version {}, producer {}, opset {}, {} node(s), {} initializer(s)",
        model.ir_version,
        if model.producer.is_empty() {
            "unknown"
        } else {
            model.producer.as_str()
        },
        model.opset,
        model.graph.nodes.len(),
        model.graph.initializers.len()
    );
    println!("ops: {}", op_chain(&model.graph).join(" -> "));

    let mut opts = ImportOptions::new(model_id.clone(), labels);
    opts.features = features;
    opts.description = description;
    opts.architecture = architecture;

    let imported = convert(&model, &opts)?;
    let spec = &imported.spec;
    println!(
        "graph input `{}`, {} features, {} labels",
        imported.input_name,
        spec.features,
        spec.labels.len()
    );
    let chain: Vec<String> = spec
        .layers
        .iter()
        .map(|l| {
            if l.op == "dense" {
                format!("dense {}->{}", l.input, l.output)
            } else {
                l.op.clone()
            }
        })
        .collect();
    println!("runtime layers: {}", chain.join(", "));
    for note in &imported.notes {
        println!("note: {note}");
    }

    let json = serde_json::to_value(spec).map_err(|e| e.to_string())?;
    let loaded = verinoo_runtime::Model::from_value(&json)?;
    println!(
        "parameters: {}, weightsHash: {}",
        loaded.parameter_count(),
        loaded.weights_hash()
    );

    if checks > 0 {
        let samples = sample_features(spec.features, checks);
        let parity = compare(spec, &imported.float, &samples)?;
        println!(
            "parity: {} sample(s), max |Δlogit| = {:.3e}, argmax {}/{}",
            parity.samples, parity.max_logit_diff, parity.argmax_agree, parity.samples
        );
        if !parity.is_exact_enough() {
            for m in &parity.mismatches {
                println!("  mismatch: {m}");
            }
            return Err(format!(
                "quantisation changed the predicted class on {} of {} samples; \
                 refusing to publish this graph",
                parity.mismatches.len(),
                parity.samples
            ));
        }
    } else {
        println!("parity: skipped (--check 0)");
    }

    if let Some(parent) = Path::new(&out).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("creating {}: {e}", parent.display()))?;
        }
    }
    let mut text = serde_json::to_string_pretty(&json).map_err(|e| e.to_string())?;
    text.push('\n');
    std::fs::write(&out, &text).map_err(|e| format!("writing {out}: {e}"))?;
    println!("wrote {out}");
    Ok(())
}
