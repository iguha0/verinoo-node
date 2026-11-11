//! `gguf-import` — convert a GGUF classifier's dense layers into a runtime graph.
//!
//! ```text
//! gguf-import --list model.gguf
//! gguf-import --gguf model.gguf --model-id acme-support-v3 \
//!   --labels billing,refund --layers fc1,fc2 --out models/acme-support-v3.graph.json
//! ```

use std::path::Path;

use verinoo_gguf::{convert, plan_layers, Gguf, GgufOptions};
use verinoo_onnx::check::{compare, sample_features};

const USAGE: &str = "\
gguf-import — convert a GGUF classifier into a verinoo-runtime graph

USAGE:
  gguf-import --gguf <file.gguf> --model-id <id> --labels <a,b,c> [options]
  gguf-import --gguf <file.gguf> --list

OPTIONS:
  --gguf <path>          GGUF file to read (required)
  --model-id <id>        identifier tasks will name in targetModel (required)
  --labels <a,b,c>       class names, in the model's output order (required)
  --layers <a,b>         layer tensor prefixes, in order (default: every *.weight)
  --activation <op>      relu (default) or none, applied between layers
  --out <path>           graph file to write (default: models/<id>.graph.json)
  --features <n>         expected input width; checked against the first layer
  --description <text>   human-readable description stored in the graph
  --architecture <id>    architecture label (default: gguf-<file architecture>)
  --check <n>            parity samples to run (default 32, 0 disables)
  --list                 print metadata and tensors, then exit
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

fn split_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path: Option<String> = None;
    let mut model_id: Option<String> = None;
    let mut labels: Option<Vec<String>> = None;
    let mut layers: Option<Vec<String>> = None;
    let mut activation = "relu".to_string();
    let mut out: Option<String> = None;
    let mut features: Option<usize> = None;
    let mut description = String::new();
    let mut architecture: Option<String> = None;
    let mut checks: usize = 32;
    let mut list = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => path = Some(next(&args, &mut i, "--gguf")?),
            "--model-id" => model_id = Some(next(&args, &mut i, "--model-id")?),
            "--labels" => labels = Some(split_list(&next(&args, &mut i, "--labels")?)),
            "--layers" => layers = Some(split_list(&next(&args, &mut i, "--layers")?)),
            "--activation" => activation = next(&args, &mut i, "--activation")?,
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
            "--list" => list = true,
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}` (try --help)")),
        }
        i += 1;
    }

    let path = path.ok_or("--gguf is required (try --help)")?;
    let mut file = Gguf::open(&path)?;
    println!(
        "read {path}: GGUF v{}, {} metadata entries, {} tensor(s), alignment {}",
        file.version,
        file.metadata.len(),
        file.tensors.len(),
        file.alignment
    );
    if let Some(arch) = file.architecture() {
        println!("architecture: {arch}");
    }

    if list {
        for tensor in &file.tensors {
            println!("  {:<48} {}", tensor.name, tensor.shape_string());
        }
        return Ok(());
    }

    let model_id = model_id.ok_or("--model-id is required (try --help)")?;
    let labels = labels.ok_or("--labels is required (try --help)")?;
    let out = out.unwrap_or_else(|| format!("models/{model_id}.graph.json"));

    let mut opts = GgufOptions::new(model_id, labels);
    opts.layers = layers;
    opts.activation = activation;
    opts.features = features;
    opts.description = description;
    opts.architecture = architecture;

    let plan = plan_layers(&file, &opts)?;
    println!(
        "plan: {}",
        plan.iter()
            .map(|l| match &l.bias {
                Some(b) => format!("{} ({}+{})", l.weight, l.weight, b),
                None => format!("{} (+zeros)", l.weight),
            })
            .collect::<Vec<_>>()
            .join(" → ")
    );

    let imported = convert(&mut file, &opts)?;
    let spec = &imported.spec;
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
    println!(
        "runtime layers: {} ({} features, {} labels)",
        chain.join(", "),
        spec.features,
        spec.labels.len()
    );
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
