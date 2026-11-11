//! Fit the reference classifier from the committed corpus.
//!
//! `cargo run --release -p verinoo-model --bin train`
//!
//! The fit is closed-form (multinomial Naive Bayes log-odds over hashed
//! unigrams), so it is deterministic and has no learning-rate to tune: every
//! run on every machine reproduces the same weights. The result is quantised
//! to Q16.16 and written to `weights/intent-v1.json`, which is committed — the
//! node never trains at runtime.

use std::path::PathBuf;

use verinoo_model::{bucket, tokens, CLASSES, FEATURES, LABELS, MODEL_ID, ONE};

#[derive(serde::Deserialize)]
struct Row {
    text: String,
    label: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let raw = std::fs::read_to_string(manifest.join("data/intents.jsonl"))?;

    let mut rows: Vec<(String, usize)> = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = serde_json::from_str(line)?;
        let class = LABELS
            .iter()
            .position(|l| *l == row.label)
            .ok_or_else(|| format!("unknown label `{}`", row.label))?;
        rows.push((row.text, class));
    }
    println!(
        "corpus: {} examples, {} features, {} classes",
        rows.len(),
        FEATURES,
        CLASSES
    );

    // ---- fit ----
    let alpha = 0.2f64; // Laplace smoothing
    let mut class_docs = [0f64; CLASSES];
    let mut class_tokens = [0f64; CLASSES];
    let mut feat_class = vec![vec![0f64; FEATURES]; CLASSES];
    let mut feat_all = vec![0f64; FEATURES];
    let mut total_tokens = 0f64;

    for (text, y) in &rows {
        class_docs[*y] += 1.0;
        for tok in tokens(text) {
            let f = bucket(&tok);
            class_tokens[*y] += 1.0;
            feat_class[*y][f] += 1.0;
            feat_all[f] += 1.0;
            total_tokens += 1.0;
        }
    }

    let n = rows.len() as f64;
    let mut w = vec![vec![0f64; FEATURES]; CLASSES];
    let mut b = [0f64; CLASSES];
    for c in 0..CLASSES {
        b[c] = (class_docs[c] / n).ln();
        for f in 0..FEATURES {
            let p_fc = (feat_class[c][f] + alpha) / (class_tokens[c] + alpha * FEATURES as f64);
            let p_f = (feat_all[f] + alpha) / (total_tokens + alpha * FEATURES as f64);
            w[c][f] = (p_fc / p_f).ln();
        }
    }

    // ---- quantise exactly as the node will run it ----
    let q = |v: f64| -> i32 { (v * ONE as f64).round() as i32 };
    let wq: Vec<Vec<i32>> = w
        .iter()
        .map(|row| row.iter().map(|v| q(*v)).collect())
        .collect();
    let bq: Vec<i32> = b.iter().map(|v| q(*v)).collect();

    let predict = |text: &str| -> usize {
        let x: Vec<i32> = verinoo_model::featurize(text);
        let mut best = 0usize;
        let mut best_v = i64::MIN;
        for c in 0..CLASSES {
            let mut acc = bq[c] as i64;
            for f in 0..FEATURES {
                acc += ((x[f] as i64) * (wq[c][f] as i64)) >> 16;
            }
            if acc > best_v {
                best_v = acc;
                best = c;
            }
        }
        best
    };

    // Deterministic holdout: every fifth example of each class.
    let mut seen = [0usize; CLASSES];
    let mut is_holdout = Vec::with_capacity(rows.len());
    for (_, y) in &rows {
        seen[*y] += 1;
        is_holdout.push(seen[*y].is_multiple_of(5));
    }

    let (mut train_ok, mut train_n, mut hold_ok, mut hold_n) = (0, 0, 0, 0);
    let mut confusions: Vec<Vec<usize>> = vec![vec![0; CLASSES]; CLASSES];
    for (i, (text, y)) in rows.iter().enumerate() {
        let pred = predict(text);
        if is_holdout[i] {
            hold_n += 1;
            if pred == *y {
                hold_ok += 1;
            }
        } else {
            train_n += 1;
            if pred == *y {
                train_ok += 1;
            }
        }
        confusions[*y][pred] += 1;
    }
    let train_acc = train_ok as f64 / train_n.max(1) as f64;
    let holdout_acc = hold_ok as f64 / hold_n.max(1) as f64;
    println!(
        "train accuracy:    {:.1}% ({} examples)",
        train_acc * 100.0,
        train_n
    );
    println!(
        "held-out accuracy: {:.1}% ({} examples)",
        holdout_acc * 100.0,
        hold_n
    );
    println!(
        "confusion (row = true, col = predicted, order {:?}):",
        LABELS
    );
    for (i, row) in confusions.iter().enumerate() {
        println!("  {:<9} {:?}", LABELS[i], row);
    }

    if holdout_acc < 0.8 {
        return Err(format!("held-out accuracy too low: {holdout_acc:.3}").into());
    }

    let flat: Vec<i32> = wq.iter().flatten().copied().collect();
    let value = serde_json::json!({
        "modelId": MODEL_ID,
        "architecture": "hash-unigram-nb-2048x4",
        "features": FEATURES,
        "classes": CLASSES,
        "labels": LABELS,
        "w": flat,
        "b": bq,
    });
    let out = manifest.join("weights/intent-v1.json");
    std::fs::write(&out, serde_json::to_string(&value)?)?;
    println!(
        "wrote {} ({} parameters)",
        out.display(),
        flat.len() + bq.len()
    );
    Ok(())
}
