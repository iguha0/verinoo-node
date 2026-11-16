//! Per-task decoder sampling parameters.
//!
//! The object is embedded in `submitInference` data, so canonical transaction
//! encoding commits it to the task id without a separate commitment field.

use serde::{Deserialize, Serialize};

pub const MAX_TEMPERATURE_MILLI: u32 = 100_000;
pub const MAX_TOP_K: u32 = 1_000_000;
pub const MAX_TOP_P_MILLI: u32 = 1_000;

/// `temperatureMilli: 0` is greedy; `topK: 0` and `topPMilli: 0` disable the
/// respective filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SamplingSpec {
    #[serde(default)]
    pub temperature_milli: u32,
    #[serde(default)]
    pub top_k: u32,
    #[serde(default)]
    pub top_p_milli: u32,
    #[serde(default)]
    pub seed: u64,
}

impl SamplingSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature_milli > MAX_TEMPERATURE_MILLI {
            return Err(format!(
                "sampling.temperatureMilli must be at most {MAX_TEMPERATURE_MILLI}"
            ));
        }
        if self.top_p_milli > MAX_TOP_P_MILLI {
            return Err(format!(
                "sampling.topPMilli must be at most {MAX_TOP_P_MILLI}"
            ));
        }
        if self.top_k > MAX_TOP_K {
            return Err(format!("sampling.topK must be at most {MAX_TOP_K}"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_uses_camel_case_and_defaults() {
        let spec: SamplingSpec = serde_json::from_value(serde_json::json!({
            "temperatureMilli": 800,
            "topK": 40,
            "topPMilli": 900,
            "seed": 18446744073709551615u64
        }))
        .unwrap();
        assert_eq!(spec.temperature_milli, 800);
        assert_eq!(spec.top_k, 40);
        assert_eq!(spec.top_p_milli, 900);
        assert_eq!(spec.seed, u64::MAX);

        let defaulted: SamplingSpec =
            serde_json::from_value(serde_json::json!({"seed": 7})).unwrap();
        assert_eq!(
            defaulted,
            SamplingSpec {
                seed: 7,
                ..Default::default()
            }
        );

        let value = serde_json::to_value(spec).unwrap();
        assert_eq!(value["temperatureMilli"], 800);
        assert_eq!(value["topK"], 40);
        assert_eq!(value["topPMilli"], 900);
        assert_eq!(value["seed"], u64::MAX);
    }

    #[test]
    fn rejects_malformed_and_unexpected_fields() {
        assert!(serde_json::from_value::<SamplingSpec>(serde_json::json!(42)).is_err());
        assert!(serde_json::from_value::<SamplingSpec>(
            serde_json::json!({"temperatureMilli": -1})
        )
        .is_err());
        assert!(serde_json::from_value::<SamplingSpec>(serde_json::json!({"seed": 1.5})).is_err());
        assert!(
            serde_json::from_value::<SamplingSpec>(serde_json::json!({"temperature": 800}))
                .is_err()
        );
    }

    #[test]
    fn validates_ranges() {
        SamplingSpec {
            temperature_milli: MAX_TEMPERATURE_MILLI,
            top_k: MAX_TOP_K,
            top_p_milli: MAX_TOP_P_MILLI,
            seed: u64::MAX,
        }
        .validate()
        .unwrap();

        let bad_temperature = SamplingSpec {
            temperature_milli: MAX_TEMPERATURE_MILLI + 1,
            ..Default::default()
        };
        assert!(bad_temperature
            .validate()
            .unwrap_err()
            .contains("temperatureMilli"));

        let bad_top_k = SamplingSpec {
            top_k: MAX_TOP_K + 1,
            ..Default::default()
        };
        assert!(bad_top_k.validate().unwrap_err().contains("topK"));

        let bad_top_p = SamplingSpec {
            top_p_milli: MAX_TOP_P_MILLI + 1,
            ..Default::default()
        };
        assert!(bad_top_p.validate().unwrap_err().contains("topPMilli"));
    }
}
