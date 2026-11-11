//! Float → Q16.16 conversion.
//!
//! The chain never sees floating point: parameters are converted once, here,
//! and the resulting integers are what every node executes. Rounding is
//! round-half-away-from-zero, which is symmetric in sign and therefore does not
//! bias a weight toward zero.

/// Q16.16 one, mirrored from the runtime.
pub const ONE: f64 = 65_536.0;

/// Largest magnitude a float parameter may have: Q16.16 stores ±32768.
pub const MAX_MAGNITUDE: f64 = 32_767.999_984_741_21;

/// Convert one parameter. Values outside the representable range are an error
/// rather than a silent clamp: a clamped weight changes the model's answers, and
/// a model that is quietly not the one the customer trained is worse than a
/// failed import.
pub fn to_q16(value: f32, what: &str) -> Result<i32, String> {
    if !value.is_finite() {
        return Err(format!("{what} is not a finite number"));
    }
    let scaled = value as f64 * ONE;
    if !(-MAX_MAGNITUDE * ONE..=MAX_MAGNITUDE * ONE).contains(&scaled) {
        return Err(format!(
            "{what} = {value} exceeds the Q16.16 range (|weight| < 32768); \
             rescale the layer before importing"
        ));
    }
    Ok(scaled.round() as i32)
}

/// Worst-case error introduced by quantising one weight, as a bound on the
/// absolute error of a single dot product (each term is off by at most 2⁻¹⁷).
pub fn dot_product_error_bound(terms: usize) -> f32 {
    terms as f32 / (2.0 * ONE as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantisation_is_symmetric_and_exact_for_representable_values() {
        assert_eq!(to_q16(1.0, "w").unwrap(), 65_536);
        assert_eq!(to_q16(-1.0, "w").unwrap(), -65_536);
        assert_eq!(to_q16(0.5, "w").unwrap(), 32_768);
        assert_eq!(to_q16(-0.5, "w").unwrap(), -32_768);
        // 1/65536 is the smallest step; half a step rounds away from zero.
        assert_eq!(to_q16(1.0 / 65_536.0, "w").unwrap(), 1);
        assert_eq!(to_q16(-1.0 / 65_536.0, "w").unwrap(), -1);
    }

    #[test]
    fn out_of_range_and_non_finite_values_are_rejected() {
        assert!(to_q16(1.0e6, "w").is_err());
        assert!(to_q16(f32::NAN, "w").is_err());
        assert!(to_q16(f32::INFINITY, "w").is_err());
    }
}
