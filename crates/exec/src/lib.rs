//! Verinoo execution core — validation and state transition logic.

pub mod engine;
pub mod gas;
pub mod leader;

pub mod spec;
pub use engine::{validate_block, Block, BlockContext, BlockHeader, EngineError, Validator};
pub use leader::{LeaderPolicy, RoundRobin};
