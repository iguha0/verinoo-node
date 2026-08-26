# ADR 002: Malachite BFT Behind LeaderPolicy Trait

## Status: Design — implementation deferred until rotation is battle-tested

## Context
Round-robin leader selection works but provides no finality guarantees,
no Byzantine fault tolerance, and no slashing for equivocation. Malachite
(informal systems' Rust Tendermint) provides all three.

## Why defer
1. Rotation must be battle-tested first — BFT on top of buggy state
   transitions just makes bugs harder to find
2. Malachite brings its own P2P, storage, and timeout management which
   may conflict with our libp2p + RocksDB stack
3. The LeaderPolicy trait boundary was designed specifically to allow
   swapping without touching engine/network/storage code

## Integration plan

### Phase 1: Abstraction (done)
`LeaderPolicy` trait with `leader_at(height, validators) -> Option<String>`.
RoundRobin implements it today; MalachiteDriver will implement it later.

### Phase 2: Malachite Driver
```rust
pub struct MalachitePolicy {
    consensus: malachite::Consensus,
    // ...
}

impl LeaderPolicy for MalachitePolicy {
    fn leader_at(&self, height: u64, validators: &[String]) -> Option<String> {
        self.consensus.proposer(height, validators)
    }
}
```

### Phase 3: Block validation changes
- Blocks need precommits from 2/3+ of validators (not just producer sig)
- Equivocation detection: same height+round different hashes → slash
- Timeout handling: if leader doesn't produce in time, next round starts

### What stays the same
- Engine execution logic (apply_block, validate_tx)
- State model and storage
- Gas market and fee distribution
- ZK verification pipeline

### Dependencies
- `malachite-consensus` crate (informal systems)
- Additional network round-trips per block (prevote/precommit phases)
- Block time increases from 10s to ~15-20s due to consensus rounds
