# ADR 001: Two-Party Trusted Setup Ceremony

## Status: Implemented (testnet-grade)

## Context
Groth16 requires a structured reference string (SRS) generated via powers-of-tau.
A single-party setup means users must trust that party destroyed their toxic waste.
For testnet, a two-phase ceremony where both contributions happen on separate
machines is sufficient. Production mainnet would need external participants.

## Implementation

### Phase 1: Powers of Tau
```bash
snarkjs powersoftau new bn128 14 pot14_0000.ptau    # Party 1 (organizer)
snarkjs powersoftau contribute pot14_0000.ptau pot14_0001.ptau \
  --name="Party1" -e="random-entropy-1"
# File transferred to Party 2 (different machine)
snarkjs powersoftau contribute pot14_0001.ptau pot14_0010.ptau \
  --name="Party2" -e="random-entropy-2"
snarkjs powersoftau prepare phase2 pot14_0010.ptau pot14_final.ptau
```

### Phase 2: Per-Circuit Setup
Each circuit runs groth16 setup against the shared ptau, producing
circuit-specific proving (zkey) and verification (vkey) keys.

### Verification
After ceremony, each participant verifies their contribution is included:
```bash
snarkjs powersoftau verify pot14_final.ptau
```

### What this proves
The final keys can only be computed if ALL parties collaborated. As long as
ONE party destroyed their entropy, the keys are secure.

### What this does NOT prove
Participants can't verify what the other party actually did locally.
Production ceremonies use hardware attestation or MPC protocols.

## Current status
Testnet uses a two-machine ceremony (Mac + VPS). Both operated by same person.
Sufficient for research demonstration; NOT for production value.
