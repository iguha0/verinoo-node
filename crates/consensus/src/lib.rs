//! Signed BFT message types, verification, and the pure round/view-change
//! state machine.
//!
//! The crate defines proposals, prevotes, precommits, commit certificates,
//! equivocation evidence, signature/quorum checks, and two pure machines:
//!
//! - [`Pacemaker`] owns `(height, round)` and the deterministic
//!   `base * 1.5^round` timeout backoff capped at [`R_MAX`]. It has no wall
//!   clock and emits [`Schedule`] values for the node's timer actor.
//! - [`ConsensusState`] owns round/lock/vote state. It accepts round-leader
//!   proposals, applies the Tendermint lock rule, emits vote intentions and
//!   `StartRound` actions, and finalizes on a precommit quorum.
//!
//! See `docs/adr/004-bft-finality.md`, sections "Messages and phases",
//! "Lock and unlock rules", and "Wire format and gossipsub topics".
//!
//! # Remaining node work (not in this crate)
//!
//! This is the pure core only; it does not by itself provide live liveness.
//! The node must still:
//!
//! - gossip proposal envelopes and route them to [`ConsensusState::on_proposal`];
//! - run a timer actor that arms/cancels from [`Schedule`] or
//!   [`Action::StartRound`] and calls [`ConsensusState::on_timeout`] when a
//!   round expires;
//! - persist and restore own votes per `(height, round, kind)` via
//!   [`ConsensusState::restore_own_vote`] before broadcasting;
//! - gate head advancement on [`Action::Finalized`] / a verified commit
//!   certificate (commit gating) instead of applying a produced block
//!   directly;
//! - wire round-aware proposal validation and peer/rate limits.
//!
//! Until the node does those, an offline round-0 leader still stalls the live
//! network even though this crate can move rounds in-process.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use verinoo_types::canonical::canonical_json;

pub const VERSION: u8 = 1;
pub const VOTE_DOMAIN: &[u8] = b"verinoo/vote/v1\n";
pub const PROPOSAL_DOMAIN: &[u8] = b"verinoo/proposal/v1\n";

mod pacemaker;
mod state;
pub use pacemaker::*;
pub use state::*;

pub fn quorum(n: usize) -> usize {
    n * 2 / 3 + 1
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("unsupported consensus message version {0}")]
    UnsupportedVersion(u8),
    #[error("empty chain id")]
    EmptyChainId,
    #[error("chain id mismatch: expected {expected}, got {actual}")]
    ChainIdMismatch { expected: String, actual: String },
    #[error("unknown validator {0}")]
    UnknownValidator(String),
    #[error("malformed public key hex")]
    MalformedPublicKey,
    #[error("public key must be 32 bytes")]
    InvalidPublicKeyLength,
    #[error("public key does not derive to validator {0}")]
    AddressMismatch(String),
    #[error("malformed signature hex")]
    MalformedSignature,
    #[error("signature must be 64 bytes")]
    InvalidSignatureLength,
    #[error("invalid signature")]
    InvalidSignature,
    #[error("certificate block hash is nil")]
    NilBlockHash,
    #[error("certificate vote is not bound to the certificate")]
    BindingMismatch,
    #[error("certificate vote is not a precommit")]
    NotPrecommit,
    #[error("duplicate validator {0} in certificate")]
    DuplicateValidator(String),
    #[error("insufficient quorum: got {got}, need {need}")]
    InsufficientQuorum { got: usize, need: usize },
    #[error("evidence votes have the same block hash")]
    NotEquivocation,
    #[error("evidence votes disagree on validator, chain, height, round, or kind")]
    EvidenceMismatch,
    #[error("evidence proposals disagree on proposer, chain, height, or round")]
    ProposalEvidenceMismatch,
    #[error("evidence proposals have the same block hash")]
    NotDoubleProposal,
    #[error(transparent)]
    Canonical(#[from] verinoo_types::canonical::CanonicalError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoteKind {
    Prevote,
    Precommit,
}

impl VoteKind {
    pub fn as_str(self) -> &'static str {
        match self {
            VoteKind::Prevote => "prevote",
            VoteKind::Precommit => "precommit",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Vote {
    pub chain_id: String,
    pub kind: VoteKind,
    pub height: u64,
    pub round: u64,
    pub block_hash: String,
    pub validator: String,
    pub public_key: String,
    pub signature: String,
    pub v: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Proposal {
    pub chain_id: String,
    pub height: u64,
    pub round: u64,
    pub block_hash: String,
    pub proposer: String,
    pub public_key: String,
    pub signature: String,
    pub v: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitCertificate {
    pub chain_id: String,
    pub height: u64,
    pub round: u64,
    pub block_hash: String,
    pub votes: Vec<Vote>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Evidence {
    pub chain_id: String,
    pub validator: String,
    pub height: u64,
    pub round: u64,
    pub kind: VoteKind,
    pub a: Vote,
    pub b: Vote,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProposalEvidence {
    pub chain_id: String,
    pub proposer: String,
    pub height: u64,
    pub round: u64,
    pub a: Proposal,
    pub b: Proposal,
}

pub fn public_key_to_address(public_key_hex: &str) -> Result<String, Error> {
    let bytes = hex::decode(public_key_hex).map_err(|_| Error::MalformedPublicKey)?;
    if bytes.len() != 32 {
        return Err(Error::InvalidPublicKeyLength);
    }
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(format!("ai_{}", hex::encode(&h.finalize()[..20])))
}

fn verify_signature(
    public_key: &str,
    validator: &str,
    signature: &str,
    payload: &[u8],
) -> Result<(), Error> {
    let pk_bytes = hex::decode(public_key).map_err(|_| Error::MalformedPublicKey)?;
    let pk_arr: [u8; 32] = pk_bytes
        .try_into()
        .map_err(|_| Error::InvalidPublicKeyLength)?;
    let key = VerifyingKey::from_bytes(&pk_arr).map_err(|_| Error::MalformedPublicKey)?;
    if public_key_to_address(public_key)? != validator {
        return Err(Error::AddressMismatch(validator.to_string()));
    }
    let sig_bytes = hex::decode(signature).map_err(|_| Error::MalformedSignature)?;
    let sig_arr: [u8; 64] = sig_bytes
        .try_into()
        .map_err(|_| Error::InvalidSignatureLength)?;
    let sig = Signature::from_bytes(&sig_arr);
    key.verify(payload, &sig)
        .map_err(|_| Error::InvalidSignature)
}

fn check_version(v: u8) -> Result<(), Error> {
    if v != VERSION {
        return Err(Error::UnsupportedVersion(v));
    }
    Ok(())
}

fn check_chain(chain_id: &str, expected: Option<&str>) -> Result<(), Error> {
    if chain_id.is_empty() {
        return Err(Error::EmptyChainId);
    }
    if let Some(expected) = expected {
        if chain_id != expected {
            return Err(Error::ChainIdMismatch {
                expected: expected.to_string(),
                actual: chain_id.to_string(),
            });
        }
    }
    Ok(())
}

fn check_known(validator: &str, validators: &[String]) -> Result<(), Error> {
    if !validators.iter().any(|known| known == validator) {
        return Err(Error::UnknownValidator(validator.to_string()));
    }
    Ok(())
}

impl Vote {
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        chain_id: &str,
        kind: VoteKind,
        height: u64,
        round: u64,
        block_hash: &str,
        validator: &str,
        signing_key: &SigningKey,
    ) -> Vote {
        let public_key = hex::encode(signing_key.verifying_key().to_bytes());
        let mut vote = Vote {
            chain_id: chain_id.to_string(),
            kind,
            height,
            round,
            block_hash: block_hash.to_string(),
            validator: validator.to_string(),
            public_key,
            signature: String::new(),
            v: VERSION,
        };
        let payload = vote
            .signing_payload()
            .expect("vote fields are always canonically encodable");
        vote.signature = hex::encode(signing_key.sign(&payload).to_bytes());
        vote
    }

    pub fn signing_payload(&self) -> Result<Vec<u8>, Error> {
        let payload = json!({
            "v": self.v,
            "chainId": self.chain_id,
            "type": self.kind,
            "height": self.height,
            "round": self.round,
            "blockHash": self.block_hash,
        });
        let mut out = VOTE_DOMAIN.to_vec();
        out.extend_from_slice(canonical_json(&payload)?.as_bytes());
        Ok(out)
    }

    pub fn verify(&self, validators: &[String]) -> Result<(), Error> {
        self.verify_inner(None, validators)
    }

    pub fn verify_with_chain(&self, chain_id: &str, validators: &[String]) -> Result<(), Error> {
        self.verify_inner(Some(chain_id), validators)
    }

    fn verify_inner(&self, chain_id: Option<&str>, validators: &[String]) -> Result<(), Error> {
        check_version(self.v)?;
        check_chain(&self.chain_id, chain_id)?;
        check_known(&self.validator, validators)?;
        verify_signature(
            &self.public_key,
            &self.validator,
            &self.signature,
            &self.signing_payload()?,
        )
    }
}

impl Proposal {
    pub fn sign(
        chain_id: &str,
        height: u64,
        round: u64,
        block_hash: &str,
        proposer: &str,
        signing_key: &SigningKey,
    ) -> Proposal {
        let public_key = hex::encode(signing_key.verifying_key().to_bytes());
        let mut proposal = Proposal {
            chain_id: chain_id.to_string(),
            height,
            round,
            block_hash: block_hash.to_string(),
            proposer: proposer.to_string(),
            public_key,
            signature: String::new(),
            v: VERSION,
        };
        let payload = proposal
            .signing_payload()
            .expect("proposal fields are always canonically encodable");
        proposal.signature = hex::encode(signing_key.sign(&payload).to_bytes());
        proposal
    }

    pub fn signing_payload(&self) -> Result<Vec<u8>, Error> {
        let payload = json!({
            "v": self.v,
            "chainId": self.chain_id,
            "height": self.height,
            "round": self.round,
            "blockHash": self.block_hash,
        });
        let mut out = PROPOSAL_DOMAIN.to_vec();
        out.extend_from_slice(canonical_json(&payload)?.as_bytes());
        Ok(out)
    }

    pub fn verify(&self, validators: &[String]) -> Result<(), Error> {
        self.verify_inner(None, validators)
    }

    pub fn verify_with_chain(&self, chain_id: &str, validators: &[String]) -> Result<(), Error> {
        self.verify_inner(Some(chain_id), validators)
    }

    fn verify_inner(&self, chain_id: Option<&str>, validators: &[String]) -> Result<(), Error> {
        check_version(self.v)?;
        check_chain(&self.chain_id, chain_id)?;
        check_known(&self.proposer, validators)?;
        verify_signature(
            &self.public_key,
            &self.proposer,
            &self.signature,
            &self.signing_payload()?,
        )
    }
}

impl CommitCertificate {
    pub fn verify(&self, validators: &[String]) -> Result<(), Error> {
        verify_certificate(self, validators)
    }
}

pub fn verify_certificate(cert: &CommitCertificate, validators: &[String]) -> Result<(), Error> {
    check_chain(&cert.chain_id, None)?;
    if cert.block_hash.is_empty() {
        return Err(Error::NilBlockHash);
    }
    let mut seen = HashSet::new();
    for vote in &cert.votes {
        if vote.kind != VoteKind::Precommit {
            return Err(Error::NotPrecommit);
        }
        if vote.chain_id != cert.chain_id
            || vote.height != cert.height
            || vote.round != cert.round
            || vote.block_hash != cert.block_hash
        {
            return Err(Error::BindingMismatch);
        }
        vote.verify(validators)?;
        if !seen.insert(vote.validator.as_str()) {
            return Err(Error::DuplicateValidator(vote.validator.clone()));
        }
    }
    let need = quorum(validators.len());
    if cert.votes.len() < need {
        return Err(Error::InsufficientQuorum {
            got: cert.votes.len(),
            need,
        });
    }
    Ok(())
}

impl Evidence {
    pub fn new(a: Vote, b: Vote) -> Result<Evidence, Error> {
        if a.chain_id != b.chain_id
            || a.validator != b.validator
            || a.height != b.height
            || a.round != b.round
            || a.kind != b.kind
        {
            return Err(Error::EvidenceMismatch);
        }
        if a.block_hash == b.block_hash {
            return Err(Error::NotEquivocation);
        }
        let (a, b) = if (a.block_hash.as_str(), a.signature.as_str())
            <= (b.block_hash.as_str(), b.signature.as_str())
        {
            (a, b)
        } else {
            (b, a)
        };
        Ok(Evidence {
            chain_id: a.chain_id.clone(),
            validator: a.validator.clone(),
            height: a.height,
            round: a.round,
            kind: a.kind,
            a,
            b,
        })
    }

    pub fn verify(&self, validators: &[String]) -> Result<(), Error> {
        if self.a.chain_id != self.chain_id
            || self.b.chain_id != self.chain_id
            || self.a.validator != self.validator
            || self.b.validator != self.validator
            || self.a.height != self.height
            || self.b.height != self.height
            || self.a.round != self.round
            || self.b.round != self.round
            || self.a.kind != self.kind
            || self.b.kind != self.kind
        {
            return Err(Error::EvidenceMismatch);
        }
        if self.a.block_hash == self.b.block_hash {
            return Err(Error::NotEquivocation);
        }
        self.a.verify(validators)?;
        self.b.verify(validators)?;
        Ok(())
    }

    pub fn id(&self) -> String {
        canonical_id(self)
    }
}

impl ProposalEvidence {
    pub fn new(a: Proposal, b: Proposal) -> Result<ProposalEvidence, Error> {
        if a.chain_id != b.chain_id
            || a.proposer != b.proposer
            || a.height != b.height
            || a.round != b.round
        {
            return Err(Error::ProposalEvidenceMismatch);
        }
        if a.block_hash == b.block_hash {
            return Err(Error::NotDoubleProposal);
        }
        let (a, b) = if (a.block_hash.as_str(), a.signature.as_str())
            <= (b.block_hash.as_str(), b.signature.as_str())
        {
            (a, b)
        } else {
            (b, a)
        };
        Ok(ProposalEvidence {
            chain_id: a.chain_id.clone(),
            proposer: a.proposer.clone(),
            height: a.height,
            round: a.round,
            a,
            b,
        })
    }

    pub fn verify(&self, validators: &[String]) -> Result<(), Error> {
        self.verify_inner(None, validators)
    }

    pub fn verify_with_chain(&self, chain_id: &str, validators: &[String]) -> Result<(), Error> {
        self.verify_inner(Some(chain_id), validators)
    }

    fn verify_inner(&self, chain_id: Option<&str>, validators: &[String]) -> Result<(), Error> {
        check_chain(&self.chain_id, chain_id)?;
        if self.a.chain_id != self.chain_id
            || self.b.chain_id != self.chain_id
            || self.a.proposer != self.proposer
            || self.b.proposer != self.proposer
            || self.a.height != self.height
            || self.b.height != self.height
            || self.a.round != self.round
            || self.b.round != self.round
        {
            return Err(Error::ProposalEvidenceMismatch);
        }
        if self.a.block_hash == self.b.block_hash {
            return Err(Error::NotDoubleProposal);
        }
        self.a.verify_inner(chain_id, validators)?;
        self.b.verify_inner(chain_id, validators)?;
        Ok(())
    }

    pub fn id(&self) -> String {
        canonical_id(self)
    }
}

fn canonical_id<T: Serialize>(value: &T) -> String {
    let value = serde_json::to_value(value).expect("evidence is serializable");
    let canonical =
        canonical_json(&value).expect("evidence fields are always canonically encodable");
    let mut h = Sha256::new();
    h.update(canonical.as_bytes());
    hex::encode(h.finalize())
}
