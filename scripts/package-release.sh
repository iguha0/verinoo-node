#!/usr/bin/env bash
# Build the verinoo-node release tarball and its SHA-256 checksum file.
#
#   scripts/package-release.sh --version v0.10.0-testnet \
#     --binary target/release/verinoo-node --out dist
#
# Optional:
#   --repo <path>      repository root (default: parent of this script)
#   --platform <name>  assert the target platform when `file` is unavailable
#                      (only x86_64-linux is supported)
#   --skip-strip       do not run strip on the packaged binary
#
# The script validates every input, checks the binary is an x86_64 Linux ELF,
# stages a clean tree with the explicit seven-model allowlist, writes README.txt
# and an internal SHA256SUMS, archives it, and leaves the archive plus
# SHA256SUMS.<version> in --out. It never writes into the repository or follows
# TinyStories/DistilGPT-2 weights into the package.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/package-release.sh --version <version> --binary <path> --out <dir>
       [--repo <path>] [--platform x86_64-linux] [--skip-strip]

  --version        release version, for example v0.10.0-testnet
  --binary         path to the built verinoo-node binary
  --out            output directory for the archive and checksum file
  --repo           repository root (default: parent of this script)
  --platform       assert x86_64-linux when `file` is not available
  --skip-strip     skip stripping the packaged binary
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

VERSION=""
BINARY=""
OUT=""
REPO=""
PLATFORM=""
SKIP_STRIP=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version)
      [ "$#" -ge 2 ] || die "--version needs a value"
      VERSION="$2"
      shift 2
      ;;
    --version=*)
      VERSION="${1#*=}"
      shift
      ;;
    --binary)
      [ "$#" -ge 2 ] || die "--binary needs a path"
      BINARY="$2"
      shift 2
      ;;
    --binary=*)
      BINARY="${1#*=}"
      shift
      ;;
    --out)
      [ "$#" -ge 2 ] || die "--out needs a directory"
      OUT="$2"
      shift 2
      ;;
    --out=*)
      OUT="${1#*=}"
      shift
      ;;
    --repo)
      [ "$#" -ge 2 ] || die "--repo needs a path"
      REPO="$2"
      shift 2
      ;;
    --repo=*)
      REPO="${1#*=}"
      shift
      ;;
    --platform)
      [ "$#" -ge 2 ] || die "--platform needs a value"
      PLATFORM="$2"
      shift 2
      ;;
    --platform=*)
      PLATFORM="${1#*=}"
      shift
      ;;
    --skip-strip)
      SKIP_STRIP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done

[ -n "$VERSION" ] || { usage; die "--version is required"; }
[ -n "$BINARY" ] || { usage; die "--binary is required"; }
[ -n "$OUT" ] || { usage; die "--out is required"; }

case "$VERSION" in
  *[!A-Za-z0-9._-]*) die "invalid version '$VERSION' (use letters, digits, dot, underscore, dash)" ;;
esac

if [ -z "$REPO" ]; then
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
[ -d "$REPO" ] || die "repository not found: $REPO"
REPO="$(cd "$REPO" && pwd)"

[ -f "$BINARY" ] || die "binary not found: $BINARY"
[ -x "$BINARY" ] || die "binary is not executable: $BINARY"
BINARY="$(cd "$(dirname "$BINARY")" && pwd)/$(basename "$BINARY")"

case "$PLATFORM" in
  ""|x86_64-linux) ;;
  *) die "unsupported --platform '$PLATFORM' (only x86_64-linux is supported)" ;;
esac

if command -v file >/dev/null 2>&1; then
  BINARY_DESC="$(file -b "$BINARY" 2>/dev/null || true)"
  [ -n "$BINARY_DESC" ] || die "file could not inspect the binary: $BINARY"
  BINARY_DESC_LC="$(printf '%s' "$BINARY_DESC" | tr '[:upper:]' '[:lower:]')"
  case "$BINARY_DESC_LC" in
    *elf*64-bit*) ;;
    *) die "binary is not a 64-bit ELF: $BINARY_DESC" ;;
  esac
  case "$BINARY_DESC_LC" in
    *x86-64*|*x86_64*|*amd64*) ;;
    *) die "binary is not x86_64: $BINARY_DESC" ;;
  esac
  case "$BINARY_DESC_LC" in
    *mach-o*|*pe32*|*ms-dos*|*arm*|*aarch64*|*riscv*|*powerpc*|*s390*)
      die "binary target is not x86_64 Linux: $BINARY_DESC"
      ;;
  esac
  case "$BINARY_DESC_LC" in
    *linux*) ;;
    *)
      [ "$PLATFORM" = "x86_64-linux" ] ||
        die "could not confirm the ELF targets Linux ($BINARY_DESC); pass --platform x86_64-linux to assert it"
      ;;
  esac
  echo "  platform: x86_64 Linux ($BINARY_DESC)"
else
  [ "$PLATFORM" = "x86_64-linux" ] ||
    die "file command not found; pass --platform x86_64-linux to assert the binary target"
  echo "  platform: x86_64 Linux (asserted by --platform; file not available)"
fi

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

NOTES="$REPO/docs/releases/${VERSION}.md"
if [ ! -f "$NOTES" ]; then
  NOTES="$REPO/docs/releases/${VERSION%-testnet}.md"
fi
[ -f "$NOTES" ] || die "release notes not found: docs/releases/${VERSION}.md (or ${VERSION%-testnet}.md)"
NOTES_BASENAME="$(basename "$NOTES")"

required_docs=(
  "$REPO/docs/snapshots.md"
  "$REPO/docs/policy.md"
  "$REPO/docs/upgrade-v2.md"
  "$NOTES"
)
required_scripts=(
  claims_report.py
  demo-claims.sh
  import_tiny_stories.py
  import_distilgpt2.py
  import_hf_gpt.py
  hf_gpt_common.py
  test_hf_gpt_import.py
  stripe_invoice.py
  test_stripe_invoice.py
  backup.sh
  upgrade-v2.sh
  verify.sh
)

for path in "${required_docs[@]}" "$REPO/LICENSE" "$REPO/deploy/node.env.example"; do
  [ -f "$path" ] || die "required input missing: $path"
done
for name in "${required_scripts[@]}"; do
  [ -f "$REPO/scripts/$name" ] || die "required script missing: scripts/$name"
done
[ -d "$REPO/models" ] || die "required directory missing: models/"
[ -d "$REPO/sdk/typescript" ] || die "required directory missing: sdk/typescript/"
[ -d "$REPO/sdk/python" ] || die "required directory missing: sdk/python/"

PKG_NAME="verinoo-node-${VERSION}-x86_64-linux"
ARCHIVE="${PKG_NAME}.tar.gz"
STAGE="$(mktemp -d "$OUT/.verinoo-package.XXXXXX")"
PKG="$STAGE/$PKG_NAME"
TMP_ARCHIVE=""

cleanup() {
  rm -rf "$STAGE"
  [ -n "${TMP_ARCHIVE:-}" ] && rm -f "$TMP_ARCHIVE"
  return 0
}
trap cleanup EXIT INT TERM

mkdir -p "$PKG/bin" "$PKG/models" "$PKG/scripts" "$PKG/sdk" "$PKG/deploy" \
  "$PKG/docs/releases"

echo "→ staging $PKG_NAME"
install -m 0755 "$BINARY" "$PKG/bin/verinoo-node"

if [ "$SKIP_STRIP" -eq 0 ]; then
  command -v strip >/dev/null 2>&1 || die "strip not found; install binutils or pass --skip-strip"
  if ! strip --strip-unneeded "$PKG/bin/verinoo-node" 2>/dev/null; then
    die "strip --strip-unneeded failed; pass --skip-strip to package without stripping"
  fi
  echo "  stripped bin/verinoo-node"
else
  echo "  --skip-strip: leaving the binary as supplied"
fi

# Explicit allowlist of the small committed models and fixtures. Never follow
# model-card links or package locally imported TinyStories/DistilGPT-2 weights.
required_models=(
  acme-support-v1.graph.json
  claims-triage-v1.graph.json
  claims-triage-v1.test.json
  intent-transformer-v1.json
  intent-transformer-v1.test.json
  tiny-gpt-v1.json
  tiny-gpt-v1.test.json
)
for name in "${required_models[@]}"; do
  [ -f "$REPO/models/$name" ] || die "required model missing: models/$name"
  [ -s "$REPO/models/$name" ] || die "required model is empty: models/$name"
  cp -p "$REPO/models/$name" "$PKG/models/$name"
done
echo "  packaged ${#required_models[@]} model files"

for name in "${required_scripts[@]}"; do
  cp -p "$REPO/scripts/$name" "$PKG/scripts/$name"
done
echo "  packaged ${#required_scripts[@]} scripts"

tar -C "$REPO/sdk" \
  --exclude='node_modules' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  -cf - typescript python | tar -C "$PKG/sdk" -xf -
echo "  packaged TypeScript and Python SDKs"

install -m 0644 "$REPO/deploy/node.env.example" "$PKG/deploy/node.env.example"
install -m 0644 "$REPO/LICENSE" "$PKG/LICENSE"
install -m 0644 "$REPO/docs/snapshots.md" "$PKG/docs/snapshots.md"
install -m 0644 "$REPO/docs/policy.md" "$PKG/docs/policy.md"
install -m 0644 "$REPO/docs/upgrade-v2.md" "$PKG/docs/upgrade-v2.md"
install -m 0644 "$NOTES" "$PKG/docs/releases/$NOTES_BASENAME"

cat <<'README_TEMPLATE' | awk -v version="$VERSION" -v notes="$NOTES_BASENAME" \
  '{ gsub(/@VERSION@/, version); gsub(/@NOTES@/, notes); print }' \
  >"$PKG/README.txt"
verinoo-node @VERSION@ — x86_64 Linux testnet build
====================================================

Contents
--------
  bin/verinoo-node          node, wallet and inference CLI
  models/                   small committed models and fixtures
  scripts/                  claims demo/report, importers, Stripe invoice,
                            backup, verify and protocol-v2 upgrade helper
  sdk/typescript, sdk/python client SDKs (batch and SSE support)
  deploy/node.env.example   environment template
  docs/                     snapshots, policy, upgrade runbook, release notes
  LICENSE                   AGPL-3.0

Quick start
-----------
  VNN_DB=/var/lib/verinoo/data \
  VNN_KEY_SEED="<unique-random-seed>" \
  VNN_VALIDATOR=1 \
  VNN_API_PORT=7100 \
  VNN_P2P_PORT=7101 \
    ./bin/verinoo-node

  curl -fsS http://127.0.0.1:7100/health
  curl -fsS http://127.0.0.1:7100/ready
  curl -fsS http://127.0.0.1:7100/v1/models

Install the binary:
  sudo install -m 0755 bin/verinoo-node /usr/local/bin/verinoo-node
  sudo systemctl restart verinoo-node

Protocol v2 upgrade warning
---------------------------
This build accepts historical v1 blocks and produces v2 blocks. A v0.9.0
binary rejects v2 blocks, so upgrade every validator and replica in one
coordinated window. Do not roll back to v0.9.0 after the first v2 block.
Read docs/upgrade-v2.md before upgrading.

Non-claims
----------
This is a testnet release. It does not provide GPU execution, BFT finality,
automatic failover, trustless snapshots, token-level streaming, or arbitrary
LLaMA/SentencePiece model import. Snapshot signatures, where enabled, are
operator attestations; there is no automatic snapshot sync. Policy is a
deterministic API-boundary filter, not ML moderation. Inputs are encrypted to
the validator set, so validators (including the operator) can decrypt them.

Verification and release notes
------------------------------
  # From the download directory, next to the published archive checksum:
  sha256sum -c SHA256SUMS.@VERSION@

  # After extracting and cd'ing into this directory:
  sha256sum -c SHA256SUMS

  See docs/releases/@NOTES@ for the full notes, evidence and artifact list.

License: AGPL-3.0. See LICENSE.
README_TEMPLATE

echo "→ writing package contents SHA256SUMS"
(
  cd "$PKG"
  find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort |
    while IFS= read -r file; do
      rel="${file#./}"
      printf '%s  %s\n' "$(hash_file "$file")" "$rel"
    done >SHA256SUMS
)

file_count="$(find "$PKG" -type f | wc -l | tr -d ' ')"
size_kb="$(du -sk "$PKG" | awk '{print $1}')"
echo "  staged $file_count files, ${size_kb} KiB"

echo "→ creating $ARCHIVE"
TMP_ARCHIVE="$OUT/.${ARCHIVE}.tmp.$$"
tar -C "$STAGE" -czf "$TMP_ARCHIVE" "$PKG_NAME"
mv "$TMP_ARCHIVE" "$OUT/$ARCHIVE"
TMP_ARCHIVE=""

echo "→ writing SHA256SUMS.$VERSION"
printf '%s  %s\n' "$(hash_file "$OUT/$ARCHIVE")" "$ARCHIVE" \
  >"$OUT/SHA256SUMS.$VERSION"

echo "✓ package ready"
echo "  $OUT/$ARCHIVE"
echo "  $OUT/SHA256SUMS.$VERSION"
cat "$OUT/SHA256SUMS.$VERSION"
