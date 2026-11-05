#!/usr/bin/env bash
# Push AlphaDesk to https://github.com/Bilpes/TradingBot (replaces history
# with this fresh single-commit history — per owner's request).
#
# Usage:
#   GITHUB_TOKEN=<your_token> ./scripts/push_to_github.sh
#
# Token: GitHub -> Settings -> Developer settings -> Fine-grained personal
# access tokens -> generate for repository "Bilpes/TradingBot" with
# Contents: Read and write. Revoke it after pushing if you prefer.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN env var first}"

# Fresh unzip (no .git)? Create the single-commit history first.
if [ ! -d .git ]; then
    echo ">> no .git found — initializing fresh single-commit history…"
    git init -b main
    git config core.fileMode false
    git add -A
    git commit -m "AlphaDesk v1.0 — multi-agent trading desk for Indian equities (Dhan)"
fi

git remote remove origin 2>/dev/null || true
git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/Bilpes/TradingBot.git"

echo ">> force-pushing fresh history to Bilpes/TradingBot (main)…"
git push --force origin main

echo ">> clearing token from remote url…"
git remote set-url origin https://github.com/Bilpes/TradingBot.git
echo ">> done → https://github.com/Bilpes/TradingBot"
