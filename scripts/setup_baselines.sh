#!/usr/bin/env bash
# Create the isolated environment of every baseline system (official releases).
# CAESURA and Nirvana are cloned into .baselines/src at the evaluated commits.
# Set PYTHON to a Python 3.12 interpreter if python3.12 is not on PATH.
mkdir -p .baselines/src
set -euo pipefail
cd "$(dirname "$0")/.."
for s in caesura lotus thalamusdb palimpzest nirvana; do
    bash "scripts/setup/$s.sh"
done
