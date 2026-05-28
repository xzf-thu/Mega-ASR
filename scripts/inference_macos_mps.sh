#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" infer.py --mps "$@"
