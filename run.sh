#!/usr/bin/env bash
# run.sh — DMS Launcher for macOS
#
# OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES must be set in the SHELL before
# Python starts (not inside Python) because macOS loads the ObjC runtime
# during dylib initialisation, which happens at process startup.
#
# Usage:
#   bash run.sh                  # normal run
#   bash run.sh --no-cnn         # skip CNN, use EAR heuristic
#   bash run.sh --no-car-sim     # skip car simulation window
#   bash run.sh --debug          # verbose frame-by-frame output

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[DMS] Starting with OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES"
exec python3 "$DIR/main.py" "$@"
