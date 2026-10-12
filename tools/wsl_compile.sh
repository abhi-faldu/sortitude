#!/bin/bash
set -e
WORK="$HOME/edgetpu_work"
PREFIX="$WORK/install"
BIN="$PREFIX/usr/bin/edgetpu_compiler"
# Repo models/ folder, resolved relative to this script (works from WSL on a Windows checkout)
MODELS="$(cd "$(dirname "$0")/../models" && pwd)"
SRC="$MODELS/sortitude_int8.tflite"
OUTDIR="$WORK/compiled"
LIBDIR=$(dirname "$(find "$PREFIX" -name 'libedgetpu*' 2>/dev/null | head -1)")

mkdir -p "$OUTDIR"
cp "$SRC" "$OUTDIR/"
cd "$OUTDIR"

echo "=== compiling sortitude_int8.tflite for Edge TPU ==="
LD_LIBRARY_PATH="$LIBDIR:$PREFIX/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
  "$BIN" -s sortitude_int8.tflite

echo "=== outputs ==="
ls -la
echo "=== copying compiled model back to models/ ==="
cp -f sortitude_int8_edgetpu.tflite "$MODELS/" 2>/dev/null && echo "copied edgetpu.tflite" || echo "WARN: no edgetpu.tflite produced"
cp -f sortitude_int8_edgetpu.log "$MODELS/" 2>/dev/null || true
echo "=== done ==="
