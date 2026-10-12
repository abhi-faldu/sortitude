#!/bin/bash
# Download and extract edgetpu_compiler WITHOUT sudo (dpkg-deb -x into a prefix).
set -e
WORK="$HOME/edgetpu_work"
PREFIX="$WORK/install"
mkdir -p "$PREFIX"
cd "$WORK"
REPO="https://packages.cloud.google.com/apt"

echo "=== fetching package index ==="
curl -fsSL "$REPO/dists/coral-edgetpu-stable/main/binary-amd64/Packages" -o Packages

DEB=$(awk '/^Package: edgetpu-compiler$/{f=1} f&&/^Filename:/{print $2; exit}' Packages)
echo "compiler deb: $DEB"
[ -n "$DEB" ] || { echo "ERROR: could not find edgetpu-compiler in index"; exit 1; }

echo "=== downloading compiler .deb ==="
curl -fsSL "$REPO/$DEB" -o compiler.deb
ls -la compiler.deb

echo "=== extracting (no sudo) ==="
dpkg-deb -x compiler.deb "$PREFIX"
BIN=$(find "$PREFIX" -name edgetpu_compiler -type f | head -1)
echo "binary: $BIN"
[ -n "$BIN" ] || { echo "ERROR: binary not found after extract"; exit 1; }
chmod +x "$BIN"

echo "=== trying to run it ==="
LIBDIR=$(dirname "$(find "$PREFIX" -name 'libedgetpu*' 2>/dev/null | head -1)")
LD_LIBRARY_PATH="$LIBDIR:$PREFIX/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" "$BIN" --version || {
  echo "--- direct run failed; missing shared libs are listed below ---"
  ldd "$BIN" | grep -i 'not found' || true
}
echo "=== done ==="
