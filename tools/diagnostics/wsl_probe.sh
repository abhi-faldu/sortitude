#!/bin/bash
echo "USER=$(whoami)"
echo "ARCH=$(uname -m)"
echo "--- internet test:"
if curl -fsSL -m 10 -o /dev/null https://packages.cloud.google.com/apt/dists/coral-edgetpu-stable/Release 2>/dev/null; then
  echo "internet: OK (reached Google coral apt repo)"
else
  echo "internet: FAILED to reach coral apt repo"
fi
echo "--- tools:"
for t in curl dpkg-deb edgetpu_compiler; do
  if command -v "$t" >/dev/null 2>&1; then echo "$t: present"; else echo "$t: missing"; fi
done
echo "--- done"
