#!/bin/bash
echo "=== USB devices now (lsusb-style via sysfs) ==="
for p in /sys/bus/usb/devices/*/product; do
  [ -f "$p" ] && echo "  - $(cat "$p")"
done
echo "=== /dev/video* ==="
ls -l /dev/video* 2>/dev/null || echo "none"
echo "=== try grabbing ONE frame via OpenCV ==="
python3 - <<'PY'
import cv2, numpy as np
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("RESULT: could NOT open camera index 0")
else:
    ok, frame = cap.read()
    if ok and frame is not None:
        print(f"RESULT: captured frame OK shape={frame.shape} mean_brightness={frame.mean():.1f}")
    else:
        print("RESULT: camera opened but frame read FAILED (no live camera on the bridge)")
cap.release()
PY
echo "=== done"
