#!/bin/bash
echo "=== video devices:"
ls -l /dev/video* 2>/dev/null || echo "NONE - webcam not detected on board"
echo "=== v4l2 devices (if v4l-utils present):"
which v4l2-ctl >/dev/null 2>&1 && v4l2-ctl --list-devices 2>/dev/null || echo "(v4l2-ctl not installed)"
echo "=== USB products:"
for p in /sys/bus/usb/devices/*/product; do
  [ -f "$p" ] && echo "  - $(cat "$p")"
done
echo "=== done"
