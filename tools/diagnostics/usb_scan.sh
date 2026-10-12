#!/bin/bash
echo "=== All USB devices (vendor:product  product-name) ==="
for d in /sys/bus/usb/devices/*; do
  if [ -f "$d/idVendor" ]; then
    vid=$(cat "$d/idVendor"); pid=$(cat "$d/idProduct")
    prod=$(cat "$d/product" 2>/dev/null)
    man=$(cat "$d/manufacturer" 2>/dev/null)
    cls=$(cat "$d/bDeviceClass" 2>/dev/null)
    echo "  ${vid}:${pid}  class=${cls}  ${man} ${prod}"
  fi
done
echo
echo "=== /dev/video* ==="
ls -l /dev/video* 2>/dev/null || echo "none"
echo
echo "=== last 40 USB / video kernel log lines ==="
dmesg 2>/dev/null | grep -iE 'usb|video|uvc|camera|over-current|cannot enable' | tail -40 || echo "(dmesg not readable as mendel; trying sudo-free /var/log)"
echo
echo "=== uvcvideo module loaded? ==="
lsmod 2>/dev/null | grep -i uvc || echo "uvcvideo NOT loaded"
echo "=== done"
