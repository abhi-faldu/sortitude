#!/bin/bash
echo "=== is dashboard.py running? ==="
if pgrep -f dashboard.py >/dev/null 2>&1; then echo "RUNNING (pids: $(pgrep -f dashboard.py | tr '\n' ' '))"; else echo "not running"; fi
echo "=== who holds /dev/video1? ==="
fuser /dev/video1 2>/dev/null && echo "(held)" || echo "(free or fuser missing)"
echo "=== foreground run for 6s to capture errors ==="
cd ~/sortitude
pkill -f dashboard.py 2>/dev/null; sleep 1
timeout 6 python3 dashboard.py --model sortitude_int8_edgetpu.tflite --labels labels.txt --camera 1 --port 8000 2>&1 | head -30
echo "=== exit; (timeout=124 means it was running fine) rc=${PIPESTATUS[0]} ==="
