#!/bin/bash
echo "=== python3:"; python3 --version
echo "=== packages:"
python3 - <<'PY'
import importlib.util as u
for m in ["numpy","cv2","tflite_runtime","pycoral","PIL","tensorflow"]:
    print(f"{m:14s} {'OK' if u.find_spec(m) else 'MISSING'}")
PY
echo "=== edgetpu device:"; ls -l /dev/apex_0
echo "=== free space on /home:"; df -h /home
echo "=== done"
