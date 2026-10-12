#!/bin/bash
cd ~/sortitude || exit 1
pkill -9 -f dashboard.py 2>/dev/null
sleep 2
# -u = unbuffered so the log flushes; detach with nohup
nohup python3 -u dashboard.py --model sortitude_int8_edgetpu.tflite --labels labels.txt --camera 1 --port 8000 > ~/sortitude/dashboard.log 2>&1 &
echo "launched pid $!"
# camera init can take several seconds; poll up to ~20s for the port
for i in $(seq 1 20); do
  if curl -fsS -m 2 -o /dev/null http://127.0.0.1:8000/ 2>/dev/null; then
    echo "UP after ${i}s"; break
  fi
  sleep 1
done
echo "=== dashboard.log ==="
cat ~/sortitude/dashboard.log
echo "=== self state test ==="
curl -fsS -m 15 http://127.0.0.1:8000/state | head -c 200
echo
echo "=== end ==="
