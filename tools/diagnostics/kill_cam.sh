#!/bin/bash
pkill -9 -f cam_test.sh 2>/dev/null
pkill -9 -f benchmark.py 2>/dev/null
pkill -9 python3 2>/dev/null
sleep 1
if pgrep -f cam_test.sh >/dev/null 2>&1; then echo "SCRIPT: still running"; else echo "SCRIPT: gone"; fi
if pgrep python3 >/dev/null 2>&1; then echo "PYTHON: still running"; else echo "PYTHON: gone"; fi
