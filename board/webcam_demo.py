#!/usr/bin/env python3
"""Headless Sortitude live demo for the Coral Dev Board (no display needed).

Captures frames from the USB webcam, classifies each on the Edge TPU, prints
the prediction + latency + FPS, and saves a few annotated snapshots to disk
(which can be pulled back to a laptop to view).
"""
import argparse
import time

import cv2
import numpy as np

from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def set_input(interp, image):
    inp = interp.get_input_details()[0]
    t = interp.tensor(inp["index"])()[0]
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        q = image.astype(np.float32) / scale + zp
        t[:, :, :] = np.clip(np.round(q), -128, 127).astype(np.int8)
    else:
        t[:, :, :] = image.astype(inp["dtype"])


def classify(interp):
    interp.invoke()
    out = interp.get_output_details()[0]
    s = interp.get_tensor(out["index"])[0]
    if out["dtype"] == np.int8:
        scale, zp = out["quantization"]
        s = (s.astype(np.float32) - zp) * scale
    s = s.astype(np.float32)
    if s.min() < 0 or s.max() > 1 or abs(s.sum() - 1) > 1e-2:
        e = np.exp(s - s.max()); s = e / e.sum()
    i = int(np.argmax(s))
    return i, float(s[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--save-dir", default="snaps")
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()

    import os
    os.makedirs(args.save_dir, exist_ok=True)
    labels = load_labels(args.labels)

    interp = make_interpreter(args.model)
    interp.allocate_tensors()
    w, h = common.input_size(interp)
    print(f"Edge TPU model loaded; input {w}x{h}; {len(labels)} classes.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    # let the camera settle
    for _ in range(5):
        cap.read(); time.sleep(0.1)

    saved = 0
    t_start = time.time()
    for n in range(1, args.frames + 1):
        ok, frame = cap.read()
        if not ok:
            print(f"frame {n}: read failed"); continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (w, h))
        set_input(interp, inp)
        t0 = time.perf_counter()
        idx, conf = classify(interp)
        ms = (time.perf_counter() - t0) * 1000
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        label = labels[idx]
        print(f"frame {n:3d}: {label:10s} {conf*100:5.1f}%  {ms:5.2f} ms  ({fps:5.1f} FPS)")

        if n % args.save_every == 0 and saved < 5:
            txt = f"{label}: {conf*100:.0f}%"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            path = os.path.join(args.save_dir, f"snap_{n:03d}_{label}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            print(f"         saved -> {path}")

    dt = time.time() - t_start
    cap.release()
    print(f"\nDone: {args.frames} frames in {dt:.1f}s ({args.frames/dt:.1f} FPS end-to-end), "
          f"{saved} snapshots in {args.save_dir}/")


if __name__ == "__main__":
    main()
