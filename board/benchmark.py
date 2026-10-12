#!/usr/bin/env python3
"""Benchmark Sortitude int8 model: CPU (tflite_runtime) vs Edge TPU (pycoral).

Loads each model once, warms up, then times N inferences on a test image.
Reports avg / min / max latency and the top prediction for each backend.
"""
import argparse
import time

import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite
from pycoral.utils.edgetpu import make_interpreter


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def prep(img_path, size):
    im = Image.open(img_path).convert("RGB").resize(size)
    return np.asarray(im, dtype=np.uint8)


def set_input(interp, image):
    inp = interp.get_input_details()[0]
    t = interp.tensor(inp["index"])()[0]
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        q = image.astype(np.float32) / scale + zp
        t[:, :, :] = np.clip(np.round(q), -128, 127).astype(np.int8)
    else:
        t[:, :, :] = image.astype(inp["dtype"])


def predict(interp, labels):
    out = interp.get_output_details()[0]
    s = interp.get_tensor(out["index"])[0]
    if out["dtype"] == np.int8:
        scale, zp = out["quantization"]
        s = (s.astype(np.float32) - zp) * scale
    s = s.astype(np.float32)
    if s.min() < 0 or s.max() > 1 or abs(s.sum() - 1) > 1e-2:
        e = np.exp(s - s.max()); s = e / e.sum()
    i = int(np.argmax(s))
    return labels[i], float(s[i])


def bench(name, interp, image, labels, n):
    in_det = interp.get_input_details()[0]
    _, h, w, _ = in_det["shape"]
    # warmup (first TPU invoke loads the model onto the device)
    set_input(interp, image); interp.invoke()
    times = []
    for _ in range(n):
        set_input(interp, image)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    label, conf = predict(interp, labels)
    times = np.array(times)
    print(f"[{name}] {n} runs | avg {times.mean():6.2f} ms | "
          f"min {times.min():6.2f} | max {times.max():6.2f} | "
          f"pred: {label} ({conf*100:.1f}%)")
    return times.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-model", required=True)
    ap.add_argument("--tpu-model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("-n", type=int, default=50)
    args = ap.parse_args()

    labels = load_labels(args.labels)

    cpu = tflite.Interpreter(model_path=args.cpu_model)
    cpu.allocate_tensors()
    in_det = cpu.get_input_details()[0]
    _, h, w, _ = in_det["shape"]
    image = prep(args.image, (int(w), int(h)))

    print(f"Image {args.image} -> input ({w}x{h}), {len(labels)} classes, n={args.n}\n")
    cpu_ms = bench("CPU    ", cpu, image, labels, args.n)

    tpu = make_interpreter(args.tpu_model)
    tpu.allocate_tensors()
    tpu_ms = bench("EdgeTPU", tpu, image, labels, args.n)

    print(f"\nSpeedup: CPU {cpu_ms:.2f} ms -> TPU {tpu_ms:.2f} ms "
          f"= {cpu_ms/tpu_ms:.1f}x faster")


if __name__ == "__main__":
    main()
