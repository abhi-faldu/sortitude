#!/usr/bin/env python3
"""
Sortitude — live waste classification demo for the Google Coral Dev Board.

Runs an Edge-TPU-compiled int8 TFLite classifier over the webcam feed (or a
single image) and overlays the predicted waste category and confidence.

Designed to run ON the Coral Dev Board (Mendel Linux) with the Edge TPU runtime.

Setup on the board:
    sudo apt-get install python3-pycoral python3-opencv
    # or: pip3 install pycoral opencv-python

Files needed alongside this script:
    model_int8_edgetpu.tflite   (Edge-TPU compiled model)
    labels.txt                  (one class name per line)

Usage:
    # Live webcam (default)
    python3 demo.py --model model_int8_edgetpu.tflite --labels labels.txt

    # Classify a single image and exit (good for testing without a camera)
    python3 demo.py --model model_int8_edgetpu.tflite --labels labels.txt \
                    --image sample.jpg

    # Use a non-compiled CPU int8 model (no Edge TPU) for a laptop dry-run
    python3 demo.py --model ../sortitude_int8.tflite --labels labels.txt --cpu
"""

import argparse
import time

import cv2
import numpy as np

# pycoral is the standard Coral API. We fall back to plain tflite_runtime if it
# is missing (e.g. running a CPU dry-run on a laptop without pycoral installed).
try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters import common
    _HAVE_PYCORAL = True
except ImportError:
    _HAVE_PYCORAL = False
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite  # last-resort fallback


def load_labels(path):
    """Read labels.txt -> list of class names, one per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def make_cpu_interpreter(model_path):
    """Build a plain (non-Edge-TPU) interpreter for laptop dry-runs.

    Import tflite_runtime here so --cpu works even when pycoral IS installed
    (e.g. on the Coral board itself), where the module-level fallback import
    never runs.
    """
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def build_interpreter(model_path, use_cpu):
    """Return (interpreter, input_size (w, h)).

    Prefers the Edge TPU via pycoral; falls back to CPU when --cpu is set or
    pycoral is unavailable.
    """
    if _HAVE_PYCORAL and not use_cpu:
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        size = common.input_size(interpreter)  # (width, height)
        return interpreter, size

    interpreter = make_cpu_interpreter(model_path)
    inp = interpreter.get_input_details()[0]
    _, h, w, _ = inp["shape"]
    return interpreter, (int(w), int(h))


def set_input_tensor(interpreter, image):
    """Copy an HxWx3 uint8 image into the interpreter input tensor.

    The model bakes MobileNetV2 preprocessing into its graph, so it expects raw
    [0,255] pixels. For the int8 model we requantize using the model's own input
    scale/zero_point (same mapping as the training-notebook benchmark) rather
    than a hardcoded offset, so it stays correct if the quant params change.
    """
    inp = interpreter.get_input_details()[0]
    tensor = interpreter.tensor(inp["index"])()[0]
    if inp["dtype"] == np.int8:
        scale, zero_point = inp["quantization"]
        q = image.astype(np.float32) / scale + zero_point
        tensor[:, :, :] = np.clip(np.round(q), -128, 127).astype(np.int8)
    else:
        tensor[:, :, :] = image.astype(inp["dtype"])


def classify(interpreter, image, top_k=1):
    """Run inference, return list of (class_index, score) sorted desc."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    out = interpreter.get_output_details()[0]
    scores = interpreter.get_tensor(out["index"])[0]

    # Dequantize int8 logits/probabilities back to floats for a readable score.
    if out["dtype"] == np.int8:
        scale, zero_point = out["quantization"]
        scores = (scores.astype(np.float32) - zero_point) * scale

    # The model exports raw logits (no final softmax), so values aren't in
    # [0,1]. Apply softmax when they don't already look like a probability
    # distribution, giving a sensible confidence for display.
    scores = scores.astype(np.float32)
    if scores.min() < 0.0 or scores.max() > 1.0 or abs(scores.sum() - 1.0) > 1e-2:
        e = np.exp(scores - scores.max())
        scores = e / e.sum()

    top = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top]


def preprocess(frame_bgr, size):
    """BGR camera frame -> RGB, resized to the model's input size."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, size)  # size is (width, height) — matches cv2 order


def draw_overlay(frame, label, score, fps):
    """Draw the prediction banner and FPS onto the BGR frame in place."""
    text = f"{label}: {score * 100:.1f}%"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} FPS", (frame.shape[1] - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def run_image(interpreter, size, labels, image_path):
    """Classify one image, print the result, and exit."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")
    inp = preprocess(frame, size)
    t0 = time.perf_counter()
    (idx, score), = classify(interpreter, inp, top_k=1)
    ms = (time.perf_counter() - t0) * 1000
    print(f"Prediction: {labels[idx]}  ({score * 100:.1f}%)  [{ms:.1f} ms]")


def run_webcam(interpreter, size, labels, camera_index):
    """Live classification loop over the webcam. Press 'q' to quit."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}")

    print("Running live classification — press 'q' in the window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; stopping.")
                break

            inp = preprocess(frame, size)
            t0 = time.perf_counter()
            (idx, score), = classify(interpreter, inp, top_k=1)
            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)

            draw_overlay(frame, labels[idx], score, fps)
            cv2.imshow("Sortitude — Waste Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Sortitude waste classifier demo")
    ap.add_argument("--model", required=True, help="Path to .tflite model")
    ap.add_argument("--labels", required=True, help="Path to labels.txt")
    ap.add_argument("--image", help="Classify a single image instead of webcam")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU interpreter (skip Edge TPU)")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    interpreter, size = build_interpreter(args.model, args.cpu)
    backend = "Edge TPU" if (_HAVE_PYCORAL and not args.cpu) else "CPU"
    print(f"Loaded {args.model} on {backend}; input size {size}; "
          f"{len(labels)} classes.")

    if args.image:
        run_image(interpreter, size, labels, args.image)
    else:
        run_webcam(interpreter, size, labels, args.camera)


if __name__ == "__main__":
    main()
