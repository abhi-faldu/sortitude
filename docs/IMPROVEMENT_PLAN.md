# 🔭 Sortitude — Improvement & Scalability Plan

**Audience:** the Sortitude team (Abhi, Jeel, Aatman)
**Purpose:** a shared, detailed plan for the next phase of the project — making detection more reliable, and making the code portable across hardware (a different camera, or a Raspberry Pi instead of the Coral).
**Status:** proposal for discussion. Nothing here changes existing behaviour on the Coral until we choose to act.

---

## 1. TL;DR (read this if nothing else)

We have a working end-to-end edge-AI system: 92.9% model, Edge TPU at 3 ms/frame, live webcam demo, web dashboard. The pipeline is **done**. The next phase is about **robustness and portability**, and it splits into two independent tracks:

| Track | One-line problem | One-line fix | Why it matters |
|---|---|---|---|
| **A. Portability** | The code is welded to the Coral — `webcam_demo.py` literally won't even *start* on a Raspberry Pi. | Add one shared `sortitude_core.py` with a swappable **backend** + **camera** layer, driven by a config file. | This is exactly the "scalable to new hardware / new camera" goal. It's a code-structure change, not new hardware. |
| **B. Detection quality** | The model is forced to pick 1 of 6 bins **every frame**, even when nothing is in view (it called an empty room "paper"). | Add a **"background/empty" class**, a **confidence gate**, and **temporal smoothing**. | This is what makes the live demo feel trustworthy. It does more for *perceived* accuracy than chasing the 92.9% number higher. |

**Recommended first move:** do Track A's core refactor first (it's safe, needs no retraining, no board, and is the literal answer to the scalability ask), then Track B's background class (the biggest reliability win).

---

## 2. Where we are today (honest snapshot)

What works, and what the limitations actually are:

| Area | Status | Limitation to be honest about |
|---|---|---|
| Model accuracy | ✅ 92.87% val (MobileNetV2, transfer learning) | Measured on **TrashNet** — clean single items on white backgrounds. Not measured on real webcam scenes. |
| Compression | ✅ int8, 8.5 MB → 2.6 MB, −3% acc | Pruned variant failed (0%); documented in `PRUNED_MODEL_FIX.md`. |
| Edge TPU | ✅ 3.09 ms, 42× vs CPU, all 67 ops on TPU | Tied to Coral's specific toolchain + `pycoral`. |
| Live demo + dashboard | ✅ ~30 FPS, pure-stdlib web UI | Always outputs a class, even when there's nothing to classify. Confidence saturates to 100%. |
| **Portability** | ❌ Single-target | Hard-coded to Coral + USB-webcam-by-index. A Pi or a different camera needs code edits in several files. |

The point of this plan: turn the two ❌/limitation rows into ✅.

---

## 3. Track A — Hardware portability (the "scalable to any hardware" track)

### 3.1 What's actually wrong right now

The project demonstrates edge AI well, but the code assumes *one specific* edge device. Three concrete coupling points:

1. **Hard dependency on the Coral runtime.**
   `webcam_demo.py` imports the Coral library at the very top of the file:
   ```python
   from pycoral.utils.edgetpu import make_interpreter   # line 14
   ```
   On a Raspberry Pi (or any machine without `pycoral`), **the script crashes on import** before doing anything. It can only ever run on the Coral. `demo.py` is better — it has a CPU fallback — but the fallback logic is ad-hoc.

2. **Duplicated inference logic.**
   The fiddly bits — int8 quantization of the input, dequantizing the output, applying softmax — are **copy-pasted** into `demo.py`, `webcam_demo.py`, `dashboard.py`, and `benchmark.py`. Adding support for one new device means editing the same logic in four places and hoping they stay in sync. That's the real maintainability smell.

3. **Camera is hard-coded.**
   Every script opens the camera as `cv2.VideoCapture(index)`. A USB webcam works that way on both Coral and Pi — but the **Raspberry Pi Camera Module** (the ribbon-cable camera) uses `libcamera`/`picamera2`, *not* a numeric index. So "swap the camera" breaks today.

### 3.2 The fix: one core module, two small interfaces, one config file

Introduce **`sortitude_core.py`** that every script imports. It hides the hardware behind two tiny abstractions:

**(a) A `Classifier` — "give me an image, I'll give you scores," however that happens underneath:**
```python
class Classifier:
    def infer(self, rgb_image) -> np.ndarray:   # returns per-class probabilities
        ...
    @property
    def input_size(self) -> tuple:              # (width, height) the model wants
        ...

def load_classifier(model_path, backend="auto") -> Classifier:
    # auto-detect at runtime:
    #   pycoral + Edge TPU present  -> EdgeTpuBackend      (Coral, or Pi + Coral USB stick)
    #   tflite_runtime present      -> TFLiteCpuBackend    (bare Raspberry Pi)
    #   tensorflow present          -> TensorFlowBackend   (laptop / dev machine)
```

**(b) A `Camera` — "give me the next frame," whatever the source:**
```python
def open_camera(source) -> Camera:
    #   "usb:1"        -> OpenCV VideoCapture(1)     (USB webcam, Coral or Pi)
    #   "picam"        -> picamera2                   (Raspberry Pi ribbon camera)
    #   "file:foo.mp4" -> looped video file          (for testing with no hardware)
```

**(c) One `config.yaml` per device** — the *only* thing that differs between machines:
```yaml
model:  sortitude_int8_edgetpu.tflite
labels: labels.txt
camera: usb:1
threshold: 0.85
```

After this, `demo.py` / `webcam_demo.py` / `dashboard.py` each shrink to ~40 lines of "open camera, loop, draw" — all the hardware-specific and math-specific code lives in one tested place.

### 3.3 Why this directly delivers "scalable to new hardware"

Here's the support matrix the abstraction unlocks. Note how much we get **for free** from models we already built:

| Target hardware | Backend used | Model file | New code needed? |
|---|---|---|---|
| Google Coral Dev Board (today) | Edge TPU via pycoral | `sortitude_int8_edgetpu.tflite` | none |
| **Raspberry Pi + Coral USB Accelerator** | Edge TPU via pycoral (USB) | **same** `_edgetpu.tflite` | **none — identical code path** |
| Raspberry Pi (CPU only) | `tflite_runtime` | `sortitude_int8.tflite` (already built) | none |
| Laptop / dev machine | `tensorflow.lite` | `sortitude_int8.tflite` | none |
| RPi 5 + Hailo AI Kit, or Jetson | new backend class | needs its own compile step | ~1 class |

**The headline for the report/demo:** a **Raspberry Pi with a $60 Coral USB Accelerator runs our exact `_edgetpu.tflite` with our exact code** — the Edge TPU chip is the same silicon, just in a USB stick instead of soldered to the board. That is the cleanest possible proof of "scalable across hardware." A bare Pi (no accelerator) still works by falling back to the int8 CPU model we already have — just slower (~100–150 ms instead of 3 ms).

### 3.4 What changes vs. what stays the same

- **Stays identical:** behaviour on the Coral, the models, the dashboard UI, the 3 ms number. This is a pure refactor — same outputs, reorganised.
- **Changes:** internal structure only. We can verify the CPU path on a laptop with no board attached, so it's low-risk to develop.

### 3.5 Effort

| Task | Rough effort |
|---|---|
| Write `sortitude_core.py` (2 backends + camera + config) | ~0.5–1 day |
| Rewire `demo.py`, `webcam_demo.py`, `dashboard.py`, `benchmark.py` onto it | ~0.5 day |
| Verify CPU path on laptop, Edge TPU path on Coral | ~0.5 day |
| (Optional) borrow/buy a Pi and run the bare-CPU fallback as a second-device demo | depends on hardware access |

---

## 4. Track B — Better trash detection

Accuracy on the TrashNet test set is already good. The real weaknesses show up **live**, and they're not about the 92.9% number — they're about *behaviour*. In priority order:

### 4.1 ① Add a "background / empty" class — **biggest single win**

**Problem:** the model has exactly 6 outputs and softmax forces them to sum to 100%, so it **must** name a bin every frame — even pointing at an empty table or a wall. Our own notes record it calling an empty white room "paper." There is no way for the system to say *"nothing is here."*

**Fix:** capture a few hundred frames of empty scene / hand / conveyor and retrain as **7 classes** (the 6 waste types + `background`). Now the system stays quiet until a real item is presented.

**Why it's #1:** this does more for how *reliable* the demo feels than any amount of accuracy tuning. It converts "always guessing" into "responds when there's something to respond to."

### 4.2 ② Confidence gate + temporal smoothing

**Problem:** decisions are made on a single frame, so the label flickers and spurious high-confidence calls slip through.

**Fix:** require the prediction to (a) clear a confidence threshold (e.g. `> 0.85`) and (b) stay stable across ~5 consecutive frames (a rolling vote or exponential moving average) before we commit to a bin. Cheap to add, kills the flicker, and pairs naturally with the background class.

### 4.3 ③ Fix the confidence calibration

**Problem:** the displayed confidence saturates to 100% because we softmax raw logits (handled in `demo.py` lines 122–127). A bar that always reads 100% carries no information.

**Fix:** temperature scaling on the logits so a genuine 60% reads as ~60%. Makes the dashboard's probability bars meaningful instead of decorative.

### 4.4 ④ Close the domain gap (fine-tune on our own footage)

**Problem:** TrashNet is single clean items on white. Our webcam sees clutter, hands, shadows, motion blur, and odd lighting. That mismatch — not model capacity — is the main reason live predictions are shakier than the 92.9% suggests. It's also the reason a *different camera* would behave differently.

**Fix:** we already captured ~150 real on-device frames during the live demo (a curated subset is in `samples/`). Label a few hundred of them and fine-tune. This is the **cheapest accuracy win available** and it directly hardens the model against camera/lighting changes — which feeds back into Track A's "swap the camera" goal.

### 4.5 ⑤ (Optional, bigger) Move from classification to object *detection*

**Problem:** the model labels the *whole frame* with one class. It can't handle two items at once or tell you *where* the item is.

**Fix:** switch to **SSD-MobileNet** or **EfficientDet-Lite** — both compile to the Edge TPU — to get bounding boxes and multiple items per frame. This is a real lift (new model, new training data format, new dashboard rendering). Only worth it if the next assignment phase rewards true detection over classification.

### 4.6 Impact vs. effort summary

| # | Improvement | Impact on reliability | Effort | Needs retraining? |
|---|---|---|---|---|
| ① | Background/empty class | 🔥🔥🔥 | Medium | Yes (quick) |
| ② | Confidence gate + smoothing | 🔥🔥 | Low | No |
| ③ | Confidence calibration | 🔥 | Low | No |
| ④ | Fine-tune on our snaps | 🔥🔥 | Medium | Yes |
| ⑤ | Object detection | 🔥🔥🔥 | High | Yes (new model) |

---

## 5. Proposed roadmap

A sensible order that front-loads the safe, high-value work:

**Phase 1 — Portability foundation (no board, no retraining required)**
- Build `sortitude_core.py` (backend + camera + config).
- Rewire the demo/dashboard/benchmark scripts onto it.
- Verify CPU path on a laptop; verify Edge TPU path unchanged on the Coral.
- *Deliverable:* the same demo, now runnable on laptop / Pi / Coral from one codebase.

**Phase 2 — Reliability (small, no new model needed yet)**
- Add confidence gate + temporal smoothing (②).
- Add temperature-scaled calibration (③).
- *Deliverable:* a steady, honest dashboard that stops flickering and stops over-claiming 100%.

**Phase 3 — Detection quality (retraining)**
- Capture/label background frames → retrain as 7 classes (①).
- Label a slice of `snaps/` and fine-tune for the real camera domain (④).
- Re-compile the new model for the Edge TPU; update the results table.
- *Deliverable:* a model that stays quiet when idle and is robust to our actual camera.

**Phase 4 — Stretch (only if scope allows)**
- Second-device demo: bare Raspberry Pi running the CPU fallback (proves portability live).
- Object detection with EfficientDet-Lite (⑤).

### Suggested split for a 3-person team
- **Person 1 (code/infra):** Phase 1 refactor + config + Pi bring-up.
- **Person 2 (ML):** Phase 3 data capture, labelling, retraining, recompile.
- **Person 3 (app/UX):** Phase 2 dashboard logic (gate, smoothing, calibrated bars) + results/report updates.

These are largely independent, so the three phases can overlap.

---

## 6. Risks & things to watch

- **Re-quantization after retraining:** adding a class or fine-tuning means re-running int8 quantization **and** the Edge TPU compile. Budget time to re-verify all 67 ops still map to the TPU (don't assume).
- **Don't repeat the pruning mistake:** as `PRUNED_MODEL_FIX.md` documents, stacking aggressive compression broke the model. Keep changes one at a time and re-measure.
- **Background class balance:** too many background samples and the model gets lazy and predicts "background" for everything. Keep the class balanced.
- **Pi camera quirks:** `picamera2` and OpenCV report colour channels and resolutions differently — test the preprocessing on the actual Pi camera, don't assume it matches the USB webcam.
- **Scope creep:** object detection (⑤) is a different project in disguise. Treat it as stretch, not baseline.

---

## 7. Recommended immediate next step

Start **Phase 1** — extract `sortitude_core.py`. Reasons: it's the literal answer to the "scalable to new hardware/camera" goal, it needs no board and no retraining, it can be developed and verified on a laptop, and it makes every later change (new device, new model, calibration) land in one place instead of four. Phase 2's confidence gate is the natural quick follow-on.

---

## 8. Appendix — plain-language glossary (for non-ML teammates)

- **Backend** — the piece of code that actually runs the model. Different hardware needs a different backend (Edge TPU vs. plain CPU), but they all answer the same question: "image in, scores out."
- **Quantization** — storing the model's numbers as 8-bit integers instead of 32-bit decimals; smaller and faster, slightly less precise.
- **Open-set / background class** — teaching the model that "none of the above / nothing here" is a valid answer, instead of forcing it to always pick a bin.
- **Temporal smoothing** — deciding based on the last few frames together, not one frame, so the label doesn't jitter.
- **Calibration** — making the confidence percentage actually mean what it says (a "70%" should be right about 70% of the time).
- **Domain gap** — the difference between the clean training photos and the messy real camera view; the bigger the gap, the worse live performance.
- **Coral USB Accelerator** — the same Edge TPU chip as on our board, but in a USB stick you can plug into a Raspberry Pi or laptop.
