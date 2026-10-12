# Setup & Running on the Coral Dev Board

How to compile the model, deploy it to a Google Coral Dev Board, and run the live demo.

## Hardware

| Coral port | Connect to |
|---|---|
| **USB-C power** | 2–3 A wall charger |
| **USB-C OTG / data** | your laptop (board shows up as a USB network device) |
| **USB-A host** | USB webcam (UVC; we used a Microsoft LifeCam NX-3000) |
| micro-USB serial console | only needed for recovery |

- The board runs **Mendel Linux 5.3**; the Edge TPU is at `/dev/apex_0`, runtime `libedgetpu1-std`.
- The webcam enumerates as **`/dev/video1`** (camera index 1). `/dev/video0` is the empty
  on-board CSI camera bridge — opening it floods `VIDIOC_QUERYCTRL` errors.
- The board is headless: HDMI staying black is normal. You use the browser instead.

## 1. Compile the model for the Edge TPU (one-time, x86 Linux or WSL2)

The Edge TPU only runs int8 models compiled with `edgetpu_compiler`. A compiled model
is already in [`models/sortitude_int8_edgetpu.tflite`](../models/), so this step is only
needed if you retrain.

```bash
bash tools/wsl_get_compiler.sh   # downloads + extracts edgetpu_compiler, no sudo needed
bash tools/wsl_compile.sh        # models/sortitude_int8.tflite -> models/sortitude_int8_edgetpu.tflite
```

All 67 ops of the model map to the Edge TPU (see `models/sortitude_int8_edgetpu.log`).

## 2. Connect to the board

Install [MDT](https://coral.ai/docs/dev-board/mdt/) (`pip install mendel-development-tool`) and
check the board is visible:

```bash
mdt devices        # prints e.g.  <board-name>  (<board-ip>)
```

Then SSH in with the key MDT generated when you paired the board:

```bash
ssh -i ~/.config/mdt/keys/mdt.key -o PubkeyAcceptedKeyTypes=+ssh-rsa mendel@<board-ip>
```

> On Windows, `mdt exec`/`mdt shell` can fail with "handle is invalid" — plain `ssh` works.
> If you lose the key, log in over the serial console and add a new public key to
> `~/.ssh/authorized_keys`.

## 3. Deploy the files

The scripts expect everything in one flat folder on the board (`~/sortitude/`):

```bash
ssh mendel@<board-ip> "mkdir -p ~/sortitude"
scp board/*.py board/start_dashboard.sh \
    models/labels.txt models/sortitude_int8.tflite models/sortitude_int8_edgetpu.tflite \
    samples/test_sample.jpg \
    mendel@<board-ip>:~/sortitude/
```

The board image already ships `python3`, `numpy`, `opencv`, `tflite_runtime` and `pycoral`.

## 4. Run the live dashboard

```bash
ssh mendel@<board-ip> "bash ~/sortitude/start_dashboard.sh"   # waits until it prints "UP after Ns"
```

Open **`http://<board-ip>:8000/`** in a browser on the laptop and hold items up to the webcam.

- Live prediction, confidence and Recycling/Landfill routing update in real time (~30 FPS).
- Click the video or press **Space** to save a snapshot; **P** pauses.
- Any class reaching 100% confidence auto-saves a snapshot to `~/sortitude/snaps/`.
- Stop it with `ssh mendel@<board-ip> "pkill -f dashboard.py"`.

Only one process can hold the camera: stop the dashboard before running `webcam_demo.py`, and vice versa.

`board/dashboard_v2_experimental.py` is a redesigned HUD version of the dashboard
(confidence sparkline, per-class counts). It has **not been tested on the board yet** —
preview its UI on a laptop with `python mock/mock_dashboard_v2.py`.

## 5. Other scripts

```bash
# CPU vs Edge TPU latency (50 runs)
python3 benchmark.py --cpu-model sortitude_int8.tflite --tpu-model sortitude_int8_edgetpu.tflite \
  --labels labels.txt --image test_sample.jpg -n 50

# Single image, Edge TPU
python3 demo.py --model sortitude_int8_edgetpu.tflite --labels labels.txt --image test_sample.jpg

# Headless webcam loop that saves annotated frames
python3 webcam_demo.py --model sortitude_int8_edgetpu.tflite --labels labels.txt --camera 1
```

`demo.py --cpu` also runs on a laptop with `tflite-runtime` or TensorFlow installed
(use `models/sortitude_int8.tflite`, not the `_edgetpu` one).

## 6. Copy snapshots back

```bash
scp "mendel@<board-ip>:~/sortitude/snaps/*.jpg" ./snaps/
```

The board's clock is not NTP-synced, so snapshot timestamps may be wrong (ordering is still correct).

## Troubleshooting

| Problem | Fix |
|---|---|
| Browser can't load the page | Re-run step 4; confirm the IP with `mdt devices` |
| "Could not open camera" | Webcam must be in the USB-A port (`/dev/video1`); re-plug and restart |
| Feed frozen | Press **P** (you may be paused), or restart the dashboard |
| Board not detected | Check the 3 A charger is on the POWER port and the laptop cable on the OTG port |
| SSH "permission denied" | Wrong/missing key — re-add one via the serial console |

`tools/diagnostics/` has the small probe scripts we used while debugging the board
(camera, USB, environment checks).
