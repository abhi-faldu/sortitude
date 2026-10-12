# 🚀 Sortitude — Next Phase Research: Hardware, Datasets & Production Readiness

**Audience:** the Sortitude team (Abhi, Jeel, Aatman)
**Purpose:** research-backed roadmap for the *next* phase — (A) upgrading or replacing the edge device, (B) using more/better datasets for stronger classification, and (C) what it takes to go from a graded prototype to a production / market-ready product.
**Status:** research report. Complements (does **not** replace) [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md), which covers code portability and live-detection quality on the *current* hardware. This report is about *upgrading the foundation*.
**Date:** 2026-06-30

---

## 0. TL;DR — the three decisions

| Question | Short answer | Why |
|---|---|---|
| **Should we stay on the Coral Dev Board?** | **No, not for anything beyond the current demo.** The board is **end-of-life** (Google EOL notice, retailers discontinued, forum/docs frozen). It still works, but it's a dead-end platform. | Building "production-ready" on EOL hardware is a non-starter. |
| **What should we upgrade to?** | **Raspberry Pi 5 + Hailo-8L AI Kit** (best value, ~3× Coral, huge community) for a like-for-like upgrade; **NVIDIA Jetson Orin Nano Super** ($249, 67 TOPS) if you want headroom for object detection + bigger models. | Both are current, supported, and dramatically more capable than 4-TOPS Coral. |
| **How do we get better classification?** | **Combine TrashNet with RealWaste + TACO**, add a *background* class, and move from a "clean-photo" model to a "real-world" model. More/better data beats a bigger model here. | The accuracy gap is a **domain gap** (clean photos vs. messy reality), not a model-capacity problem. |

**The honest framing for a report/pitch:** *"Our prototype proved the concept on Coral. The next phase moves to current, supported hardware (Pi 5 + Hailo or Jetson), trains on real-world waste data instead of clean studio photos, and adds the reliability layer (background detection, calibration, monitoring) that separates a demo from a product."*

---

## Part A — Edge Device Landscape (2026)

### A.1 Critical context: the Coral Dev Board is EOL

This is the single most important finding and it reframes the whole "next phase":

- **Google issued an End-of-Life notice** for the Coral Dev Board (4GB) and its SoM ([Mouser PCN](https://www.mouser.com/PCN/Coral_CORAL_DEV_BOARD(4G)__SOM(4G)_EOL_Notice.pdf)).
- Major retailers (e.g. [The Pi Hut](https://thepihut.com/products/google-coral-dev-board-4gb)) list it as **discontinued**.
- The original 2019 board has been **left in neglect** — the official forum is closed and documentation moved to read-only ([Using the Coral Dev Board in 2025](https://syllepsis.live/2025/01/14/using-the-coral-dev-board-in-2025/)).
- Google **did** relaunch an edge brand in May 2026 — a **new "Coralboard" built with Synaptics**, ~1 TOPS, aimed at running small generative models like **Gemma 3 270M** on-device. But it's a *different product line* (on-device GenAI, not the classic Edge TPU vision accelerator), was a **limited edition** at Google I/O 2026, and general availability/pricing is still "later this year" ([coverage](https://pasqualepillitteri.it/en/news/3583/coralboard-google-gemma-edge-ai-1-tops-en), [Coral news](https://www.coral.ai/news/)). **It is not a drop-in successor** for your int8 vision pipeline.

**Implication:** the Edge-TPU `_edgetpu.tflite` you compiled is a per-vendor artifact tied to a sunset toolchain. Any serious next phase should treat Coral as "the device we proved it on," not "the device we ship on."

### A.2 The realistic upgrade options

> ⚠️ **Migration cost note:** every NPU has its *own* model format. Moving off Coral means re-converting the model — Hailo uses `.hef` (Hailo Dataflow Compiler), Jetson uses TensorRT engines, Rockchip uses `.rknn`. Your **Keras/TFLite source model is portable**; the *compiled* artifact is not. Budget a few days per target for conversion + re-validation. This is exactly why [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) Track A (swappable backend) matters.

| Platform | AI perf (TOPS)* | Power | ~Price (kit) | Model formats | Best for | Verdict for Sortitude |
|---|---|---|---|---|---|---|
| **Coral Dev Board** (current) | 4 | ~2–4 W | EOL | int8 TFLite → Edge TPU | MobileNet/EfficientDet-Lite, ultra-low power | ❌ EOL — demo only |
| **Raspberry Pi 5 + Hailo-8L (AI Kit)** | 13 | ~2.5 W (accel) | **~$70 kit + ~$80 Pi** | ONNX → Hailo `.hef` | Drop-in upgrade, classification + light detection | ✅ **Best value / recommended baseline** |
| **Raspberry Pi 5 + Hailo-8 (M.2)** | 26 | ~2.5 W | ~$120 module + Pi | ONNX → `.hef` | Heavier detection (YOLOv8), multi-stream | ✅ If you need real object detection |
| **NVIDIA Jetson Orin Nano Super** | **67** | 7–25 W | **$249** | PyTorch/ONNX/TensorRT (FP16 + INT8) | Max flexibility, VLMs/LLMs, robotics | ✅ **Best ceiling** — runs *anything* |
| **Rockchip RK3588** (Orange Pi 5 / Radxa Rock 5) | 6 | ~5–8 W | ~$90–150 | ONNX → `.rknn` | Cheap general-purpose SBC w/ NPU | ⚠️ Capable but weaker tooling/community |
| **Luxonis OAK smart camera** | (on-cam) | ~4–5 W | ~$150–250 | OpenVINO/blob | Camera + inference in one unit | ⚠️ Niche; good if you want an all-in-one sensor |

\* **TOPS is a marketing number** measured differently per vendor — the only number that matters is *your* model on *your* hardware under real load ([TechnoLynx](https://www.technolynx.com/post/embedded-edge-device-cv-deployment), [Privacy Smart Home](https://www.privacysmarthome.com/guides/coral-tpu-vs-hailo-vs-gpu-home-ai-2026/)). Treat the table as a rough ranking, not gospel.

### A.3 What the upgrade actually buys you

- **Raspberry Pi 5 + Hailo-8L** — ResNet-50 runs **>200 FPS**, YOLOv8s ~**25–35 FPS** at 640×640 (vs ~2–3 FPS on the Pi CPU alone), and it's reported at roughly **3× Coral's performance at a similar price**, with first-class support in Frigate/Home Assistant and an active community ([Hailo community benchmark](https://community.hailo.ai/t/raspberry-pi-5-with-hailo-8l-benchmark/746), [Raspberry Pi AI Kit](https://www.raspberrypi.com/products/ai-kit/), [Mustafa.net](https://mustafa.net/2026/03/12/hailo-8l-ai-accelerator-edge-ai-for-raspberry-pi-and-homelab-projects/)). It runs **YOLO, ResNet, EfficientDet** families that Coral struggles with.
- **Jetson Orin Nano Super** — **67 TOPS at $249**, runs **arbitrary ONNX/PyTorch** via TensorRT (FP16 *and* INT8), and can host vision transformers and even small LLMs/VLMs. The trade-off is **higher power (7–25 W)** and cost — overkill for pure 6-class classification, but the right call if the roadmap includes detection, multi-camera, or an on-device assistant ([NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), [Electronics-Lab](https://www.electronics-lab.com/exploring-the-nvidia-jetson-orin-nano-super-developer-kit-with-67-tops-ai-performance-at-just-249/)).
- **RK3588 (Orange Pi 5 Ultra / Radxa Rock 5)** — 6-TOPS NPU, **ResNet18 at 244 FPS / 4.09 ms**, mature enough for "surveillance, robotics, smart sensors," but the **RKNN toolchain and community are smaller** than Hailo/Jetson ([TinyComputers deep dive](https://tinycomputers.io/posts/rockchip-rk3588-npu-benchmarks.html), [Tristan Penman](https://tristanpenman.com/blog/posts/2025/07/20/edge-ai-using-the-rockchip-npu/)).

### A.4 Recommendation

1. **Primary upgrade → Raspberry Pi 5 + Hailo-8L AI Kit.** Cheapest path off EOL hardware, ~3× the compute, runs the model families you'll want next (YOLO/EfficientDet for detection), and the Pi ecosystem is the de-facto standard for this kind of project. It's also the *literal* second-device story already hinted at in your [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md).
2. **If the roadmap includes true object detection / multiple items / future LLM features → Jetson Orin Nano Super.** It removes every ceiling for $249.
3. **Keep the Coral as the "we proved it on 2 W" data point** — it's a genuinely good story for power efficiency; just don't build the product on it.

---

## Part B — Datasets: Stronger Classification Through Better Data

Your model scores 92.9% on TrashNet but wobbles live. That's the classic **domain gap**: TrashNet is ~2,500 photos of *single clean items on white backgrounds*, while your webcam sees clutter, hands, shadows, and motion blur. **The fix is more realistic and more varied data, not (only) a bigger model.**

### B.1 The dataset shortlist (verified on Hugging Face + beyond)

> Numbers below were **verified directly from the dataset pages** (2026-06-30), not just search snippets. Licenses noted where stated.

| Dataset | Size (verified) | Classes | Style | Task | Where / License | Value to Sortitude |
|---|---|---|---|---|---|---|
| **TrashNet** (current) | **5,054 rows** on HF (classic TrashNet is ~2,527; HF copy is expanded), 512² px, single `train` split | 6 (cardboard, glass, metal, paper, plastic, trash) | Clean items, white bg | Classification | [HF: garythung/trashnet](https://huggingface.co/datasets/garythung/trashnet) · **MIT** | Baseline — keep, but recognise its limits |
| **omasteam / waste-garbage-management** ⭐ | **19,762 images** (790 MB) | **10** — Metal 1,020 · Glass 3,061 · Biological 997 · Paper 1,680 · Battery 944 · Trash 947 · Cardboard 1,825 · Shoes 1,977 · Clothes 5,327 · Plastic 1,984 | Mixed | Classification | [HF](https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset) · **MIT** | **Biggest HF image set** — superset of your 6 classes + Battery/Clothes/Shoes/Biological. Great scale-up. |
| **RealWaste** ⭐ | **4,752 images** (524² px) | 9 (Cardboard, Food Organics, Glass, Metal, Misc Trash, Paper, Plastic, Textile Trash, Vegetation) | **Real landfill conditions** | Classification | [UCI #908](https://archive.ics.uci.edu/dataset/908/realwaste) | **#1 for the domain gap** — real (not studio) imagery; ResNet101 hits 98.86% val on it |
| **rootstrap-org / waste-classifier** | **3,261 images**, pre-split train/test | 7 (TrashNet 6 **+ compost**) | TrashNet + Bing-search images, mostly white bg | Classification | [HF](https://huggingface.co/datasets/rootstrap-org/waste-classifier) · **MIT** | Drop-in TrashNet superset; adds a **compost** class. (Still white-bg — same domain limit.) |
| **TACO** ⭐ | 1,500 imgs / 4,784 anns | 60 (28 super) | **In-the-wild** (streets, beaches) | Detection / segmentation | [GitHub](https://github.com/pedropro/TACO), [Kaggle](https://www.kaggle.com/datasets/kneroma/tacotrashdataset) | Best if you move to **object detection** (boxes, multiple items) |
| **detect-waste** | 100k+ images | 11 cats / 187 sub | Mixed/real | Detection | [waste-datasets-review](https://github.com/AgaMiko/waste-datasets-review) | Largest pool; great for pre-training / scaling up |
| **Kaggle Waste Classification** | ~25k images | 2 (Organic/Recyclable) | Mixed | Classification | linked in review repo | Huge, but only binary — useful for a coarse first stage |
| **thomasavare / waste-classification** ⚠️ | 45,936 rows (**text/tabular, NOT images**) | 50 label categories | Disposal Q&A + labels | Text classification | [HF](https://huggingface.co/datasets/thomasavare/waste-classification) | **Not for vision training** — but a ready-made **50-category taxonomy** to align labels against |
| **The Garbage Dataset (GD)** | benchmark | multi-class | — | Classification | [arXiv](https://arxiv.org/html/2602.10500v2) | Recent multi-class benchmark for comparison |
| **UAVVaste** | — | — | Aerial | Detection | review repo | Only if drone/overhead use case |

A maintained master list lives at [AgaMiko/waste-datasets-review](https://github.com/AgaMiko/waste-datasets-review) — bookmark it.

**Takeaways from verifying the HF sets:**
- **omasteam (19.7k)** is the largest ready-to-use HF *image* set and a clean superset of your classes — the best single HF source to **scale up** training volume.
- **rootstrap-org (3.3k)** is the easiest drop-in (already TrashNet-shaped, adds *compost*), but it's still white-background — it grows quantity, not realism.
- **RealWaste (4.7k)** remains the key for *realism* even though it lives on UCI, not HF.
- **thomasavare** is a trap if you grab it blind — it's text, not images. Its value is the 50-label taxonomy for mapping.
- Most HF waste sets ship as **a single `train` split with no test set** — you must carve out your own held-out (ideally real-world) test split, or accuracy numbers will be optimistic.

### B.2 Why RealWaste is the highest-value addition

The RealWaste authors make the point directly: models trained on **pristine** datasets collapsed to **~49.7% accuracy** on real-world waste, whereas training on RealWaste's contaminated/commingled images kept every model **above 85%** (Inception V3 89.2%, ResNet101 98.9%) ([MDPI paper](https://www.mdpi.com/2078-2489/14/12/633), [Nature Sci Reports](https://www.nature.com/articles/s41598-026-41001-8)). That ~49.7% number is *exactly the failure mode your live demo hints at.* RealWaste's 9 classes are also a near-superset of yours (it splits "trash" into Food Organics / Miscellaneous / Textile / Vegetation).

### B.3 How to combine datasets without making things worse

Combining datasets is powerful but has sharp edges — do it deliberately:

1. **Unify the taxonomy first.** TrashNet's 6 classes ≠ RealWaste's 9 ≠ TACO's 60. Define a single canonical label set (e.g. your 6 + `background`) and write an explicit mapping table for each source. Mismatched labels silently poison training.
2. **Add a `background` / empty class** (also the #1 item in [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) §4.1). Pull frames from your own live frames from the board (a curated subset is in `samples/`) plus empty-scene captures. This is the single biggest *perceived* reliability win.
3. **Mix domains, don't just stack them.** A model trained 90% on clean TrashNet + 10% RealWaste will still behave like a clean-photo model. Aim for a balanced blend, and **hold out a real-world-only test set** (RealWaste or your own snaps) so your reported accuracy reflects reality, not studio conditions.
4. **Watch class balance.** detect-waste/Kaggle are huge and will swamp TrashNet if naively merged. Cap per-class counts or use weighted sampling.
5. **Fine-tune on *your* camera last.** After training on the blended set, a final fine-tune on labelled `snaps/` frames adapts to your specific lens/lighting — the cheapest accuracy win available.

### B.4 Two-stage strategy (recommended)

```
Stage 0 (data):   TrashNet + RealWaste + (your snaps) + background  →  unified 7-class set
Stage 1 (train):  EfficientNet-Lite / MobileNetV3 backbone, train on blended set
Stage 2 (adapt):  fine-tune on labelled real webcam frames
Stage 3 (eval):   report accuracy on a REAL-WORLD-ONLY held-out set
(optional) →      TACO/detect-waste for an object-detection variant (YOLOv8 / EfficientDet-Lite)
```

---

## Part C — Model Upgrade Path (unlocked by better hardware)

Better hardware (Part A) lets you outgrow MobileNetV2. Match the model to the device:

| Hardware | Recommended model | Why |
|---|---|---|
| Coral (current) | MobileNetV2 int8 (today) | Only int8 MobileNet/EfficientDet-Lite compile cleanly |
| **Pi 5 + Hailo-8L** | **EfficientNet-Lite / MobileNetV3**, or **YOLOv8n/s** for detection | Hailo runs these natively at high FPS |
| Jetson Orin Nano Super | **YOLOv8/v11**, EfficientNet, or a ViT classifier | TensorRT runs arbitrary architectures; room to spare |

**Key upgrade:** move from **whole-frame classification** → **object detection** (bounding boxes, multiple items per frame). This is the difference between "tell me what's in the picture" and "find and sort each item" — and it's what every commercial system (Part D) actually does. TACO/detect-waste are your training data for this; EfficientDet-Lite (Coral-compatible) or YOLOv8 (Hailo/Jetson) are the models.

---

## Part D — Production / Market Readiness

### D.1 The market is real and growing fast

This isn't a toy problem — it's a fast-growing commercial space, which is good news for a portfolio/market angle:

- **Waste-sorting robots market:** **$3.29B (2025) → $3.93B (2026) → $16.47B (2034)**, CAGR **19.6%** ([Market Data Forecast](https://www.marketdataforecast.com/market-reports/waste-sorting-robots-market)).
- **AI-in-waste-management** adoption growing ~**22%/yr through 2030** ([OpenPR](https://www.openpr.com/news/4503351/ai-in-waste-management-market-is-booming-worldwide-amp)).
- **Greyparrot** runs AI analyzers above conveyor belts in **20+ countries**, now combining NIR polymer detection with vision ([greyparrot.ai](https://www.greyparrot.ai/)).
- **AMP Robotics** has **300+ robotic sorting systems** deployed, **10B+ item detections**, **90%+ recovery** on target materials ([Robot Report](https://www.therobotreport.com/how-waste-robotics-and-greyparrot-enable-more-effective-sorting-robots/)).
- AMP, Glacier, and Greyparrot made **Time's 2025 Top 100 Innovations** ([Packaging World](https://www.packworld.com/sustainable-packaging/recycling/article/22952428/aipowered-recycling-tech-gains-national-spotlight-in-times-top-100-list)).

**Where Sortitude fits:** the giants target industrial MRFs (conveyor belts, robotic arms, $$$). The open lane for a small team is the **smart-bin / point-of-disposal** segment — a low-cost unit at the bin that classifies and guides sorting *before* waste is commingled. That's a credible, defensible niche.

### D.2 Prototype → Product: the honest gap analysis

| Dimension | Where Sortitude is now | What "production-ready" needs |
|---|---|---|
| **Hardware** | EOL Coral, dev board, bench setup | Supported platform (Pi 5+Hailo/Jetson), enclosure, mounting, proper camera + lighting |
| **Model** | 6-class, clean-photo trained | Real-world trained, background class, calibrated confidence, object detection |
| **Reliability** | Always guesses; 100% confidence; single-frame | Confidence gate, temporal smoothing, "I'm not sure" state (see [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) §4) |
| **Robustness** | One camera, one lighting, indoor | Tested across lighting, cameras, item poses, dirty/wet items |
| **Software** | Scripts coupled to Coral | Portable core (Track A), packaged install, OTA update path |
| **MLOps** | Train-once in Colab | Versioned datasets/models, retraining pipeline, **drift monitoring**, A/B eval |
| **Connectivity** | Local dashboard over USB | Optional cloud telemetry, fleet dashboard, remote model push (Greyparrot-style analytics) |
| **Data flywheel** | Static dataset | Capture-label-retrain loop from deployed units (this is the real moat) |
| **Compliance/Ops** | None | Power/thermal/safety, enclosure rating (IP), data privacy, cost-per-unit BOM, support |

### D.3 What separates a demo from a product (priority order)

1. **Run on supported hardware** (Part A) — table stakes.
2. **Reliability layer** — background class + confidence gate + smoothing + calibration. A device that confidently says "paper" at an empty bin is unshippable. (Already scoped in [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md).)
3. **Real-world data + monitoring** — train on RealWaste/your own data; then **monitor for drift** in the field and retrain. The *data flywheel* is what AMP/Greyparrot's moat is built on.
4. **Object detection** — handle multiple/partial items; localise, don't just label.
5. **Packaging** — enclosure, camera+lighting rig, one-command install, optional cloud telemetry.
6. **Unit economics** — BOM cost, power, expected accuracy SLA, support model.

---

## Part E — Proposed Roadmap (research-driven)

This sequences the research above; it dovetails with the portability/quality work already in [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md).

**Phase A — De-risk the platform (no new model)**
- Acquire a **Pi 5 + Hailo-8L AI Kit** (~$150 total). Stand up the portable `sortitude_core.py` backend (Track A) with a Hailo backend.
- Re-convert the existing model to Hailo `.hef`; reproduce the live demo on supported hardware.
- *Deliverable:* the demo running on current, supported hardware — and a clean "Coral vs Pi+Hailo" benchmark comparison (great for the report).

**Phase B — Better data, better model**
- Build the unified dataset: TrashNet + **RealWaste** + your `snaps/` + a **background** class.
- Retrain (EfficientNet-Lite / MobileNetV3); evaluate on a **real-world-only** held-out set.
- Add confidence gate + temporal smoothing + temperature calibration ([`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) §4.2–4.3).
- *Deliverable:* a model that's robust live and reports honest confidence.

**Phase C — From classification to detection (optional, bigger)**
- Train **YOLOv8** (Hailo/Jetson) or **EfficientDet-Lite** on **TACO/detect-waste**.
- Update the dashboard to draw boxes + handle multiple items.
- *Deliverable:* localise-and-sort, the commercial-grade capability.

**Phase D — Production hardening (if pursuing market)**
- Enclosure + camera/lighting rig; packaged install; optional cloud telemetry + fleet dashboard.
- Drift monitoring + capture-label-retrain flywheel.
- BOM/cost analysis; pilot at a real bin.
- *Deliverable:* a deployable smart-bin unit + the data loop that improves it over time.

---

## 6. Sources

**Hardware**
- [Coral Dev Board EOL notice (Mouser PCN)](https://www.mouser.com/PCN/Coral_CORAL_DEV_BOARD(4G)__SOM(4G)_EOL_Notice.pdf)
- [Using the Coral Dev Board in 2025 — Syllepsis](https://syllepsis.live/2025/01/14/using-the-coral-dev-board-in-2025/)
- [Google Coral Dev Board [discontinued] — The Pi Hut](https://thepihut.com/products/google-coral-dev-board-4gb)
- [New Coralboard (Google + Synaptics, Gemma 3, 1 TOPS)](https://pasqualepillitteri.it/en/news/3583/coralboard-google-gemma-edge-ai-1-tops-en) · [Coral News](https://www.coral.ai/news/)
- [Jetson vs Coral vs Hailo vs OAK-D — TechnoLynx](https://www.technolynx.com/post/embedded-edge-device-cv-deployment)
- [Coral TPU vs Hailo vs GPU 2026 — Privacy Smart Home](https://www.privacysmarthome.com/guides/coral-tpu-vs-hailo-vs-gpu-home-ai-2026/)
- [Raspberry Pi AI Kit (Hailo-8L)](https://www.raspberrypi.com/products/ai-kit/) · [Hailo-8L benchmark thread](https://community.hailo.ai/t/raspberry-pi-5-with-hailo-8l-benchmark/746) · [Mustafa.net Hailo-8L review](https://mustafa.net/2026/03/12/hailo-8l-ai-accelerator-edge-ai-for-raspberry-pi-and-homelab-projects/)
- [Jetson Orin Nano Super Developer Kit — NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/) · [$249 / 67 TOPS — Electronics-Lab](https://www.electronics-lab.com/exploring-the-nvidia-jetson-orin-nano-super-developer-kit-with-67-tops-ai-performance-at-just-249/)
- [RK3588 NPU deep dive — TinyComputers](https://tinycomputers.io/posts/rockchip-rk3588-npu-benchmarks.html) · [Edge AI on Rockchip NPU — Tristan Penman](https://tristanpenman.com/blog/posts/2025/07/20/edge-ai-using-the-rockchip-npu/)

**Datasets**
- [garythung/trashnet (HF)](https://huggingface.co/datasets/garythung/trashnet)
- [RealWaste — UCI #908](https://archive.ics.uci.edu/dataset/908/realwaste) · [RealWaste paper (MDPI)](https://www.mdpi.com/2078-2489/14/12/633) · [Real-world waste CNN study (Nature)](https://www.nature.com/articles/s41598-026-41001-8)
- [TACO dataset (GitHub)](https://github.com/pedropro/TACO) · [TACO paper (arXiv)](https://arxiv.org/pdf/2003.06975) · [TACO on Kaggle](https://www.kaggle.com/datasets/kneroma/tacotrashdataset)
- [AgaMiko/waste-datasets-review (master list)](https://github.com/AgaMiko/waste-datasets-review)
- [omasteam/waste-garbage-management-dataset (HF)](https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset) · [rootstrap-org/waste-classifier (HF)](https://huggingface.co/datasets/rootstrap-org/waste-classifier) · [thomasavare/waste-classification (HF)](https://huggingface.co/datasets/thomasavare/waste-classification)
- [The Garbage Dataset (GD) — arXiv](https://arxiv.org/html/2602.10500v2) · [DWaste: Greener AI for Waste Sorting on Edge — arXiv](https://arxiv.org/pdf/2510.18513)

**Market / production**
- [Waste Sorting Robots Market — Market Data Forecast](https://www.marketdataforecast.com/market-reports/waste-sorting-robots-market)
- [AI in Waste Management market — OpenPR](https://www.openpr.com/news/4503351/ai-in-waste-management-market-is-booming-worldwide-amp)
- [Greyparrot](https://www.greyparrot.ai/) · [Waste Robotics + Greyparrot — The Robot Report](https://www.therobotreport.com/how-waste-robotics-and-greyparrot-enable-more-effective-sorting-robots/)
- [AI-powered recycling in Time's Top 100 — Packaging World](https://www.packworld.com/sustainable-packaging/recycling/article/22952428/aipowered-recycling-tech-gains-national-spotlight-in-times-top-100-list)
- [How AI is revolutionizing recycling — Columbia Climate](https://news.climate.columbia.edu/2025/06/18/how-ai-is-revolutionizing-the-recycling-industry/)
