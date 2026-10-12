#!/usr/bin/env python3
"""Sortitude interactive web dashboard for the Coral Edge TPU.

A background thread continuously captures webcam frames and runs Edge TPU
inference, so the server can stream a live annotated MJPEG feed AND answer fast
state-polls without fighting over the camera. Stdlib-only HTTP server.

Endpoints:
  /          interactive HTML UI
  /stream    multipart MJPEG live feed (annotated)
  /state     JSON: current label, confidence, per-class scores, latency, fps
  /snapshot  JSON: frozen annotated frame (base64) + prediction (for history)
  /control?action=pause|resume   freeze / resume the live feed
"""
import argparse
import base64
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

INTERP = None
LABELS = []
CAM = None
INPUT_WH = (224, 224)
SAVE_DIR = "snaps"

LATEST = {"jpeg": None, "label": "—", "conf": 0.0, "scores": [],
          "ms": 0.0, "fps": 0.0, "n": 0}
PAUSED = False
RUN = True

# class -> (emoji, bin/category, css color)
ROUTING = {
    "cardboard": ("📦", "Recycling", "#4ade80"),
    "paper":     ("📄", "Recycling", "#4ade80"),
    "glass":     ("🍶", "Recycling", "#38bdf8"),
    "metal":     ("🥫", "Recycling", "#facc15"),
    "plastic":   ("🧴", "Recycling", "#a78bfa"),
    "trash":     ("🗑️", "Landfill",  "#f87171"),
}


def set_input(interp, image):
    inp = interp.get_input_details()[0]
    t = interp.tensor(inp["index"])()[0]
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        q = image.astype(np.float32) / scale + zp
        t[:, :, :] = np.clip(np.round(q), -128, 127).astype(np.int8)
    else:
        t[:, :, :] = image.astype(inp["dtype"])


def softmax_scores(interp):
    out = interp.get_output_details()[0]
    s = interp.get_tensor(out["index"])[0]
    if out["dtype"] == np.int8:
        scale, zp = out["quantization"]
        s = (s.astype(np.float32) - zp) * scale
    s = s.astype(np.float32)
    if s.min() < 0 or s.max() > 1 or abs(s.sum() - 1) > 1e-2:
        e = np.exp(s - s.max()); s = e / e.sum()
    return s


def worker():
    """Continuously grab frames, run inference, update LATEST."""
    global LATEST
    times = deque(maxlen=30)
    while RUN:
        if PAUSED:
            time.sleep(0.05)
            continue
        ok, frame = CAM.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, INPUT_WH)
        set_input(INTERP, inp)
        t0 = time.perf_counter()
        INTERP.invoke()
        scores = softmax_scores(INTERP)
        ms = (time.perf_counter() - t0) * 1000
        idx = int(np.argmax(scores))
        label = LABELS[idx]; conf = float(scores[idx])

        emoji, _, color_hex = ROUTING.get(label, ("", "", "#4ade80"))
        bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))  # hex->BGR
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(frame, f"{label}  {conf*100:.0f}%", (12, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, bgr, 2, cv2.LINE_AA)
        ok2, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

        now = time.perf_counter(); times.append(now)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
        LATEST = {"jpeg": jpg.tobytes(), "label": label, "conf": conf,
                  "scores": [float(x) for x in scores], "ms": ms,
                  "fps": fps, "n": LATEST["n"] + 1}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sortitude — Live Waste Classifier</title>
<style>
 :root{--bg:#0b1220;--card:#16213a;--mut:#94a3b8;--line:#27324a;}
 *{box-sizing:border-box;}
 body{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;background:
   radial-gradient(1200px 600px at 50% -10%,#1e293b,var(--bg));color:#e2e8f0;min-height:100vh;}
 .wrap{max-width:1040px;margin:0 auto;padding:20px;}
 header{display:flex;align-items:center;gap:12px;margin-bottom:6px;}
 header h1{font-size:24px;margin:0;font-weight:800;letter-spacing:.3px;}
 .pill{font-size:11px;padding:3px 9px;border-radius:999px;background:#0f2a1c;color:#4ade80;
   border:1px solid #1f5137;font-weight:700;}
 .pill.paused{background:#2a160f;color:#fbbf24;border-color:#5a3a17;}
 .sub{color:var(--mut);font-size:13px;margin-bottom:18px;}
 .grid{display:grid;grid-template-columns:1.4fr 1fr;gap:18px;}
 @media(max-width:820px){.grid{grid-template-columns:1fr;}}
 .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;
   box-shadow:0 12px 40px rgba(0,0,0,.45);}
 .feedwrap{position:relative;}
 #feed{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:12px;background:#000;
   border:1px solid var(--line);cursor:crosshair;}
 .live-badge{position:absolute;top:10px;right:10px;font-size:11px;font-weight:800;
   background:rgba(220,38,38,.9);color:#fff;padding:4px 9px;border-radius:6px;
   box-shadow:0 2px 8px rgba(0,0,0,.5);}
 .live-badge.paused{background:rgba(202,138,4,.95);}
 .auto-toast{position:absolute;top:10px;left:10px;font-size:11px;font-weight:800;
   background:rgba(34,197,94,.95);color:#04210f;padding:4px 9px;border-radius:6px;
   opacity:0;transition:opacity .2s;pointer-events:none;}
 .auto-toast.show{opacity:1;}
 #hist .autob{position:absolute;top:3px;right:3px;font-size:8px;font-weight:800;
   background:#22c55e;color:#04210f;border-radius:4px;padding:1px 4px;}
 .controls{display:flex;gap:10px;margin-top:12px;}
 button{flex:1;padding:13px;font-size:15px;font-weight:700;border:none;border-radius:11px;
   cursor:pointer;color:#fff;transition:transform .05s,opacity .2s;}
 button:active{transform:scale(.98);} button:disabled{opacity:.5;cursor:not-allowed;}
 .b-snap{background:linear-gradient(135deg,#06b6d4,#3b82f6);}
 .b-pause{background:#334155;}
 .now{text-align:center;padding:6px 0 2px;}
 .now .emoji{font-size:46px;line-height:1;}
 .now .lab{font-size:30px;font-weight:800;text-transform:capitalize;margin-top:4px;}
 .now .route{display:inline-block;margin-top:8px;font-size:12px;font-weight:700;
   padding:4px 12px;border-radius:999px;}
 .stats{display:flex;justify-content:space-around;margin-top:12px;color:var(--mut);font-size:12px;}
 .stats b{color:#e2e8f0;font-size:15px;display:block;}
 .bars{margin-top:14px;}
 .bar{display:flex;align-items:center;gap:8px;margin:7px 0;font-size:12px;}
 .bar .name{width:74px;text-transform:capitalize;color:#cbd5e1;text-align:right;}
 .bar .track{flex:1;background:#0b1220;border-radius:6px;overflow:hidden;height:13px;}
 .bar .fill{height:100%;width:0;transition:width .25s ease;}
 .bar .pct{width:40px;color:var(--mut);text-align:right;}
 h3{margin:2px 0 10px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.6px;}
 #hist{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px;min-height:8px;}
 #hist .t{position:relative;flex:0 0 auto;}
 #hist img{height:70px;border-radius:8px;border:1px solid var(--line);display:block;}
 #hist .cap{position:absolute;bottom:3px;left:3px;right:3px;font-size:9px;font-weight:700;
   background:rgba(0,0,0,.6);border-radius:4px;text-align:center;text-transform:capitalize;padding:1px;}
 .hint{color:#64748b;font-size:11px;margin-top:8px;text-align:center;}
 kbd{background:#0b1220;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:10px;}
</style></head><body><div class="wrap">
 <header><h1>♻️ Sortitude</h1><span id="badge" class="pill">● LIVE</span></header>
 <div class="sub">Real-time waste classification on the Google Coral Edge TPU</div>
 <div class="grid">
  <div class="card">
   <div class="feedwrap">
     <img id="feed" src="/stream" onclick="snap(false)">
     <span id="lb" class="live-badge">● LIVE</span>
     <span id="toast" class="auto-toast">⚡ AUTO-SAVED</span>
   </div>
   <div class="controls">
     <button class="b-snap" onclick="snap(false)">📸 Snapshot <kbd>space</kbd></button>
     <button id="pz" class="b-pause" onclick="togglePause()">⏸ Pause <kbd>p</kbd></button>
   </div>
   <div class="hint">Click the video or press <kbd>space</kbd> to save a snapshot · <kbd>p</kbd> to pause</div>
  </div>
  <div class="card">
   <div class="now">
     <div id="emoji" class="emoji">·</div>
     <div id="lab" class="lab">—</div>
     <div id="route" class="route" style="background:#1e293b;color:#94a3b8;">waiting…</div>
   </div>
   <div class="stats">
     <div><b id="conf">—</b>confidence</div>
     <div><b id="ms">—</b>inference</div>
     <div><b id="fps">—</b>fps</div>
   </div>
   <div id="bars" class="bars"></div>
  </div>
 </div>
 <div class="card" style="margin-top:18px;">
   <h3>Snapshot history</h3>
   <div id="hist"></div>
 </div>
</div>
<script>
const ROUTING={cardboard:['📦','Recycling','#4ade80'],paper:['📄','Recycling','#4ade80'],
 glass:['🍶','Recycling','#38bdf8'],metal:['🥫','Recycling','#facc15'],
 plastic:['🧴','Recycling','#a78bfa'],trash:['🗑️','Landfill','#f87171']};
let paused=false, barsInit=false, names=[];
let lastAutoLabel='', lastAutoTs=0;

function ensureBars(scores){
 const box=document.getElementById('bars');
 if(barsInit) return;
 box.innerHTML=''; names=Object.keys(ROUTING);
 names.forEach(n=>{
   const c=ROUTING[n][2];
   const row=document.createElement('div'); row.className='bar'; row.id='bar-'+n;
   row.innerHTML=`<div class="name">${n}</div><div class="track"><div class="fill" id="fill-${n}" style="background:${c}"></div></div><div class="pct" id="pct-${n}">0%</div>`;
   box.appendChild(row);
 });
 barsInit=true;
}

async function poll(){
 try{
  const d=await (await fetch('/state')).json();
  document.getElementById('lab').textContent=d.label;
  document.getElementById('conf').textContent=(d.conf*100).toFixed(0)+'%';
  document.getElementById('ms').textContent=d.ms.toFixed(1)+'ms';
  document.getElementById('fps').textContent=d.fps.toFixed(0);
  const r=ROUTING[d.label]||['·','','#94a3b8'];
  document.getElementById('emoji').textContent=r[0];
  const route=document.getElementById('route');
  route.textContent=r[1]||'—'; route.style.background=r[2]+'22'; route.style.color=r[2];
  if(d.scores&&d.scores.length){ensureBars(d.scores);
   d.scores.forEach((p,i)=>{const n=d.names[i];
    const f=document.getElementById('fill-'+n), pc=document.getElementById('pct-'+n);
    if(f){f.style.width=(p*100).toFixed(0)+'%';} if(pc){pc.textContent=(p*100).toFixed(0)+'%';}});
  }
  // AUTO-CAPTURE: when a class hits 100%, save once per new class (1.5s debounce)
  if(!paused && d.conf>=0.999 && d.label!==lastAutoLabel && (Date.now()-lastAutoTs>1500)){
    lastAutoLabel=d.label; lastAutoTs=Date.now(); snap(true);
  } else if(d.conf<0.95){ lastAutoLabel=''; }
 }catch(e){}
}

function flashToast(){
  const t=document.getElementById('toast'); t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),900);
}

async function snap(auto){
 try{
  const d=await (await fetch('/snapshot?save=1'+(auto?'&auto=1':''))).json();
  if(!d.ok) return;
  if(auto) flashToast();
  const hist=document.getElementById('hist');
  const t=document.createElement('div'); t.className='t';
  const badge=auto?'<div class="autob">AUTO</div>':'';
  t.innerHTML=`<img src="data:image/jpeg;base64,${d.image}" title="${d.label} ${(d.conf*100).toFixed(0)}%"><div class="cap">${d.label} ${(d.conf*100).toFixed(0)}%</div>${badge}`;
  hist.prepend(t);
  while(hist.children.length>12) hist.removeChild(hist.lastChild);
 }catch(e){}
}

async function togglePause(){
 paused=!paused;
 await fetch('/control?action='+(paused?'pause':'resume'));
 const badge=document.getElementById('badge'), lb=document.getElementById('lb'),
       pz=document.getElementById('pz');
 if(paused){badge.textContent='⏸ PAUSED';badge.classList.add('paused');
   lb.textContent='⏸ PAUSED';lb.classList.add('paused');pz.textContent='▶ Resume';}
 else{badge.textContent='● LIVE';badge.classList.remove('paused');
   lb.textContent='● LIVE';lb.classList.remove('paused');pz.innerHTML='⏸ Pause <kbd>p</kbd>';}
}

document.addEventListener('keydown',e=>{
 if(e.code==='Space'){e.preventDefault();snap(false);}
 else if(e.key==='p'){togglePause();}
});
setInterval(poll,250); poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path.startswith("/index"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/state":
            d = LATEST
            body = json.dumps({"label": d["label"], "conf": d["conf"],
                               "scores": d["scores"], "names": LABELS,
                               "ms": d["ms"], "fps": d["fps"], "n": d["n"]}).encode()
            self._send(200, body, "application/json")
        elif path == "/snapshot":
            d = LATEST
            q = parse_qs(urlparse(self.path).query)
            do_save = q.get("save", ["0"])[0] == "1"
            is_auto = q.get("auto", ["0"])[0] == "1"
            if d["jpeg"]:
                if do_save:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    tag = "auto" if is_auto else "manual"
                    fn = os.path.join(SAVE_DIR, f"{tag}_{ts}_{d['label']}.jpg")
                    try:
                        with open(fn, "wb") as fh:
                            fh.write(d["jpeg"])
                    except Exception:
                        pass
                body = json.dumps({"ok": True, "label": d["label"], "conf": d["conf"],
                                   "image": base64.b64encode(d["jpeg"]).decode()}).encode()
            else:
                body = json.dumps({"ok": False}).encode()
            self._send(200, body, "application/json")
        elif path == "/control":
            global PAUSED
            q = parse_qs(urlparse(self.path).query)
            action = (q.get("action", [""])[0])
            PAUSED = (action == "pause")
            self._send(200, json.dumps({"paused": PAUSED}).encode(), "application/json")
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while RUN:
                    jpg = LATEST["jpeg"]
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode() +
                                         b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._send(404, b"not found", "text/plain")


def main():
    global INTERP, LABELS, CAM, INPUT_WH
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(args.labels) as f:
        LABELS = [ln.strip() for ln in f if ln.strip()]
    INTERP = make_interpreter(args.model)
    INTERP.allocate_tensors()
    INPUT_WH = common.input_size(INTERP)
    CAM = cv2.VideoCapture(args.camera)
    if not CAM.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    for _ in range(5):
        CAM.read(); time.sleep(0.05)

    threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Sortitude LIVE dashboard on http://0.0.0.0:{args.port}/  "
          f"(input {INPUT_WH}, {len(LABELS)} classes, camera {args.camera})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
