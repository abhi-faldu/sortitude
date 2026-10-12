#!/usr/bin/env python3
"""Sortitude interactive web dashboard for the Coral Edge TPU — v2 HUD.

Background thread captures webcam frames and runs Edge TPU inference, serving a
live annotated MJPEG feed plus fast state-polls. Stdlib-only HTTP server.

v2 HUD additions (same backend pipeline):
  - Confidence sparkline  -> /state "conf_hist" (rolling deque of confidences)
  - Per-class counts      -> /state "counts" (debounced finalised classifications)

Endpoints:
  /          interactive HTML UI (v2 HUD)
  /stream    multipart MJPEG live feed (annotated)
  /state     JSON: label, conf, per-class scores, ms, fps, conf_hist, counts
  /snapshot  JSON: frozen annotated frame (base64) + prediction
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

# v2 HUD rolling state
CONF_HIST = deque(maxlen=120)   # ~last 30s of confidences (for the sparkline)
COUNTS = {}                     # running total per class (filled in main)
_count_last = ""                # last counted label (debounce)
_count_ts = 0.0

# class -> (emoji, bin/category, css color) — used for the baked-in frame label
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


def _update_counts(label, conf):
    """Count a NEW finalised classification: a high-confidence label that differs
    from the last counted one (1s debounce). Re-counts the same class after the
    frame clears (conf drops), so showing different items each bumps a counter."""
    global _count_last, _count_ts
    now = time.time()
    if conf >= 0.85 and label != _count_last and (now - _count_ts) > 1.0:
        COUNTS[label] = COUNTS.get(label, 0) + 1
        _count_last = label
        _count_ts = now
    elif conf < 0.55:
        _count_last = ""


def worker():
    """Continuously grab frames, run inference, update LATEST + HUD rolling state."""
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

        # HUD rolling state
        CONF_HIST.append(conf)
        _update_counts(label, conf)

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


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Sortitude — Live Waste Classifier</title>
<style>
 :root{--g:#22e36a;--g2:#16c95b;--bg:#000000;--panel:rgba(10,26,16,.55);
   --line:rgba(34,227,106,.28);--mut:#7fae90;--ink:#d6ffe6;
   --amber:#fbbf24;--red:#f87171;}
 *{box-sizing:border-box;}
 body{margin:0;min-height:100vh;font-family:"Segoe UI",system-ui,Roboto,sans-serif;
   color:var(--ink);background:#000000;}
 .frame{max-width:1320px;margin:22px auto;padding:22px;border:1px solid var(--line);
   border-radius:26px;position:relative;
   box-shadow:0 0 0 1px rgba(34,227,106,.06),0 0 60px rgba(34,227,106,.10) inset,
     0 30px 80px rgba(0,0,0,.6);background:#000000;}
 .frame::before,.frame::after{content:"";position:absolute;width:46px;height:46px;
   border:2px solid var(--g);border-radius:4px;opacity:.7;}
 .frame::before{top:10px;left:10px;border-right:none;border-bottom:none;}
 .frame::after{bottom:10px;right:10px;border-left:none;border-top:none;}
 header{display:flex;align-items:center;justify-content:space-between;
   padding:6px 8px 18px;border-bottom:1px solid var(--line);margin-bottom:18px;}
 .brand{display:flex;align-items:center;gap:14px;}
 .logo{width:46px;height:46px;border-radius:12px;display:grid;place-items:center;
   background:radial-gradient(circle at 30% 30%,#1f7a45,#0c3d22);
   border:1px solid var(--line);font-size:26px;color:var(--g);
   box-shadow:0 0 18px rgba(34,227,106,.35);font-weight:900;}
 .brand h1{margin:0;font-size:26px;font-weight:800;letter-spacing:3px;color:var(--g);
   text-shadow:0 0 14px rgba(34,227,106,.5);}
 .brand .tag{font-size:11px;letter-spacing:3px;color:var(--mut);margin-top:2px;}
 .livepill{display:flex;align-items:center;gap:9px;font-weight:800;letter-spacing:2px;
   padding:9px 22px;border:1px solid var(--line);border-radius:10px;font-size:15px;
   background:rgba(34,227,106,.06);}
 .livepill .dot{width:9px;height:9px;border-radius:50%;background:var(--g);
   box-shadow:0 0 10px var(--g);animation:blink 1.4s infinite;}
 .livepill.paused{color:var(--amber);border-color:#5a3a17;}
 .livepill.paused .dot{background:var(--amber);box-shadow:0 0 10px var(--amber);}
 @keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
 .meta{font-size:12px;letter-spacing:1px;line-height:1.7;text-align:right;}
 .meta .k{color:var(--mut);margin-right:10px;}
 .meta .v{color:var(--ink);}
 .grid{display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:18px;}
 @media(max-width:980px){.grid{grid-template-columns:1fr;}}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;
   padding:16px;backdrop-filter:blur(4px);
   box-shadow:0 0 30px rgba(0,0,0,.4),0 0 0 1px rgba(34,227,106,.04) inset;}
 .ct{font-size:12px;letter-spacing:2px;color:var(--mut);text-transform:uppercase;
   display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}
 .ct .mini{display:flex;align-items:center;gap:7px;color:var(--g);font-weight:800;}
 .ct .mini .dot{width:8px;height:8px;border-radius:50%;background:var(--g);
   box-shadow:0 0 8px var(--g);animation:blink 1.4s infinite;}
 .feed{position:relative;aspect-ratio:16/11;border-radius:12px;overflow:hidden;
   background:linear-gradient(160deg,#0a140d,#050a07);
   border:1px solid var(--line);display:grid;place-items:center;}
 .feed .feedimg{width:100%;height:100%;object-fit:cover;display:block;background:#000;}
 .note{font-size:10px;letter-spacing:1px;color:var(--mut);text-align:center;margin-top:9px;}
 .pred .label{font-size:40px;font-weight:800;color:var(--g);letter-spacing:.5px;
   text-transform:capitalize;text-shadow:0 0 18px rgba(34,227,106,.45);
   display:flex;align-items:center;justify-content:space-between;}
 .pred .label .ico{font-size:38px;}
 .sub{font-size:12px;letter-spacing:2px;color:var(--mut);text-transform:uppercase;
   margin:18px 0 6px;}
 .conf{font-size:62px;font-weight:900;color:var(--g);line-height:1;
   text-shadow:0 0 22px rgba(34,227,106,.5);}
 .cbar{height:12px;border-radius:8px;background:rgba(255,255,255,.06);overflow:hidden;
   margin-top:14px;border:1px solid var(--line);}
 .cbar i{display:block;height:100%;width:0;border-radius:8px;
   background:linear-gradient(90deg,var(--g2),var(--g));transition:width .3s ease;
   box-shadow:0 0 12px rgba(34,227,106,.6);}
 .badge{display:inline-flex;align-items:center;gap:8px;margin-top:14px;font-weight:700;
   font-size:14px;}
 .badge.high{color:var(--g);} .badge.mod{color:var(--amber);} .badge.low{color:var(--red);}
 .spark-wrap{margin-top:16px;border:1px solid var(--line);border-radius:10px;
   padding:8px 10px;background:rgba(0,0,0,.25);}
 .spark-wrap .lbl{font-size:10px;letter-spacing:2px;color:var(--mut);
   display:flex;justify-content:space-between;margin-bottom:4px;}
 canvas.spark{display:block;width:100%;height:120px;}
 .probs{display:flex;flex-direction:column;}
 .probs #bars{flex:1;display:flex;flex-direction:column;justify-content:space-around;
   padding:4px 0;}
 .bar{display:flex;align-items:center;gap:12px;margin:0;font-size:14px;}
 .bar .ico{width:24px;text-align:center;font-size:18px;}
 .bar .nm{width:82px;text-transform:capitalize;color:var(--ink);}
 .bar .tk{flex:1;height:14px;border-radius:7px;background:rgba(255,255,255,.06);
   overflow:hidden;border:1px solid var(--line);}
 .bar .fl{height:100%;width:0;border-radius:7px;transition:width .3s ease;}
 .bar .pc{width:54px;text-align:right;color:var(--g);font-weight:700;}
 .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-top:18px;}
 @media(max-width:980px){.stats{grid-template-columns:1fr 1fr;}}
 .stat{background:var(--panel);border:1px solid var(--line);border-radius:16px;
   padding:16px 18px;display:flex;align-items:center;gap:16px;}
 .stat .sico{font-size:30px;color:var(--g);filter:drop-shadow(0 0 8px rgba(34,227,106,.4));}
 .stat .k{font-size:11px;letter-spacing:2px;color:var(--mut);text-transform:uppercase;}
 .stat .v{font-size:30px;font-weight:800;color:var(--g);line-height:1.05;}
 .stat .d{font-size:12px;color:var(--mut);margin-top:2px;}
 .controls{display:flex;gap:10px;margin-top:14px;}
 .controls button{flex:1;padding:11px;font-weight:700;border-radius:10px;cursor:pointer;
   border:1px solid var(--line);background:rgba(34,227,106,.08);color:var(--g);
   letter-spacing:1px;font-size:13px;}
 .controls button:active{transform:scale(.98);}
 .bottom{margin-top:18px;}
 .counts{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;}
 @media(max-width:980px){.counts{grid-template-columns:repeat(3,1fr);}}
 @media(max-width:560px){.counts{grid-template-columns:repeat(2,1fr);}}
 .cnt{border:1px solid var(--line);border-radius:12px;padding:14px 10px;text-align:center;
   background:rgba(0,0,0,.25);transition:transform .25s,box-shadow .25s;}
 .cnt.bump{transform:scale(1.07);box-shadow:0 0 18px rgba(34,227,106,.5);}
 .cnt .e{font-size:26px;line-height:1;}
 .cnt .n{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:var(--mut);
   margin-top:6px;}
 .cnt .v{font-size:30px;font-weight:800;color:var(--g);margin-top:4px;line-height:1;}
</style></head><body>
<div class="frame">
 <header>
   <div class="brand">
     <div class="logo">♻</div>
     <div><h1>THE SORTITUDE</h1><div class="tag">AI WASTE CLASSIFICATION</div></div>
   </div>
   <div id="livepill" class="livepill"><span class="dot"></span>LIVE</div>
   <div class="meta">
     <div><span class="k">MODEL</span><span class="v">MobileNetV2 · int8 (Edge TPU)</span></div>
     <div><span class="k">DEVICE</span><span class="v">Google Coral Dev Board</span></div>
   </div>
 </header>

 <div class="grid">
   <div class="card">
     <div class="ct">LIVE CAMERA FEED <span class="mini"><span class="dot"></span>LIVE</span></div>
     <div class="feed">
       <img id="feed" class="feedimg" src="/stream" alt="live feed">
     </div>
     <div class="note">Live annotated feed from the Coral Edge TPU · press P to pause</div>
     <div class="controls">
       <button onclick="togglePause()" id="pz">⏸ PAUSE</button>
     </div>
   </div>

   <div class="card pred">
     <div class="ct">PREDICTED CLASS</div>
     <div class="label"><span id="lab">—</span><span id="ico" class="ico">·</span></div>
     <div class="sub">Confidence Score</div>
     <div id="conf" class="conf">—</div>
     <div class="cbar"><i id="cfill"></i></div>
     <div class="badge high" id="badge">🛡 <span id="cmsg">Waiting…</span></div>
     <div class="spark-wrap">
       <div class="lbl"><span>CONFIDENCE · LAST 30s</span><span id="sparkNow">—</span></div>
       <canvas id="spark" class="spark"></canvas>
     </div>
   </div>

   <div class="card probs">
     <div class="ct">CLASS PROBABILITIES</div>
     <div id="bars"></div>
   </div>
 </div>

 <div class="stats">
   <div class="stat"><div class="sico">⏱</div><div>
     <div class="k">Inference Time</div><div class="v"><span id="ms">—</span>ms</div>
     <div class="d" id="msd">Very Fast</div></div></div>
   <div class="stat"><div class="sico">⚡</div><div>
     <div class="k">FPS</div><div class="v" id="fps">—</div>
     <div class="d">Real-time</div></div></div>
   <div class="stat"><div class="sico">📈</div><div>
     <div class="k">Status</div><div class="v" id="status">LIVE</div>
     <div class="d">System Online</div></div></div>
   <div class="stat"><div class="sico">🎯</div><div>
     <div class="k">Avg Confidence</div><div class="v"><span id="avg">—</span>%</div>
     <div class="d">last 30 frames</div></div></div>
 </div>

 <div class="bottom">
   <div class="card">
     <div class="ct">CLASSIFICATION COUNTS
       <span class="mini">SESSION TOTAL · <span id="total">0</span></span></div>
     <div id="counts" class="counts"></div>
   </div>
 </div>
</div>

<script>
const ROUTING={cardboard:['📦','Recycle Bin','#4ade80'],paper:['📄','Recycle Bin','#a3e635'],
 glass:['🍶','Recycle Bin','#38bdf8'],metal:['🥫','Recycle Bin','#facc15'],
 plastic:['🧴','Recycle Bin','#a78bfa'],trash:['🗑️','Landfill','#f87171']};
let paused=false, barsInit=false, confWin=[], countEls={}, prevCounts={};

function ensureBars(){
 if(barsInit) return; const box=document.getElementById('bars'); box.innerHTML='';
 Object.keys(ROUTING).forEach(n=>{
   const c=ROUTING[n][2];
   const row=document.createElement('div'); row.className='bar';
   row.innerHTML=`<div class="ico">${ROUTING[n][0]}</div><div class="nm">${n}</div>
     <div class="tk"><div class="fl" id="fl-${n}" style="background:${c}"></div></div>
     <div class="pc" id="pc-${n}" style="color:${c}">0%</div>`;
   box.appendChild(row);
 });
 barsInit=true;
}

function drawSpark(vals){
 const cv=document.getElementById('spark'); const ctx=cv.getContext('2d');
 const dpr=devicePixelRatio||1; const W=cv.clientWidth, H=cv.clientHeight;
 cv.width=W*dpr; cv.height=H*dpr; ctx.scale(dpr,dpr);
 ctx.clearRect(0,0,W,H);
 const LO=0.5, HI=1.0, GUT=30;
 const yOf=v=>H-((Math.max(LO,Math.min(HI,v))-LO)/(HI-LO))*H;
 const xOf=(i,n)=>GUT+(n>1?(i/(n-1)):0)*(W-GUT);
 ctx.font='9px Segoe UI';
 [[1.0,'100'],[0.85,'85'],[0.75,'75'],[0.6,'60'],[0.5,'50']].forEach(([g,lbl])=>{
   const yp=yOf(g);
   ctx.strokeStyle='rgba(127,174,144,.18)'; ctx.lineWidth=1; ctx.setLineDash([3,4]);
   ctx.beginPath(); ctx.moveTo(GUT,yp); ctx.lineTo(W,yp); ctx.stroke(); ctx.setLineDash([]);
   ctx.fillStyle='rgba(127,174,144,.75)'; ctx.fillText(lbl+'%',2,Math.min(Math.max(yp+3,8),H-2));
 });
 if(vals.length<2) return;
 const n=vals.length;
 const grad=ctx.createLinearGradient(0,0,0,H);
 grad.addColorStop(0,'rgba(34,227,106,.5)'); grad.addColorStop(1,'rgba(34,227,106,0)');
 ctx.fillStyle=grad; ctx.beginPath(); ctx.moveTo(GUT,H);
 vals.forEach((v,i)=>ctx.lineTo(xOf(i,n),yOf(v)));
 ctx.lineTo(W,H); ctx.closePath(); ctx.fill();
 ctx.strokeStyle='#22e36a'; ctx.lineWidth=2; ctx.beginPath();
 vals.forEach((v,i)=>{const x=xOf(i,n),y=yOf(v); if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);});
 ctx.stroke();
}

function renderCounts(counts){
 const box=document.getElementById('counts');
 if(!Object.keys(countEls).length){
   box.innerHTML='';
   Object.keys(ROUTING).forEach(n=>{
     const r=ROUTING[n];
     const el=document.createElement('div'); el.className='cnt';
     el.innerHTML=`<div class="e">${r[0]}</div><div class="n">${n}</div>
       <div class="v" style="color:${r[2]}">0</div>`;
     box.appendChild(el); countEls[n]=el;
   });
 }
 let total=0;
 Object.keys(ROUTING).forEach(n=>{
   const c=counts[n]||0; total+=c;
   const v=countEls[n].querySelector('.v');
   if(prevCounts[n]!==undefined && c>prevCounts[n]){
     countEls[n].classList.add('bump');
     setTimeout(()=>countEls[n].classList.remove('bump'),350);
   }
   v.textContent=c; prevCounts[n]=c;
 });
 document.getElementById('total').textContent=total;
}

async function poll(){
 if(paused) return;
 try{
  const d=await (await fetch('/state')).json();
  const r=ROUTING[d.label]||['·','—','#22e36a'];
  document.getElementById('lab').textContent=d.label;
  document.getElementById('ico').textContent=r[0];
  const pct=(d.conf*100);
  document.getElementById('conf').textContent=pct.toFixed(1)+'%';
  document.getElementById('cfill').style.width=pct.toFixed(1)+'%';
  const badge=document.getElementById('badge'), msg=document.getElementById('cmsg');
  badge.classList.remove('high','mod','low');
  if(d.conf>=0.85){badge.classList.add('high');msg.textContent='High Confidence';}
  else if(d.conf>=0.6){badge.classList.add('mod');msg.textContent='Moderate Confidence';}
  else{badge.classList.add('low');msg.textContent='Low Confidence';}
  document.getElementById('ms').textContent=d.ms.toFixed(0);
  document.getElementById('msd').textContent=d.ms<15?'Very Fast':d.ms<25?'Fast':'Degraded';
  document.getElementById('fps').textContent=d.fps.toFixed(0);
  confWin.push(d.conf); if(confWin.length>30) confWin.shift();
  const avg=confWin.reduce((a,b)=>a+b,0)/confWin.length;
  document.getElementById('avg').textContent=(avg*100).toFixed(0);
  ensureBars();
  d.scores.forEach((p,i)=>{const n=d.names[i];
    const f=document.getElementById('fl-'+n), pc=document.getElementById('pc-'+n);
    if(f) f.style.width=(p*100).toFixed(0)+'%';
    if(pc) pc.textContent=(p*100).toFixed(1)+'%';});
  document.getElementById('sparkNow').textContent=pct.toFixed(1)+'%';
  drawSpark(d.conf_hist||[]);
  renderCounts(d.counts||{});
 }catch(e){}
}

function togglePause(){
 paused=!paused;
 fetch('/control?action='+(paused?'pause':'resume')).catch(()=>{});
 const pill=document.getElementById('livepill'), pz=document.getElementById('pz'),
       st=document.getElementById('status');
 if(paused){pill.classList.add('paused');pill.childNodes[1].textContent='PAUSED';
   pz.textContent='▶ RESUME';st.textContent='PAUSED';}
 else{pill.classList.remove('paused');pill.childNodes[1].textContent='LIVE';
   pz.textContent='⏸ PAUSE';st.textContent='LIVE';}
}
document.addEventListener('keydown',e=>{if(e.key==='p')togglePause();});
window.addEventListener('resize',()=>{poll();});
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
                               "ms": d["ms"], "fps": d["fps"], "n": d["n"],
                               "conf_hist": list(CONF_HIST),
                               "counts": COUNTS}).encode()
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
    global INTERP, LABELS, CAM, INPUT_WH, COUNTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(args.labels) as f:
        LABELS = [ln.strip() for ln in f if ln.strip()]
    COUNTS = {lab: 0 for lab in LABELS}
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
    print(f"Sortitude v2 HUD dashboard on http://0.0.0.0:{args.port}/  "
          f"(input {INPUT_WH}, {len(LABELS)} classes, camera {args.camera})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
