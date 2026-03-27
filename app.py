import cv2
import time
import math
import random
import threading
from datetime import datetime
from flask import Flask, Response, jsonify, render_template_string

app = Flask(__name__)

VIDEO_PATH = "1.mp4"
FRAME_RATE = 30

latest_frame = None
frame_lock = threading.Lock()

camera_state = {
    "CAM-001": {"floodLevel": "Dry", "depthCm": 0, "confidence": 0, "rateOfRise": 0, "timeToCritical": None},
    "CAM-002": {"floodLevel": "Dry", "depthCm": 0, "confidence": 0, "rateOfRise": 0, "timeToCritical": None},
}

depth_history = {"CAM-001": [], "CAM-002": []}

CAMERAS = [
    {"id": "CAM-001", "name": "CCTV 1"},
    {"id": "CAM-002", "name": "CCTV 2"},
]

FLOOD_LEVELS = ["Dry", "Ankle Level", "Knee Level", "Waist Level", "Impassable"]
LEVEL_COLORS_BGR = {
    "Dry":        (86, 197, 34),
    "Ankle Level":(8,  179, 234),
    "Knee Level": (22, 115, 249),
    "Waist Level":(68,  68, 239),
    "Impassable": (50,  38, 220),
}

try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("[BahaAlerto] YOLOv8 model loaded.")
except Exception as e:
    model = None
    print(f"[BahaAlerto] YOLOv8 not available: {e}")

def simulate_flood(cam_id):
    seed = ord(cam_id[-1])
    t = time.time() / 60
    depth = max(0, (math.sin(t * 0.4 + seed) + 1) * 35 + random.gauss(0, 2))
    confidence = round(random.uniform(0.72, 0.96), 2)
    if depth >= 90:   level = "Impassable"
    elif depth >= 50: level = "Waist Level"
    elif depth >= 25: level = "Knee Level"
    elif depth >= 5:  level = "Ankle Level"
    else:             level = "Dry"
    return {"floodLevel": level, "depthCm": round(depth, 1), "confidence": confidence}

def classify_depth(depth_cm):
    if depth_cm >= 90:   return "Impassable"
    elif depth_cm >= 50: return "Waist Level"
    elif depth_cm >= 25: return "Knee Level"
    elif depth_cm >= 5:  return "Ankle Level"
    else:                return "Dry"

def video_thread():
    global latest_frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[BahaAlerto] Could not open video: {VIDEO_PATH}")
        return
    print("[BahaAlerto] Video feed started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        flood_depth = 0
        flood_level = "Dry"
        confidence  = 0.0

        if model is not None:
            results = model(frame, conf=0.4, verbose=False)
            annotated = results[0].plot()

            boxes = results[0].boxes
            person_detections = []
            for box in boxes:
                cls = int(box.cls[0])
                conf_val = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                h_box = xyxy[3] - xyxy[1]
                if cls == 0:
                    person_detections.append({"height": h_box, "conf": conf_val})

            if person_detections:
                avg_h = sum(p["height"] for p in person_detections) / len(person_detections)
                ratio = avg_h / frame.shape[0]
                flood_depth = max(0, (1 - ratio) * 80 + random.gauss(0, 3))
                confidence  = max(p["conf"] for p in person_detections)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = gray.mean()
                flood_depth = max(0, (1 - brightness / 255) * 60 + random.gauss(0, 5))
                confidence  = round(random.uniform(0.65, 0.88), 2)

            flood_level = classify_depth(flood_depth)
            color_bgr = LEVEL_COLORS_BGR.get(flood_level, (86, 197, 34))

            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (0, 0, 0), -1)
            timestamp = datetime.now().strftime("%H:%M:%S") + " PHL | CCTV 1"
            cv2.putText(annotated, timestamp, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1)
            cv2.putText(annotated, "REC", (annotated.shape[1] - 55, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 220), 1)

            label = f"FLOOD: {flood_level.upper()}  {flood_depth:.1f}cm  CONF:{confidence:.0%}"
            cv2.rectangle(annotated, (0, annotated.shape[0] - 40), (annotated.shape[1], annotated.shape[0]), (0, 0, 0), -1)
            cv2.putText(annotated, label, (10, annotated.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            cv2.putText(annotated, "YOLOv8n | BahaAlerto", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 160, 255), 1)

        else:
            annotated = frame.copy()
            cv2.putText(annotated, "YOLOv8 not loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = buffer.tobytes()

        prev_depth = camera_state["CAM-001"]["depthCm"]
        rate = round((flood_depth - prev_depth) / 0.5, 2)
        ttc  = round((50 - flood_depth) / rate, 1) if rate > 0 and flood_depth < 50 else None
        camera_state["CAM-001"].update({
            "floodLevel": flood_level,
            "depthCm":    round(flood_depth, 1),
            "confidence": round(confidence, 2),
            "rateOfRise": rate,
            "timeToCritical": ttc,
        })
        depth_history["CAM-001"].append(round(flood_depth, 1))
        if len(depth_history["CAM-001"]) > 30:
            depth_history["CAM-001"].pop(0)

        time.sleep(1 / FRAME_RATE)

    cap.release()

def sim_thread():
    while True:
        result = simulate_flood("CAM-002")
        prev   = camera_state["CAM-002"]["depthCm"]
        rate   = round((result["depthCm"] - prev) / 0.5, 2)
        ttc    = round((50 - result["depthCm"]) / rate, 1) if rate > 0 and result["depthCm"] < 50 else None
        camera_state["CAM-002"].update({**result, "rateOfRise": rate, "timeToCritical": ttc})
        depth_history["CAM-002"].append(result["depthCm"])
        if len(depth_history["CAM-002"]) > 30:
            depth_history["CAM-002"].pop(0)
        time.sleep(3)

def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(1 / FRAME_RATE)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    return jsonify({
        "cameras": camera_state,
        "depthHistory": depth_history,
    })

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BahaAlerto</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#0a0f1a; --bg2:#0f1728;
  --surface:#1a2640; --surface2:#1f2e4a;
  --border:rgba(99,162,255,0.12); --border2:rgba(99,162,255,0.22);
  --text:#e8edf8; --muted:#8899bb; --dim:#4a5a7a;
  --accent:#3b82f6; --accent2:#60a5fa;
  --dry:#22c55e; --ankle:#eab308; --knee:#f97316; --waist:#ef4444; --impassable:#dc2626;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--text); font-family:'Barlow',sans-serif; height:100vh; overflow:hidden; }
body::before { content:''; position:fixed; inset:0; background-image:linear-gradient(rgba(59,130,246,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(59,130,246,0.03) 1px,transparent 1px); background-size:40px 40px; pointer-events:none; z-index:0; }

header { background:rgba(10,15,26,0.95); border-bottom:1px solid var(--border2); padding:0 24px; display:flex; align-items:center; justify-content:space-between; height:52px; position:relative; z-index:10; }
.logo { display:flex; align-items:center; gap:10px; }
.logo-icon { width:30px; height:30px; background:var(--accent); border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:15px; }
.logo-text { font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:22px; letter-spacing:1px; }
.logo-text span { color:var(--accent2); }
.live-badge { display:flex; align-items:center; gap:6px; font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--dry); background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.2); padding:4px 10px; border-radius:4px; }
.live-dot { width:6px; height:6px; background:var(--dry); border-radius:50%; animation:pulse 1.5s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

.stat-bar { display:grid; grid-template-columns:repeat(3,1fr); border-bottom:1px solid var(--border); background:var(--bg2); position:relative; z-index:10; }
.stat-item { padding:10px 24px; border-right:1px solid var(--border); }
.stat-item:last-child { border-right:none; }
.stat-label { font-size:10px; letter-spacing:1.2px; text-transform:uppercase; color:var(--dim); font-weight:600; }
.stat-value { font-family:'IBM Plex Mono',monospace; font-size:20px; font-weight:500; color:var(--text); line-height:1.2; }
.ok{color:var(--dry)} .warn{color:var(--ankle)} .danger{color:var(--waist)}

.main { display:grid; grid-template-columns:200px 1fr; height:calc(100vh - 52px - 48px); position:relative; z-index:1; overflow:hidden; }

.panel-left { border-right:1px solid var(--border); background:var(--bg2); display:flex; flex-direction:column; }
.panel-header { padding:12px 16px; border-bottom:1px solid var(--border); font-size:10px; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); font-weight:600; }

.camera-node { padding:14px 16px; border-bottom:1px solid var(--border); }
.camera-node.active { background:var(--surface); border-left:3px solid var(--accent); }
.camera-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
.camera-name { font-family:'Barlow Condensed',sans-serif; font-size:17px; font-weight:700; letter-spacing:0.5px; }
.flood-badge { font-family:'IBM Plex Mono',monospace; font-size:10px; padding:2px 7px; border-radius:3px; letter-spacing:0.5px; text-transform:uppercase; }
.badge-dry        { background:rgba(34,197,94,0.12);  color:var(--dry);   border:1px solid rgba(34,197,94,0.25); }
.badge-ankle      { background:rgba(234,179,8,0.12);  color:var(--ankle); border:1px solid rgba(234,179,8,0.25); }
.badge-knee       { background:rgba(249,115,22,0.12); color:var(--knee);  border:1px solid rgba(249,115,22,0.25); }
.badge-waist      { background:rgba(239,68,68,0.12);  color:var(--waist); border:1px solid rgba(239,68,68,0.25); }
.badge-impassable { background:rgba(220,38,38,0.2);   color:#ff6b6b;      border:1px solid rgba(220,38,38,0.4); }
.depth-bar-wrap { height:3px; background:var(--surface2); border-radius:2px; overflow:hidden; margin-bottom:6px; }
.depth-bar { height:100%; border-radius:2px; transition:width 1s ease; }
.camera-meta { display:flex; flex-direction:column; gap:3px; font-family:'IBM Plex Mono',monospace; font-size:10px; color:var(--dim); }

.panel-center { display:flex; flex-direction:column; background:var(--bg); overflow:hidden; }
.feed-wrap { flex:1; position:relative; background:#000; overflow:hidden; display:flex; align-items:center; justify-content:center; }
.feed-wrap img { width:100%; height:100%; object-fit:contain; display:block; }
.feed-hud { position:absolute; top:10px; left:10px; display:flex; flex-direction:column; gap:5px; pointer-events:none; }
.hud-chip { background:rgba(0,0,0,0.7); border:1px solid var(--border2); padding:3px 9px; border-radius:4px; font-family:'IBM Plex Mono',monospace; font-size:10px; color:var(--text); }
.hud-chip.model { color:var(--accent2); }
.scan-line { position:absolute; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,rgba(59,130,246,0.5),transparent); animation:scan 3s linear infinite; pointer-events:none; }
@keyframes scan { 0%{top:0;opacity:0} 5%{opacity:1} 95%{opacity:1} 100%{top:100%;opacity:0} }

.feed-bottom { padding:8px 14px; background:var(--bg2); border-top:1px solid var(--border); display:flex; align-items:center; gap:14px; }
.inf-bar { width:100px; height:4px; background:var(--surface2); border-radius:2px; overflow:hidden; }
.inf-fill { height:100%; background:var(--accent); border-radius:2px; animation:inf 3s ease-in-out infinite; }
@keyframes inf { 0%{width:0%} 50%{width:80%} 90%{width:95%} 100%{width:0%} }
.depth-readout { font-family:'Barlow Condensed',sans-serif; font-size:26px; font-weight:700; letter-spacing:1px; margin-left:auto; transition:color 0.5s; }

.chart-panel { padding:10px 14px; border-top:1px solid var(--border); background:var(--bg2); height:110px; }
.chart-label { font-size:10px; text-transform:uppercase; letter-spacing:1px; color:var(--dim); font-weight:600; margin-bottom:4px; }

::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:transparent} ::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:2px}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🌊</div>
    <span class="logo-text">BAHA<span>ALERTO</span></span>
  </div>
  <div style="display:flex;align-items:center;gap:16px">
    <div class="live-badge"><div class="live-dot"></div>YOLOv8 LIVE</div>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted)" id="clock">--:--:--</span>
  </div>
</header>

<div class="stat-bar">
  <div class="stat-item"><div class="stat-label">Cameras Online</div><div class="stat-value ok">2 / 2</div></div>
  <div class="stat-item"><div class="stat-label">Flooded</div><div class="stat-value" id="sFlooded">0</div></div>
  <div class="stat-item"><div class="stat-label">Model</div><div class="stat-value" style="font-size:13px;padding-top:4px;color:var(--muted)">YOLOv8n Live</div></div>
</div>

<div class="main">

  <div class="panel-left">
    <div class="panel-header">Camera Nodes</div>
    <div id="camList"></div>
  </div>

  <div class="panel-center">
    <div class="feed-wrap">
      <img id="videoFeed" src="/video_feed" alt="Live Feed">
      <div class="scan-line"></div>
      <div class="feed-hud">
        <div class="hud-chip model">MODEL: YOLOv8n</div>
        <div class="hud-chip">CCTV 1</div>
        <div class="hud-chip" id="hudConf">CONF: --</div>
      </div>
    </div>
    <div class="feed-bottom">
      <div style="display:flex;align-items:center;gap:8px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted)">
        <div class="inf-bar"><div class="inf-fill"></div></div>
        Running YOLOv8 inference...
      </div>
      <span style="font-size:11px;color:var(--muted)">CCTV 1</span>
      <div class="depth-readout" id="depthReadout" style="color:var(--dry)">DRY</div>
    </div>
    <div class="chart-panel">
      <div class="chart-label">Depth History — CCTV 1</div>
      <canvas id="chartCanvas" style="width:100%;height:72px"></canvas>
    </div>
  </div>

</div>

<script>
const CAMERAS = [
  {id:"CAM-001", name:"CCTV 1"},
  {id:"CAM-002", name:"CCTV 2"},
];
const LEVEL_COLORS = {Dry:"#22c55e","Ankle Level":"#eab308","Knee Level":"#f97316","Waist Level":"#ef4444",Impassable:"#dc2626"};
const BADGE = {Dry:"badge-dry","Ankle Level":"badge-ankle","Knee Level":"badge-knee","Waist Level":"badge-waist",Impassable:"badge-impassable"};

let chartCtx, chartCanvas;

function initChart() {
  chartCanvas = document.getElementById("chartCanvas");
  chartCtx = chartCanvas.getContext("2d");
}

function drawChart(history) {
  const w = chartCanvas.parentElement.clientWidth - 28;
  const h = 72;
  chartCanvas.width = w; chartCanvas.height = h;
  chartCtx.clearRect(0, 0, w, h);
  if (!history || history.length < 2) return;

  [5, 25, 50].forEach(d => {
    const y = h - (d / 120) * h;
    chartCtx.strokeStyle = "rgba(99,162,255,0.1)"; chartCtx.lineWidth = 1;
    chartCtx.setLineDash([4,4]); chartCtx.beginPath();
    chartCtx.moveTo(0, y); chartCtx.lineTo(w, y); chartCtx.stroke();
    chartCtx.setLineDash([]);
  });

  chartCtx.beginPath();
  history.forEach((d, i) => {
    const x = (i / (history.length - 1)) * w;
    const y = h - (Math.min(d, 120) / 120) * h;
    i === 0 ? chartCtx.moveTo(x, y) : chartCtx.lineTo(x, y);
  });
  chartCtx.strokeStyle = "#3b82f6"; chartCtx.lineWidth = 2; chartCtx.stroke();
  chartCtx.lineTo(w, h); chartCtx.lineTo(0, h); chartCtx.closePath();
  const grad = chartCtx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(59,130,246,0.25)"); grad.addColorStop(1, "rgba(59,130,246,0)");
  chartCtx.fillStyle = grad; chartCtx.fill();
}

function renderCameras(cameras) {
  const list = document.getElementById("camList");
  list.innerHTML = "";
  CAMERAS.forEach(cam => {
    const s = cameras[cam.id] || {};
    const level = s.floodLevel || "Dry";
    const depth = s.depthCm || 0;
    const pct = Math.min((depth / 120) * 100, 100);
    const color = LEVEL_COLORS[level] || "#22c55e";
    const div = document.createElement("div");
    div.className = "camera-node" + (cam.id === "CAM-001" ? " active" : "");
    div.innerHTML = `
      <div class="camera-top">
        <span class="camera-name">${cam.name}</span>
        <span class="flood-badge ${BADGE[level]}">${level === "Dry" ? "DRY" : level.toUpperCase()}</span>
      </div>
      <div class="depth-bar-wrap"><div class="depth-bar" style="width:${pct}%;background:${color}"></div></div>
      <div class="camera-meta">
        <span>${depth.toFixed(1)} cm depth</span>
        ${s.rateOfRise > 0 ? `<span style="color:${color}">&#8679; ${s.rateOfRise.toFixed(1)} cm/min</span>` : ""}
        ${s.timeToCritical ? `<span style="color:#ef4444">&#9888; ${s.timeToCritical} min to critical</span>` : ""}
        <span>conf: ${Math.round((s.confidence||0)*100)}%</span>
      </div>`;
    list.appendChild(div);
  });
}

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const cam1 = data.cameras["CAM-001"] || {};
    const level = cam1.floodLevel || "Dry";
    const color = LEVEL_COLORS[level] || "#22c55e";

    document.getElementById("depthReadout").textContent = level === "Dry" ? "DRY" : `${(cam1.depthCm||0).toFixed(1)}cm`;
    document.getElementById("depthReadout").style.color = color;
    document.getElementById("hudConf").textContent = `CONF: ${Math.round((cam1.confidence||0)*100)}%`;

    const flooded = Object.values(data.cameras).filter(s => s.floodLevel !== "Dry").length;
    document.getElementById("sFlooded").textContent = flooded;
    document.getElementById("sFlooded").className = "stat-value " + (flooded > 0 ? "warn" : "ok");

    renderCameras(data.cameras);
    drawChart(data.depthHistory["CAM-001"] || []);
  } catch(e) {}
}

setInterval(() => { document.getElementById("clock").textContent = new Date().toLocaleTimeString("en-PH", {hour12:false}); }, 1000);
initChart();
fetchStatus();
setInterval(fetchStatus, 2000);
</script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)

if __name__ == "__main__":
    threading.Thread(target=video_thread, daemon=True).start()
    threading.Thread(target=sim_thread,   daemon=True).start()
    print("[BahaAlerto] Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
