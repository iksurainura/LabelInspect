"""
LabelInspect — Real-Time Defect Detection
Full-page layout, no sidebar.
"""

import streamlit as st
import numpy as np
import cv2
import time
import serial
import serial.tools.list_ports
from collections import deque
from datetime import datetime
from ultralytics import YOLO
import pandas as pd
import config

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="LabelInspect",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with open("styles.css", encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ================================================================
# SESSION STATE
# ================================================================
def init():
    defaults = {
        "running":           False,
        "conf":              config.DEFAULT_CONF_THRESHOLD,
        "iou":               config.DEFAULT_IOU_THRESHOLD,
        "confirm_time":      config.CONFIRMATION_TIME,
        "camera_index":      config.CAMERA_INDEX,
        "fps_target":        config.FPS_TARGET,
        "video_width":       config.VIDEO_WIDTH,
        "video_height":      config.VIDEO_HEIGHT,
        "confirmed_output":  "IDLE",
        "pending_output":    None,
        "pending_since":     None,
        "output_history":    deque(maxlen=100),
        "ser":               None,
        "connected":         False,
        "last_cmd":          None,
        "last_response":     None,
        "servo_pos":         90,
        "ir_triggered":      False,
        "pending_servo_cmd": None,
        "frame_count":       0,
        "stats":             {"defect": 0, "no_defect": 0, "idle": 0, "total": 0},
        # settings panel state
        "camera_index_tmp":  config.CAMERA_INDEX,
        "fps_tmp":           config.FPS_TARGET,
        "vw_tmp":            config.VIDEO_WIDTH,
        "vh_tmp":            config.VIDEO_HEIGHT,
        "model_path_tmp":    config.MODEL_PATH,
        "device_tmp":        config.DEVICE,
        "conf_tmp":          config.DEFAULT_CONF_THRESHOLD,
        "iou_tmp":           config.DEFAULT_IOU_THRESHOLD,
        "confirm_tmp":       config.CONFIRMATION_TIME,
        "serial_port_tmp":   config.SERIAL_PORT,
        "baud_tmp":          9600,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()

# ================================================================
# ARDUINO
# ================================================================
def connect_arduino(port, baud):
    try:
        if st.session_state.ser and st.session_state.ser.is_open:
            st.session_state.ser.close()
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.5)
        st.session_state.ser       = ser
        st.session_state.connected = True
        st.session_state.last_response = f"OK  {port} @ {baud}"
        return True
    except Exception as e:
        st.session_state.last_response = f"ERR  {e}"
        return False

def disconnect_arduino():
    if st.session_state.ser and st.session_state.ser.is_open:
        st.session_state.ser.close()
    st.session_state.ser       = None
    st.session_state.connected = False
    st.session_state.last_response = "Disconnected"

def send_servo(cmd):
    if not st.session_state.ser or not st.session_state.ser.is_open:
        st.session_state.last_response = "Not connected"
        return False
    try:
        st.session_state.ser.write(cmd.encode())
        st.session_state.ser.flush()
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.last_cmd = f"{ts}  TX  {cmd}"
        if cmd == "L":   st.session_state.servo_pos = 0
        elif cmd == "C": st.session_state.servo_pos = 90
        elif cmd == "R": st.session_state.servo_pos = 180
        time.sleep(0.05)
        if st.session_state.ser.in_waiting:
            r = st.session_state.ser.readline().decode("utf-8", errors="ignore").strip()
            if r: st.session_state.last_response = r
        return True
    except Exception as e:
        st.session_state.last_response = f"TX FAIL  {e}"
        return False

def poll_ir():
    if not st.session_state.ser or not st.session_state.ser.is_open:
        return False
    try:
        while st.session_state.ser.in_waiting:
            ch = st.session_state.ser.read(1).decode("utf-8", errors="ignore")
            if ch == config.IR_TRIGGER_CHAR:
                st.session_state.ir_triggered = True
                if st.session_state.pending_servo_cmd:
                    send_servo(st.session_state.pending_servo_cmd)
                    st.session_state.pending_servo_cmd = None
                return True
    except Exception:
        pass
    return False

def list_ports():
    return [p.device for p in serial.tools.list_ports.comports()]

# ================================================================
# DETECTION LOGIC
# ================================================================
def confirm(detections):
    now      = time.time()
    ct       = st.session_state.confirm_time

    if not detections:
        cur  = "IDLE"
        info = {"label":"NO DETECTION","badge":"pill-muted","cmd":None,
                "angle":90,"color":config.THEME["text_muted"],"cls":"sc-idle"}
    elif any(d["class_id"] in config.DEFECT_CLASSES for d in detections):
        cur  = "DEFECT"
        info = {"label":"DEFECT DETECTED","badge":"pill-red","cmd":config.SERVO_DEFECT_CMD,
                "angle":0,"color":config.THEME["red"],"cls":"sc-defect"}
    else:
        cur  = "NO_DEFECT"
        info = {"label":"NO DEFECT","badge":"pill-green","cmd":config.SERVO_OK_CMD,
                "angle":90,"color":config.THEME["green"],"cls":"sc-ok"}

    is_new = False
    prog   = 0.0
    if st.session_state.pending_output == cur:
        elapsed = now - (st.session_state.pending_since or now)
        prog    = min(elapsed / ct, 1.0)
        if elapsed >= ct and st.session_state.confirmed_output != cur:
            st.session_state.confirmed_output = cur
            if info["cmd"]:
                st.session_state.pending_servo_cmd = info["cmd"]
            is_new = True
    else:
        st.session_state.pending_output = cur
        st.session_state.pending_since  = now

    return st.session_state.confirmed_output, is_new, info, prog

# ================================================================
# MODEL
# ================================================================
@st.cache_resource
def load_model(path, dev):
    try:
        m = YOLO(path)
        m.to(dev)
        return m
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

# ================================================================
# HTML HELPERS
# ================================================================
def sh(label):
    return f'<div class="sh"><span class="sh-text">{label}</span><div class="sh-line"></div></div>'

def pill(text, cls="pill-muted", dot=False, pulse=False):
    d = f'<span class="dot{"dot-pulse" if pulse else ""}"></span>' if dot else ""
    return f'<span class="pill {cls}">{d}{text}</span>'

def logline(key, val):
    return f'<div class="logline"><span class="lkey">{key}</span><span class="lval">{val}</span></div>'

# ================================================================
# TOPBAR
# ================================================================
run_cls  = "run"  if st.session_state.running   else "stop"
ard_cls  = "ok"   if st.session_state.connected else "err"
ir_cls   = "warn" if st.session_state.ir_triggered else "stop"

status_txt = "RUNNING"    if st.session_state.running   else "STOPPED"
ard_txt    = "ARDUINO OK" if st.session_state.connected else "NO ARDUINO"
ir_txt     = "IR ACTIVE"  if st.session_state.ir_triggered else "IR IDLE"

st.markdown(f"""
<div class="topbar">
  <div class="tb-brand">
    <div class="tb-logo">🔍</div>
    <div>
      <div class="tb-name">LabelInspect</div>
      <div class="tb-sub">AI Visual Inspection &nbsp;·&nbsp; YOLO &nbsp;·&nbsp; Arduino Servo + IR Gate</div>
    </div>
  </div>
  <div class="tb-right">
    {pill(status_txt, "pill-green" if st.session_state.running else "pill-muted", dot=True, pulse=st.session_state.running)}
    {pill(ard_txt,    "pill-green" if st.session_state.connected else "pill-red", dot=True)}
    {pill(ir_txt,     "pill-orange" if st.session_state.ir_triggered else "pill-muted", dot=True)}
  </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# CONTROLS ROW
# ================================================================
cc1, cc2, cc3 = st.columns([1, 1, 6])
with cc1:
    if st.button("▶  START DETECTION", use_container_width=True, type="primary",
                 disabled=st.session_state.running):
        st.session_state.running      = True
        st.session_state.ir_triggered = False
        st.rerun()
with cc2:
    if st.button("⏹  STOP DETECTION", use_container_width=True,
                 disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ================================================================
# MAIN ROW — Feed | Status + Servo + IR
# ================================================================
model_path = st.session_state.model_path_tmp
device     = st.session_state.device_tmp
model      = load_model(model_path, device)

feed_col, right_col = st.columns([3, 1], gap="large")

with feed_col:
    st.markdown(sh("Live Feed"), unsafe_allow_html=True)
    video_ph = st.empty()

with right_col:
    st.markdown(sh("Detection Status"), unsafe_allow_html=True)
    status_ph = st.empty()
    servo_ph  = st.empty()
    ir_ph     = st.empty()

    st.markdown(sh("Metrics"), unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1: fps_ph  = st.empty()
    with m2: det_ph  = st.empty()
    with m3: cf_ph   = st.empty()

# ================================================================
# HISTORY + STATS ROW
# ================================================================
st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
hist_col, stat_col = st.columns([3, 1], gap="large")

with hist_col:
    st.markdown(sh("Detection History"), unsafe_allow_html=True)
    table_ph = st.empty()
    table_ph.dataframe(
        pd.DataFrame(columns=["Time","Status","Confidence","Objects","Cmd","Angle"]),
        use_container_width=True, height=220
    )

with stat_col:
    st.markdown(sh("Statistics"), unsafe_allow_html=True)
    stats_ph = st.empty()
    s = st.session_state.stats
    stats_ph.markdown(f"""
<div class="stats-grid">
  <div class="sg-cell sg-total"><div class="sg-val">{s['total']}</div><div class="sg-lbl">Total</div></div>
  <div class="sg-cell sg-defect"><div class="sg-val">{s['defect']}</div><div class="sg-lbl">Defect</div></div>
  <div class="sg-cell sg-good"><div class="sg-val">{s['no_defect']}</div><div class="sg-lbl">Good</div></div>
  <div class="sg-cell sg-idle"><div class="sg-val">{s['idle']}</div><div class="sg-lbl">Idle</div></div>
</div>""", unsafe_allow_html=True)

# ================================================================
# SETTINGS PANEL — Tabbed, all inline
# ================================================================
st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
st.markdown(sh("Settings"), unsafe_allow_html=True)

tab_cam, tab_model, tab_thresh, tab_ard, tab_servo, tab_ir = st.tabs([
    "📷  Camera",
    "🧠  Model",
    "🎯  Thresholds",
    "🔌  Arduino",
    "🎛  Servo Debug",
    "🔴  IR Gate",
])

# ── Camera tab ────────────────────────────────────────────────
with tab_cam:
    tc1, tc2, tc3 = st.columns([1,1,2])
    with tc1:
        cam_idx = st.number_input("Camera Index", 0, 10,
                                  st.session_state.camera_index_tmp,
                                  help="0 = default, 1 = USB, etc.")
        st.session_state.camera_index_tmp = cam_idx
        st.session_state.camera_index     = cam_idx
    with tc2:
        fps_v = st.slider("Target FPS", 1, 30, st.session_state.fps_tmp)
        st.session_state.fps_tmp    = fps_v
        st.session_state.fps_target = fps_v
    with tc3:
        st.markdown("**Resolution**")
        tr1, tr2, tr3 = st.columns(3)
        with tr1:
            if st.button("640 x 480", use_container_width=True):
                st.session_state.vw_tmp = 640; st.session_state.vh_tmp = 480
                st.session_state.video_width = 640; st.session_state.video_height = 480
        with tr2:
            if st.button("960 x 540", use_container_width=True, type="primary"):
                st.session_state.vw_tmp = 960; st.session_state.vh_tmp = 540
                st.session_state.video_width = 960; st.session_state.video_height = 540
        with tr3:
            if st.button("1280 x 720", use_container_width=True):
                st.session_state.vw_tmp = 1280; st.session_state.vh_tmp = 720
                st.session_state.video_width = 1280; st.session_state.video_height = 720
        st.markdown(
            f'<div style="margin-top:8px;">'
            + logline("Active", f"{st.session_state.vw_tmp} x {st.session_state.vh_tmp}  @  {fps_v} fps")
            + '</div>',
            unsafe_allow_html=True
        )

# ── Model tab ─────────────────────────────────────────────────
with tab_model:
    tm1, tm2 = st.columns(2)
    with tm1:
        mp = st.text_input("Model Path", st.session_state.model_path_tmp,
                           help="Relative or absolute path to .pt file")
        st.session_state.model_path_tmp = mp
    with tm2:
        dv = st.selectbox("Inference Device", ["cuda","cpu"],
                          index=0 if st.session_state.device_tmp == "cuda" else 1,
                          help="cuda = GPU, cpu = fallback")
        st.session_state.device_tmp = dv
    st.markdown(
        logline("Model", mp) + logline("Device", dv.upper()),
        unsafe_allow_html=True
    )

# ── Thresholds tab ────────────────────────────────────────────
with tab_thresh:
    tt1, tt2, tt3 = st.columns(3)
    with tt1:
        cv = st.slider("Confidence", 0.01, 1.0, st.session_state.conf_tmp, 0.01)
        st.session_state.conf_tmp = cv; st.session_state.conf = cv
    with tt2:
        iv = st.slider("IOU / NMS",  0.01, 1.0, st.session_state.iou_tmp,  0.01)
        st.session_state.iou_tmp = iv; st.session_state.iou = iv
    with tt3:
        ctv = st.slider("Confirm Time (s)", 0.5, 5.0,
                         st.session_state.confirm_tmp, 0.5,
                         help="Seconds detection must stay stable before state commits")
        st.session_state.confirm_tmp  = ctv
        st.session_state.confirm_time = ctv

# ── Arduino tab ───────────────────────────────────────────────
with tab_ard:
    _cls = "ok" if st.session_state.connected else "no"
    _txt = "CONNECTED" if st.session_state.connected else "DISCONNECTED"
    _pinfo = f"  {st.session_state.ser.port}" if st.session_state.connected and st.session_state.ser else ""
    st.markdown(
        f'<div class="conn-banner {_cls}">'
        f'<span class="conn-dot {"blink" if st.session_state.connected else ""}"></span>'
        f'{_txt}{_pinfo}</div>',
        unsafe_allow_html=True
    )

    ta1, ta2, ta3, ta4 = st.columns([2,1,1,1])
    with ta1:
        avail = list_ports()
        if avail:
            sp = st.selectbox("Serial Port", avail,
                              index=avail.index(config.SERIAL_PORT) if config.SERIAL_PORT in avail else 0)
        else:
            sp = st.text_input("Serial Port", st.session_state.serial_port_tmp)
            st.caption("No ports detected — enter manually")
        st.session_state.serial_port_tmp = sp
    with ta2:
        bd = st.selectbox("Baud Rate", [9600, 115200], index=0)
        st.session_state.baud_tmp = bd
    with ta3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Connect", use_container_width=True, type="primary"):
            if connect_arduino(sp, bd):
                st.toast("Arduino connected", icon="🔗"); st.rerun()
            else:
                st.error("Connection failed")
    with ta4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Disconnect", use_container_width=True):
            disconnect_arduino(); st.rerun()

    st.markdown(
        logline("TX", st.session_state.last_cmd or "---") +
        logline("RX", st.session_state.last_response or "---"),
        unsafe_allow_html=True
    )

# ── Servo Debug tab ───────────────────────────────────────────
with tab_servo:
    pos      = st.session_state.servo_pos
    pos_pct  = (pos / 180) * 100
    pcol     = config.THEME["red"]   if pos == 0  else \
               config.THEME["green"] if pos == 90 else \
               config.THEME["blue"]
    pname    = "LEFT" if pos == 0 else "CENTER" if pos == 90 else "RIGHT"

    ts1, ts2 = st.columns([1,2])
    with ts1:
        st.markdown(f"""
<div class="servo-card">
  <div class="sv-header">
    <span class="sv-title">Current Position</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px;">
    <span class="sv-val" style="color:{pcol};">{pos}&deg;</span>
    <span style="font-family:var(--mono);font-size:11px;color:var(--t-secondary);">{pname}</span>
  </div>
  <div class="sv-track">
    <div class="sv-thumb" style="left:calc({pos_pct:.1f}% - 8px);background:{pcol};color:{pcol};"></div>
  </div>
  <div class="sv-ticks"><span>0&deg;</span><span>90&deg;</span><span>180&deg;</span></div>
</div>
        """, unsafe_allow_html=True)

    with ts2:
        st.markdown("**Manual Control**")
        _dis = not st.session_state.connected
        sm1, sm2, sm3 = st.columns(3)
        with sm1:
            if st.button("LEFT  0deg",   key="sv_l", use_container_width=True, disabled=_dis):
                send_servo("L"); st.rerun()
        with sm2:
            if st.button("CENTER  90deg",key="sv_c", use_container_width=True, disabled=_dis):
                send_servo("C"); st.rerun()
        with sm3:
            if st.button("RIGHT  180deg",key="sv_r", use_container_width=True, disabled=_dis):
                send_servo("R"); st.rerun()

        st.markdown(
            logline("TX", st.session_state.last_cmd or "---") +
            logline("RX", st.session_state.last_response or "---"),
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run L > C > R Test Sequence", use_container_width=True, disabled=_dis):
            with st.spinner("Testing..."):
                for c in ("L","C","R","C"):
                    send_servo(c); time.sleep(0.6)
            st.toast("Sequence done", icon="✅"); st.rerun()

# ── IR Gate tab ───────────────────────────────────────────────
with tab_ir:
    ti1, ti2 = st.columns([1,2])
    with ti1:
        ir_on  = st.session_state.ir_triggered
        pend   = st.session_state.pending_servo_cmd
        ir_cls2= "ir-hot" if ir_on else ""
        led_c  = "on" if ir_on else "off"
        ir_col = config.THEME["orange"] if ir_on else config.THEME["text_muted"]
        ir_lbl = "TAG IN POSITION" if ir_on else "WAITING FOR TAG"
        pend_h = f'<div class="ir-pending">PENDING &nbsp; {pend} &nbsp; — awaiting IR</div>' if pend else ""

        st.markdown(f"""
<div class="ir-card {ir_cls2}">
  <div class="ir-row">
    <span class="ir-led {led_c}"></span>
    <span class="ir-label" style="color:{ir_col};">{ir_lbl}</span>
  </div>
  <div class="ir-note">
    Servo fires only when the IR sensor detects the belt tag.<br>
    Arduino must send <code>{config.IR_TRIGGER_CHAR}</code> on detection.
  </div>
  {pend_h}
</div>
        """, unsafe_allow_html=True)

    with ti2:
        st.markdown(f"""
**How it works**

1. Camera detects a defect or good part continuously for **{st.session_state.confirm_time:.1f}s**
2. State commits → servo command is queued (not sent yet)
3. When the belt tag passes the IR sensor, Arduino sends `{config.IR_TRIGGER_CHAR}` over serial
4. The queued command fires immediately — servo moves at exactly the right moment

**Wiring**
- IR sensor OUT → Arduino digital pin
- Servo signal → Pin 9, 5V → Red, GND → Black
- Close Arduino IDE Serial Monitor before connecting here

**Serial test (Linux)**
```bash
ls /dev/ttyACM*
sudo usermod -a -G dialout $USER
```
        """)

# ================================================================
# SYSTEM STATUS BAR
# ================================================================
st.markdown(f"""
<div class="sysbar">
  <div class="sb-item {run_cls}">
    <span class="sb-dot"></span>
    <span class="sb-text">{"RUNNING" if st.session_state.running else "STOPPED"}</span>
  </div>
  <span class="sb-sep">|</span>
  <div class="sb-item {ard_cls}">
    <span class="sb-dot"></span>
    <span class="sb-text">{"ARDUINO" if st.session_state.connected else "NO ARDUINO"}</span>
  </div>
  <span class="sb-sep">|</span>
  <div class="sb-item {ir_cls}">
    <span class="sb-dot"></span>
    <span class="sb-text">{"IR ACTIVE" if st.session_state.ir_triggered else "IR IDLE"}</span>
  </div>
  <span class="sb-sep">|</span>
  <div class="sb-item">
    <span class="sb-text">{st.session_state.video_width}x{st.session_state.video_height} @ {st.session_state.fps_target}fps</span>
  </div>
  <span class="sb-sep">|</span>
  <div class="sb-item">
    <span class="sb-text">{model_path}  [{device.upper()}]</span>
  </div>
  <span class="sb-sep">|</span>
  <div class="sb-item">
    <span class="sb-text">conf {st.session_state.conf:.2f}  iou {st.session_state.iou:.2f}  confirm {st.session_state.confirm_time:.1f}s</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# IDLE PLACEHOLDERS
# ================================================================
if not st.session_state.running:
    W = st.session_state.video_width
    H = min(st.session_state.video_height, 480)

    video_ph.markdown(f"""
<div class="cam-idle" style="width:{W}px;height:{H}px;max-width:100%;">
  <div class="ci-icon">📷</div>
  <div class="ci-title">Detection Paused</div>
  <div class="ci-sub">cam {st.session_state.camera_index} &nbsp;·&nbsp;
    {W}x{st.session_state.video_height} &nbsp;·&nbsp; {st.session_state.fps_target}fps</div>
</div>""", unsafe_allow_html=True)

    status_ph.markdown("""
<div class="status-card sc-idle">
  <div class="sc-eyebrow">System Output</div>
  <div style="margin-bottom:8px;">
    <span class="pill pill-muted"><span class="dot"></span>IDLE</span>
  </div>
  <div class="sc-state" style="color:var(--t-muted);">IDLE</div>
  <div class="sc-sub">Press START to begin</div>
</div>""", unsafe_allow_html=True)

    p2 = st.session_state.servo_pos
    p2c = ((p2/180)*100)
    pc2 = config.THEME["red"] if p2==0 else config.THEME["green"] if p2==90 else config.THEME["blue"]
    servo_ph.markdown(f"""
<div class="servo-card" style="margin-top:10px;">
  <div class="sv-header">
    <span class="sv-title">Servo</span>
    <span style="font-family:var(--mono);font-size:10px;color:var(--t-muted);">IDLE</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px;">
    <span class="sv-val" style="color:{pc2};">{p2}&deg;</span>
    <span style="font-family:var(--mono);font-size:11px;color:var(--t-secondary);">
      {"LEFT" if p2==0 else "CENTER" if p2==90 else "RIGHT"}
    </span>
  </div>
  <div class="sv-track">
    <div class="sv-thumb" style="left:calc({p2c:.1f}% - 8px);background:{pc2};color:{pc2};"></div>
  </div>
  <div class="sv-ticks"><span>0&deg;</span><span>90&deg;</span><span>180&deg;</span></div>
</div>""", unsafe_allow_html=True)

    ir_ph.markdown(f"""
<div class="ir-card" style="margin-top:10px;">
  <div class="ir-row">
    <span class="ir-led off"></span>
    <span class="ir-label" style="color:var(--t-muted);">WAITING FOR TAG</span>
  </div>
  <div class="ir-note">Servo fires on IR trigger <code>{config.IR_TRIGGER_CHAR}</code></div>
</div>""", unsafe_allow_html=True)

    fps_ph.metric("FPS",    "—")
    det_ph.metric("Det.",   "—")
    cf_ph.metric("Conf.",  "—")

# ================================================================
# MAIN PROCESSING LOOP
# ================================================================
if st.session_state.running and model:
    cap = cv2.VideoCapture(st.session_state.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  st.session_state.video_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, st.session_state.video_height)

    if not cap.isOpened():
        st.error("Failed to open camera — check the index in Camera settings.")
        st.session_state.running = False
    else:
        while st.session_state.running:
            t0 = time.time()

            poll_ir()

            ret, frame = cap.read()
            if not ret:
                st.warning("Frame capture failed.")
                break
            frame = cv2.resize(frame, (st.session_state.video_width, st.session_state.video_height))

            results = model.predict(
                frame,
                conf=st.session_state.conf,
                iou=st.session_state.iou,
                device=device,
                verbose=False,
            )
            result = results[0]

            dets = []
            if result.boxes:
                for box in result.boxes:
                    dets.append({"class_id": int(box.cls[0]),
                                 "confidence": float(box.conf[0])})

            confirmed, is_new, info, prog = confirm(dets)

            # ── Video ──────────────────────────────────────────
            ann = result.plot(boxes=True, labels=True, conf=True, line_width=2)
            ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            video_ph.image(ann, channels="RGB",
                           use_container_width=False,
                           width=st.session_state.video_width)

            # ── Status Card ────────────────────────────────────
            avg_c    = np.mean([d["confidence"] for d in dets]) if dets else 0.0
            bar_pct  = prog * 100
            cstate   = ("Confirming..." if 0 < prog < 1.0 and dets
                        else "Stable" if confirmed != "IDLE" else "Idle")

            status_ph.markdown(f"""
<div class="status-card {info['cls']}">
  <div class="sc-eyebrow">System Output</div>
  <div style="margin-bottom:8px;">
    <span class="pill {info['badge']}"><span class="dot"></span>{info['label']}</span>
  </div>
  <div class="sc-state" style="color:{info['color']};">{confirmed}</div>
  <div class="sc-sub">conf&nbsp;{avg_c:.2f} &nbsp;&bull;&nbsp; {len(dets)} obj</div>
  <div class="cbar-wrap">
    <div class="cbar-track">
      <div class="cbar-fill" style="width:{bar_pct:.0f}%;background:{info['color']};"></div>
    </div>
    <div class="cbar-label">
      <span>{cstate}</span><span>{bar_pct:.0f}%</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Servo ──────────────────────────────────────────
            p     = st.session_state.servo_pos
            pp    = (p / 180) * 100
            pc    = config.THEME["red"]   if p==0  else \
                    config.THEME["green"] if p==90 else \
                    config.THEME["blue"]
            pn    = "LEFT" if p==0 else "CENTER" if p==90 else "RIGHT"
            pend  = st.session_state.pending_servo_cmd
            ph2   = f'<div class="ir-pending">PENDING &nbsp;{pend}&nbsp; — awaiting IR</div>' if pend else ""
            ir2   = st.session_state.ir_triggered
            irl   = "TAG ACTIVE" if ir2 else "NO TAG"
            irc   = config.THEME["orange"] if ir2 else config.THEME["text_muted"]

            servo_ph.markdown(f"""
<div class="servo-card" style="margin-top:10px;">
  <div class="sv-header">
    <span class="sv-title">Servo</span>
    <span style="font-family:var(--mono);font-size:9px;color:{irc};">&bull; {irl}</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px;">
    <span class="sv-val" style="color:{pc};">{p}&deg;</span>
    <span style="font-family:var(--mono);font-size:11px;color:var(--t-secondary);">{pn}</span>
  </div>
  <div class="sv-track">
    <div class="sv-thumb" style="left:calc({pp:.1f}% - 8px);background:{pc};color:{pc};"></div>
  </div>
  <div class="sv-ticks"><span>0&deg;</span><span>90&deg;</span><span>180&deg;</span></div>
  {ph2}
</div>""", unsafe_allow_html=True)

            # ── IR ─────────────────────────────────────────────
            ir_on2 = st.session_state.ir_triggered
            ir_ph.markdown(f"""
<div class="ir-card {"ir-hot" if ir_on2 else ""}" style="margin-top:10px;">
  <div class="ir-row">
    <span class="ir-led {"on" if ir_on2 else "off"}"></span>
    <span class="ir-label" style="color:{config.THEME["orange"] if ir_on2 else config.THEME["text_muted"]};">
      {"TAG IN POSITION" if ir_on2 else "WAITING FOR TAG"}
    </span>
  </div>
  <div class="ir-note">Trigger char: <code>{config.IR_TRIGGER_CHAR}</code></div>
</div>""", unsafe_allow_html=True)

            # ── Metrics ────────────────────────────────────────
            fps = 1.0 / max(time.time() - t0, 1e-9)
            fps_ph.metric("FPS",   f"{fps:.1f}")
            det_ph.metric("Det.",  len(dets))
            cf_ph.metric("Conf.", f"{avg_c:.2f}")

            # ── History ────────────────────────────────────────
            if is_new or st.session_state.frame_count % 5 == 0:
                entry = {
                    "Time":       datetime.now().strftime("%H:%M:%S"),
                    "Status":     confirmed,
                    "Confidence": f"{avg_c:.2f}",
                    "Objects":    len(dets),
                    "Cmd":        info["cmd"] or "---",
                    "Angle":      info["angle"],
                }
                st.session_state.output_history.append(entry)
                df = pd.DataFrame(list(st.session_state.output_history))
                table_ph.dataframe(df.tail(15), use_container_width=True, height=220)

                if is_new:
                    key = {"DEFECT":"defect","NO_DEFECT":"no_defect",
                           "IDLE":"idle"}.get(confirmed,"idle")
                    st.session_state.stats[key]    += 1
                    st.session_state.stats["total"] += 1

            # ── Stats ──────────────────────────────────────────
            s2 = st.session_state.stats
            stats_ph.markdown(f"""
<div class="stats-grid">
  <div class="sg-cell sg-total"><div class="sg-val">{s2['total']}</div><div class="sg-lbl">Total</div></div>
  <div class="sg-cell sg-defect"><div class="sg-val">{s2['defect']}</div><div class="sg-lbl">Defect</div></div>
  <div class="sg-cell sg-good"><div class="sg-val">{s2['no_defect']}</div><div class="sg-lbl">Good</div></div>
  <div class="sg-cell sg-idle"><div class="sg-val">{s2['idle']}</div><div class="sg-lbl">Idle</div></div>
</div>""", unsafe_allow_html=True)

            st.session_state.frame_count += 1

            # ── FPS cap ────────────────────────────────────────
            wait = (1.0 / st.session_state.fps_target) - (time.time() - t0)
            if wait > 0: time.sleep(wait)

        cap.release()

# ================================================================
# FOOTER
# ================================================================
st.markdown(f"""
<div class="footer">
  LabelInspect &nbsp;&bull;&nbsp;
  {model_path} &nbsp;&bull;&nbsp; {device.upper()} &nbsp;&bull;&nbsp;
  {st.session_state.video_width}x{st.session_state.video_height} @ {st.session_state.fps_target}fps &nbsp;&bull;&nbsp;
  {config.SERIAL_PORT} @ {config.SERIAL_BAUDRATE} &nbsp;&bull;&nbsp;
  IR char: <code style="font-size:9px;">{config.IR_TRIGGER_CHAR}</code>
</div>
""", unsafe_allow_html=True)