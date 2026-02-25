"""
LabelInspect — Real-Time Defect Detection
Full-page layout, no sidebar.
Arduino protocol: single char  'O' = OK label, 'D' = DEFECT label, 'R' = Reset
Two IR sensors + two servos + conveyor relay.
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
        # detection runtime
        "running":           False,
        "conf":              config.DEFAULT_CONF_THRESHOLD,
        "iou":               config.DEFAULT_IOU_THRESHOLD,
        "confirm_time":      config.CONFIRMATION_TIME,
        "camera_index":      config.CAMERA_INDEX,
        "fps_target":        config.FPS_TARGET,
        "video_width":       config.VIDEO_WIDTH,
        "video_height":      config.VIDEO_HEIGHT,
        # state machine
        "confirmed_output":  "IDLE",
        "pending_output":    None,
        "pending_since":     None,
        "last_sent_cmd":     None,
        # history
        "output_history":    deque(maxlen=100),
        "stats":             {"defect": 0, "no_defect": 0, "idle": 0, "total": 0},
        "frame_count":       0,
        # serial
        "ser":               None,
        "connected":         False,
        "last_tx":           None,
        "last_rx":           None,
        # hardware state (live, updated from Arduino serial responses)
        "conveyor_on":       False,
        "ir_ok_triggered":   False,
        "ir_def_triggered":  False,
        "servo_ok_active":   False,
        "servo_def_active":  False,
        # settings (editable via tabs)
        "model_path":        config.MODEL_PATH,
        "device":            config.DEVICE,
        "serial_port":       config.SERIAL_PORT,
        "baud":              config.SERIAL_BAUDRATE,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()

# ================================================================
# SERIAL / ARDUINO
# ================================================================
def connect_arduino(port, baud):
    try:
        if st.session_state.ser and st.session_state.ser.is_open:
            st.session_state.ser.close()
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.5)
        st.session_state.ser       = ser
        st.session_state.connected = True
        st.session_state.last_rx   = "Connected"
        return True
    except Exception as e:
        st.session_state.last_rx = f"ERR: {e}"
        return False

def disconnect_arduino():
    if st.session_state.ser and st.session_state.ser.is_open:
        st.session_state.ser.close()
    st.session_state.ser       = None
    st.session_state.connected = False
    st.session_state.last_rx   = "Disconnected"

def send_cmd(char):
    """Send a single character command to Arduino."""
    if not st.session_state.ser or not st.session_state.ser.is_open:
        st.session_state.last_rx = "Not connected"
        return False
    try:
        st.session_state.ser.write(char.encode())
        st.session_state.ser.flush()
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.last_tx       = f"{ts}  ->  '{char}'"
        st.session_state.last_sent_cmd = char
        return True
    except Exception as e:
        st.session_state.last_rx = f"TX FAIL: {e}"
        return False

def poll_serial():
    """
    Non-blocking read of all pending Arduino responses.
    Parses known ACK strings and updates hardware state flags.
    """
    if not st.session_state.ser or not st.session_state.ser.is_open:
        return
    try:
        while st.session_state.ser.in_waiting:
            raw = st.session_state.ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue
            st.session_state.last_rx = raw

            if raw == "READY":
                st.session_state.conveyor_on = True

            elif raw == "ACK:OK":
                # Arduino received 'O' — conveyor stopping, waiting for IR-OK
                st.session_state.conveyor_on      = False
                st.session_state.ir_ok_triggered  = False
                st.session_state.servo_ok_active  = False

            elif raw == "ACK:DEFECT":
                # Arduino received 'D' — conveyor stopping, waiting for IR-DEFECT
                st.session_state.conveyor_on      = False
                st.session_state.ir_def_triggered = False
                st.session_state.servo_def_active = False

            elif raw == "ACK:RESET":
                st.session_state.conveyor_on      = True
                st.session_state.ir_ok_triggered  = False
                st.session_state.ir_def_triggered = False
                st.session_state.servo_ok_active  = False
                st.session_state.servo_def_active = False

            elif raw == "IR:OK":
                # IR-OK sensor triggered — servoOK sweeping
                st.session_state.ir_ok_triggered  = True
                st.session_state.servo_ok_active  = True

            elif raw == "IR:DEFECT":
                # IR-DEFECT sensor triggered — servoDEFECT sweeping
                st.session_state.ir_def_triggered = True
                st.session_state.servo_def_active = True

    except Exception:
        pass

def list_ports():
    return [p.device for p in serial.tools.list_ports.comports()]

# ================================================================
# DETECTION / CONFIRMATION STATE MACHINE
# ================================================================
def check_confirm(detections):
    """
    Returns: confirmed_state, is_new_confirmation, info_dict, progress_0_to_1
    When a state is stable for confirm_time seconds:
      - DEFECT   -> sends 'D' to Arduino
      - NO_DEFECT -> sends 'O' to Arduino
      - IDLE     -> no command
    """
    now = time.time()
    ct  = st.session_state.confirm_time

    if not detections:
        cur  = "IDLE"
        info = {
            "label": "NO DETECTION",
            "badge": "pill-muted",
            "cmd":   None,
            "color": config.THEME["t_muted"],
            "cls":   "sc-idle",
        }
    elif any(d["class_id"] in config.DEFECT_CLASSES for d in detections):
        cur  = "DEFECT"
        info = {
            "label": "DEFECT DETECTED",
            "badge": "pill-red",
            "cmd":   config.CMD_DEFECT,
            "color": config.THEME["red"],
            "cls":   "sc-defect",
        }
    else:
        cur  = "NO_DEFECT"
        info = {
            "label": "NO DEFECT",
            "badge": "pill-green",
            "cmd":   config.CMD_OK,
            "color": config.THEME["green"],
            "cls":   "sc-ok",
        }

    is_new = False
    prog   = 0.0

    if st.session_state.pending_output == cur:
        elapsed = now - (st.session_state.pending_since or now)
        prog    = min(elapsed / ct, 1.0)
        if elapsed >= ct and st.session_state.confirmed_output != cur:
            st.session_state.confirmed_output = cur
            if info["cmd"]:
                send_cmd(info["cmd"])
            is_new = True
    else:
        st.session_state.pending_output = cur
        st.session_state.pending_since  = now

    return st.session_state.confirmed_output, is_new, info, prog

# ================================================================
# MODEL LOADER
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
    """Section header with label + extending rule line."""
    return (
        f'<div class="sh">'
        f'<span class="sh-text">{label}</span>'
        f'<div class="sh-line"></div>'
        f'</div>'
    )

def pill(text, cls="pill-muted", dot=True, pulse=False):
    d = f'<span class="dot{" dot-pulse" if pulse else ""}"></span>' if dot else ""
    return f'<span class="pill {cls}">{d}{text}</span>'

def logline(k, v):
    return (
        f'<div class="logline">'
        f'<span class="lk">{k}</span>'
        f'<span class="lv">{v}</span>'
        f'</div>'
    )

def hw_row(label, led_cls, val_text, val_color="var(--t-pri)"):
    return (
        f'<div class="hw-row">'
        f'<span class="hw-row-label">'
        f'<span class="hw-led {led_cls}"></span>{label}</span>'
        f'<span class="hw-val" style="color:{val_color};">{val_text}</span>'
        f'</div>'
    )

# ================================================================
# HARDWARE PANEL RENDERER
# ================================================================
def render_hw():
    conv_on  = st.session_state.conveyor_on
    ir_ok    = st.session_state.ir_ok_triggered
    ir_def   = st.session_state.ir_def_triggered
    sv_ok    = st.session_state.servo_ok_active
    sv_def   = st.session_state.servo_def_active

    T = config.THEME

    conv_led = "on-green" if conv_on  else "on-orange blink"
    conv_txt = "RUNNING"  if conv_on  else "STOPPED"
    conv_col = T["green"] if conv_on  else T["orange"]

    ir2_led  = "on-green blink" if ir_ok  else "off"
    ir2_txt  = "TRIGGERED"      if ir_ok  else "WAITING"
    ir2_col  = T["green"]       if ir_ok  else T["t_secondary"]

    ir1_led  = "on-red blink"   if ir_def else "off"
    ir1_txt  = "TRIGGERED"      if ir_def else "WAITING"
    ir1_col  = T["red"]         if ir_def else T["t_secondary"]

    sv2_led  = "on-green blink" if sv_ok  else "off"
    sv2_txt  = "SWEEPING"       if sv_ok  else "IDLE (0 deg)"
    sv2_col  = T["green"]       if sv_ok  else T["t_secondary"]

    sv1_led  = "on-red blink"   if sv_def else "off"
    sv1_txt  = "SWEEPING"       if sv_def else "IDLE (0 deg)"
    sv1_col  = T["red"]         if sv_def else T["t_secondary"]

    hw_ph.markdown(f"""
<div class="hw-card">
  <div class="hw-title">Hardware Status</div>
  {hw_row("Conveyor  (Relay Pin 8)", conv_led, conv_txt, conv_col)}
  {hw_row("IR Sensor OK  (Pin 2)", ir2_led, ir2_txt, ir2_col)}
  {hw_row("IR Sensor DEFECT  (Pin 3)", ir1_led, ir1_txt, ir1_col)}
  {hw_row("Servo OK  (Pin 9)", sv2_led, sv2_txt, sv2_col)}
  {hw_row("Servo DEFECT  (Pin 10)", sv1_led, sv1_txt, sv1_col)}
</div>""", unsafe_allow_html=True)

# ================================================================
# STATS RENDERER
# ================================================================
def render_stats():
    s = st.session_state.stats
    stats_ph.markdown(f"""
<div class="stats-grid">
  <div class="sg-cell sg-total">
    <div class="sg-val">{s['total']}</div>
    <div class="sg-lbl">Total</div>
  </div>
  <div class="sg-cell sg-defect">
    <div class="sg-val">{s['defect']}</div>
    <div class="sg-lbl">Defect</div>
  </div>
  <div class="sg-cell sg-good">
    <div class="sg-val">{s['no_defect']}</div>
    <div class="sg-lbl">Good</div>
  </div>
  <div class="sg-cell sg-idle">
    <div class="sg-val">{s['idle']}</div>
    <div class="sg-lbl">Idle</div>
  </div>
</div>""", unsafe_allow_html=True)

# ================================================================
# TOP BAR
# ================================================================
run_cls = "run"  if st.session_state.running    else "stop"
ard_cls = "ok"   if st.session_state.connected  else "err"
con_cls = "ok"   if st.session_state.conveyor_on else "warn"

st.markdown(f"""
<div class="topbar">
  <div class="tb-brand">
    <div class="tb-logo">&#128269;</div>
    <div>
      <div class="tb-name">LabelInspect</div>
      <div class="tb-sub">
        AI Visual Inspection &nbsp;·&nbsp; YOLO &nbsp;·&nbsp;
        Dual Servo + Dual IR &nbsp;·&nbsp; Conveyor Control
      </div>
    </div>
  </div>
  <div class="tb-right">
    {pill("RUNNING" if st.session_state.running else "STOPPED",
          "pill-green" if st.session_state.running else "pill-muted",
          dot=True, pulse=st.session_state.running)}
    {pill("ARDUINO" if st.session_state.connected else "NO ARDUINO",
          "pill-green" if st.session_state.connected else "pill-red",
          dot=True)}
    {pill("CONVEYOR ON" if st.session_state.conveyor_on else "CONVEYOR OFF",
          "pill-blue" if st.session_state.conveyor_on else "pill-orange",
          dot=True)}
  </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# CONTROLS ROW
# ================================================================
cc1, cc2, cc3, _sp = st.columns([1, 1, 1, 5])
with cc1:
    if st.button("START", use_container_width=True, type="primary",
                 disabled=st.session_state.running):
        st.session_state.running     = True
        st.session_state.conveyor_on = True
        st.rerun()
with cc2:
    if st.button("STOP", use_container_width=True,
                 disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()
with cc3:
    if st.button("RESET", use_container_width=True,
                 disabled=not st.session_state.connected,
                 help="Send 'R' to Arduino: restarts conveyor, clears label state"):
        send_cmd(config.CMD_RESET)
        st.toast("Reset sent to Arduino", icon="&#8635;")
        st.rerun()

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

# ================================================================
# MAIN ROW  —  Feed  |  Status + Hardware + Metrics
# ================================================================
model = load_model(st.session_state.model_path, st.session_state.device)

feed_col, right_col = st.columns([3, 1], gap="large")

with feed_col:
    st.markdown(sh("Live Feed"), unsafe_allow_html=True)
    video_ph = st.empty()

with right_col:
    st.markdown(sh("Detection Status"), unsafe_allow_html=True)
    status_ph = st.empty()

    hw_ph = st.empty()   # hardware status card (no section header, hw-card has its own title)

    st.markdown(sh("Metrics"), unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    with mc1: fps_ph  = st.empty()
    with mc2: det_ph  = st.empty()
    with mc3: cf_ph   = st.empty()

# ================================================================
# HISTORY + STATS ROW
# ================================================================
st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
hist_col, stat_col = st.columns([3, 1], gap="large")

with hist_col:
    st.markdown(sh("Detection History"), unsafe_allow_html=True)
    table_ph = st.empty()
    table_ph.dataframe(
        pd.DataFrame(columns=["Time", "Status", "Confidence", "Objects", "Sent", "Action"]),
        use_container_width=True, height=220
    )

with stat_col:
    st.markdown(sh("Statistics"), unsafe_allow_html=True)
    stats_ph = st.empty()
    render_stats()

# ================================================================
# SETTINGS  —  Tabbed, all inline
# ================================================================
st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
st.markdown(sh("Settings"), unsafe_allow_html=True)

tab_cam, tab_model, tab_thresh, tab_ard, tab_ref = st.tabs([
    "  Camera  ",
    "  Model  ",
    "  Thresholds  ",
    "  Arduino  ",
    "  Hardware Reference  ",
])

# ── Camera ────────────────────────────────────────────────────
with tab_cam:
    tc1, tc2, tc3 = st.columns([1, 1, 2])
    with tc1:
        ci = st.number_input("Camera Index", 0, 10,
                             st.session_state.camera_index,
                             help="0=default webcam, 1=USB camera")
        st.session_state.camera_index = ci
    with tc2:
        fp = st.slider("Target FPS", 1, 30, st.session_state.fps_target)
        st.session_state.fps_target = fp
    with tc3:
        st.markdown("**Resolution**")
        rr1, rr2, rr3 = st.columns(3)
        with rr1:
            if st.button("640x480", use_container_width=True):
                st.session_state.video_width  = 640
                st.session_state.video_height = 480
        with rr2:
            if st.button("960x540", use_container_width=True, type="primary"):
                st.session_state.video_width  = 960
                st.session_state.video_height = 540
        with rr3:
            if st.button("1280x720", use_container_width=True):
                st.session_state.video_width  = 1280
                st.session_state.video_height = 720
        st.markdown(
            logline("Active",
                    f"{st.session_state.video_width}x{st.session_state.video_height}"
                    f"  @  {fp} fps"),
            unsafe_allow_html=True
        )

# ── Model ──────────────────────────────────────────────────────
with tab_model:
    tm1, tm2 = st.columns(2)
    with tm1:
        mp = st.text_input("Model Path", st.session_state.model_path,
                           help="Relative or absolute path to .pt file")
        if mp != st.session_state.model_path:
            st.session_state.model_path = mp
            st.cache_resource.clear()
    with tm2:
        dv = st.selectbox("Device", ["cuda", "cpu"],
                          index=0 if st.session_state.device == "cuda" else 1,
                          help="cuda=GPU (fast), cpu=fallback")
        if dv != st.session_state.device:
            st.session_state.device = dv
            st.cache_resource.clear()
    st.markdown(
        logline("Model", mp) + logline("Device", dv.upper()),
        unsafe_allow_html=True
    )

# ── Thresholds ─────────────────────────────────────────────────
with tab_thresh:
    tt1, tt2, tt3 = st.columns(3)
    with tt1:
        cv = st.slider("Confidence", 0.01, 1.0, st.session_state.conf, 0.01,
                       help="Minimum confidence to count a detection")
        st.session_state.conf = cv
    with tt2:
        iv = st.slider("IOU / NMS", 0.01, 1.0, st.session_state.iou, 0.01,
                       help="Overlap threshold for deduplication")
        st.session_state.iou = iv
    with tt3:
        ctv = st.slider("Confirm Time (s)", 0.5, 5.0,
                        st.session_state.confirm_time, 0.5,
                        help="Detection must be stable this long before command is sent")
        st.session_state.confirm_time = ctv

# ── Arduino ────────────────────────────────────────────────────
with tab_ard:
    _cls = "ok" if st.session_state.connected else "no"
    _txt = "CONNECTED" if st.session_state.connected else "DISCONNECTED"
    _pi  = (f"  ·  {st.session_state.ser.port}"
            if st.session_state.connected and st.session_state.ser else "")
    st.markdown(
        f'<div class="conn-banner {_cls}">'
        f'<span class="conn-dot {"blink" if st.session_state.connected else ""}"></span>'
        f'{_txt}{_pi}</div>',
        unsafe_allow_html=True
    )

    ta1, ta2, ta3, ta4 = st.columns([2, 1, 1, 1])
    with ta1:
        avail = list_ports()
        if avail:
            sp = st.selectbox(
                "Serial Port", avail,
                index=avail.index(st.session_state.serial_port)
                if st.session_state.serial_port in avail else 0
            )
        else:
            sp = st.text_input("Serial Port", st.session_state.serial_port)
            st.caption("No ports detected — enter manually")
        st.session_state.serial_port = sp
    with ta2:
        bd = st.selectbox("Baud Rate", [9600, 115200],
                          index=0 if st.session_state.baud == 9600 else 1)
        st.session_state.baud = bd
    with ta3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Connect", use_container_width=True, type="primary"):
            if connect_arduino(sp, bd):
                st.toast("Arduino connected", icon="&#128279;")
                st.rerun()
            else:
                st.error("Connection failed")
    with ta4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Disconnect", use_container_width=True):
            disconnect_arduino()
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Manual Commands**")
    mc_1, mc_2, mc_3 = st.columns(3)
    _dis = not st.session_state.connected
    with mc_1:
        if st.button("Send 'O'  (OK)", use_container_width=True, disabled=_dis):
            send_cmd(config.CMD_OK)
            st.rerun()
    with mc_2:
        if st.button("Send 'D'  (DEFECT)", use_container_width=True, disabled=_dis):
            send_cmd(config.CMD_DEFECT)
            st.rerun()
    with mc_3:
        if st.button("Send 'R'  (RESET)", use_container_width=True, disabled=_dis):
            send_cmd(config.CMD_RESET)
            st.rerun()

    st.markdown(
        logline("TX", st.session_state.last_tx or "---") +
        logline("RX", st.session_state.last_rx or "---"),
        unsafe_allow_html=True
    )

# ── Hardware Reference ─────────────────────────────────────────
with tab_ref:
    hr1, hr2 = st.columns(2)
    with hr1:
        st.markdown("""
**Arduino Pin Reference**

| Component | Pin |
|---|---|
| IR Sensor OK lane | 2 |
| IR Sensor DEFECT lane | 3 |
| Conveyor Relay | 8 |
| Servo OK | 9 |
| Servo DEFECT | 10 |

**Serial Protocol — single char**

| Char | Python sends when | Arduino action |
|---|---|---|
| `O` | NO_DEFECT confirmed 2s | Arms IR-OK, stops conveyor 4s |
| `D` | DEFECT confirmed 2s | Arms IR-DEFECT, stops conveyor 4s |
| `R` | Manual reset button | Conveyor ON, clears labelType |

**Arduino responses parsed by Python**

| Response | Meaning |
|---|---|
| `READY` | Boot complete, conveyor on |
| `ACK:OK` | 'O' received |
| `ACK:DEFECT` | 'D' received |
| `ACK:RESET` | 'R' received |
| `IR:OK` | IR-OK sensor triggered, servoOK sweeping |
| `IR:DEFECT` | IR-DEFECT sensor triggered, servoDEFECT sweeping |
        """)
    with hr2:
        st.markdown("""
**System Flow**

```
Camera detects item on belt
         |
   Stable for 2s?
   /           \\
 YES            NO
  |              |
Send 'O'      Send 'D'     (stay IDLE)
  |              |
Arduino stops conveyor (4s)
  |              |
Item reaches   Item reaches
IR-OK sensor   IR-DEFECT sensor
  |              |
servoOK        servoDEFECT
sweeps 0->180  sweeps 0->180
then returns   then returns
  |              |
Conveyor resumes
```

**Troubleshooting**

- Close Arduino IDE Serial Monitor before connecting here
- Linux: `ls /dev/ttyACM*`
- Permissions: `sudo usermod -a -G dialout $USER`
- IR sensors must output LOW on detection (standard NPN type)
- If conveyor doesn't stop: check relay on pin 8, check relay logic (HIGH=ON in sketch)
- Test manually: use the Manual Commands buttons above
        """)

# ================================================================
# SYSTEM STATUS BAR
# ================================================================
run_cls2 = "run"  if st.session_state.running    else "stop"
ard_cls2 = "ok"   if st.session_state.connected  else "err"
con_cls2 = "ok"   if st.session_state.conveyor_on else "warn"

st.markdown(f"""
<div class="sysbar">
  <div class="sb-item {run_cls2}">
    <span class="sb-dot"></span>
    <span class="sb-text">{"RUNNING" if st.session_state.running else "STOPPED"}</span>
  </div>
  <span class="sb-sep">|</span>
  <div class="sb-item {ard_cls2}">
    <span class="sb-dot"></span>
    <span class="sb-text">{"ARDUINO" if st.session_state.connected else "NO ARDUINO"}</span>
  </div>
  <span class="sb-sep">|</span>
  <div class="sb-item {con_cls2}">
    <span class="sb-dot"></span>
    <span class="sb-text">{"CONVEYOR ON" if st.session_state.conveyor_on else "CONVEYOR OFF"}</span>
  </div>
  <span class="sb-sep">|</span>
  <div class="sb-item">
    <span class="sb-text">
      {st.session_state.video_width}x{st.session_state.video_height}
      @ {st.session_state.fps_target}fps
    </span>
  </div>
  <span class="sb-sep">|</span>
  <div class="sb-item">
    <span class="sb-text">{st.session_state.model_path}  [{st.session_state.device.upper()}]</span>
  </div>
  <span class="sb-sep">|</span>
  <div class="sb-item">
    <span class="sb-text">
      conf {st.session_state.conf:.2f}
      &nbsp; iou {st.session_state.iou:.2f}
      &nbsp; confirm {st.session_state.confirm_time:.1f}s
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# IDLE STATE PLACEHOLDERS
# ================================================================
if not st.session_state.running:
    W = st.session_state.video_width
    H = min(st.session_state.video_height, 480)

    video_ph.markdown(f"""
<div class="cam-idle" style="width:{W}px; height:{H}px; max-width:100%;">
  <div class="ci-icon">&#128247;</div>
  <div class="ci-title">Detection Paused</div>
  <div class="ci-sub">
    cam {st.session_state.camera_index} &nbsp;·&nbsp;
    {W}x{st.session_state.video_height} &nbsp;·&nbsp;
    {st.session_state.fps_target} fps
  </div>
</div>""", unsafe_allow_html=True)

    status_ph.markdown("""
<div class="status-card sc-idle">
  <div class="sc-eyebrow">System Output</div>
  <div style="margin-bottom:8px;">
    <span class="pill pill-muted">
      <span class="dot"></span>IDLE
    </span>
  </div>
  <div class="sc-state" style="color:var(--t-mute);">IDLE</div>
  <div class="sc-sub">Press START to begin inspection</div>
</div>""", unsafe_allow_html=True)

    render_hw()
    fps_ph.metric("FPS",   "--")
    det_ph.metric("Det.",  "--")
    cf_ph.metric("Conf.", "--")

# ================================================================
# MAIN DETECTION LOOP
# ================================================================
if st.session_state.running and model:
    cap = cv2.VideoCapture(st.session_state.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  st.session_state.video_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, st.session_state.video_height)

    if not cap.isOpened():
        st.error("Failed to open camera — check Camera Index in settings.")
        st.session_state.running = False
    else:
        while st.session_state.running:
            t0 = time.time()

            # Poll Arduino for hardware state updates
            poll_serial()

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                st.warning("Frame capture failed — camera may have disconnected.")
                break

            frame = cv2.resize(
                frame,
                (st.session_state.video_width, st.session_state.video_height)
            )

            # Run inference
            results = model.predict(
                frame,
                conf=st.session_state.conf,
                iou=st.session_state.iou,
                device=st.session_state.device,
                verbose=False,
            )
            result = results[0]

            # Parse detections
            dets = []
            if result.boxes:
                for box in result.boxes:
                    dets.append({
                        "class_id":   int(box.cls[0]),
                        "confidence": float(box.conf[0]),
                    })

            # Run confirmation state machine
            confirmed, is_new, info, prog = check_confirm(dets)

            # Draw annotated frame
            ann     = result.plot(boxes=True, labels=True, conf=True, line_width=2)
            ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            video_ph.image(
                ann_rgb, channels="RGB",
                use_container_width=False,
                width=st.session_state.video_width
            )

            # ── Status card ────────────────────────────────
            avg_c   = np.mean([d["confidence"] for d in dets]) if dets else 0.0
            bar_pct = prog * 100
            c_state = (
                "Confirming..." if 0 < prog < 1.0 and dets
                else "Stable" if confirmed != "IDLE"
                else "Idle"
            )
            sent_note = ""
            if st.session_state.last_sent_cmd:
                sent_note = (
                    f'<div style="margin-top:8px;font-family:var(--mono);'
                    f'font-size:10px;color:var(--t-sec);">'
                    f'Last sent: &nbsp;<code style="color:var(--blue);">'
                    f"'{st.session_state.last_sent_cmd}'"
                    f'</code></div>'
                )

            status_ph.markdown(f"""
<div class="status-card {info['cls']}">
  <div class="sc-eyebrow">System Output</div>
  <div style="margin-bottom:8px;">
    <span class="pill {info['badge']}">
      <span class="dot"></span>{info['label']}
    </span>
  </div>
  <div class="sc-state" style="color:{info['color']};">{confirmed}</div>
  <div class="sc-sub">conf {avg_c:.2f} &nbsp;·&nbsp; {len(dets)} obj</div>
  <div class="cbar-wrap">
    <div class="cbar-track">
      <div class="cbar-fill"
           style="width:{bar_pct:.0f}%; background:{info['color']};"></div>
    </div>
    <div class="cbar-label">
      <span>{c_state}</span><span>{bar_pct:.0f}%</span>
    </div>
  </div>
  {sent_note}
</div>""", unsafe_allow_html=True)

            # ── Hardware panel ─────────────────────────────
            render_hw()

            # ── Metrics ────────────────────────────────────
            fps = 1.0 / max(time.time() - t0, 1e-9)
            fps_ph.metric("FPS",   f"{fps:.1f}")
            det_ph.metric("Det.",  len(dets))
            cf_ph.metric("Conf.", f"{avg_c:.2f}")

            # ── History table ──────────────────────────────
            if is_new or st.session_state.frame_count % 5 == 0:
                action = (
                    f"Sent '{info['cmd']}'" if is_new and info["cmd"]
                    else confirmed
                )
                entry = {
                    "Time":       datetime.now().strftime("%H:%M:%S"),
                    "Status":     confirmed,
                    "Confidence": f"{avg_c:.2f}",
                    "Objects":    len(dets),
                    "Sent":       info["cmd"] or "---",
                    "Action":     action,
                }
                st.session_state.output_history.append(entry)
                df = pd.DataFrame(list(st.session_state.output_history))
                table_ph.dataframe(df.tail(15), use_container_width=True, height=220)

                if is_new:
                    key = {
                        "DEFECT":    "defect",
                        "NO_DEFECT": "no_defect",
                        "IDLE":      "idle",
                    }.get(confirmed, "idle")
                    st.session_state.stats[key]    += 1
                    st.session_state.stats["total"] += 1

            render_stats()
            st.session_state.frame_count += 1

            # FPS cap
            wait = (1.0 / st.session_state.fps_target) - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)

        cap.release()

# ================================================================
# FOOTER
# ================================================================
st.markdown(f"""
<div class="footer">
  LabelInspect &nbsp;·&nbsp;
  {st.session_state.model_path} &nbsp;·&nbsp;
  {st.session_state.device.upper()} &nbsp;·&nbsp;
  {st.session_state.video_width}x{st.session_state.video_height}
  @ {st.session_state.fps_target}fps &nbsp;·&nbsp;
  {st.session_state.serial_port} @ {st.session_state.baud} &nbsp;·&nbsp;
  Protocol: <code style="font-size:9px;">'O' / 'D' / 'R'</code>
</div>
""", unsafe_allow_html=True)