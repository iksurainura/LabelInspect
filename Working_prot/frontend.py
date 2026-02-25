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
# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
   page_title="Defect Detection System",
   page_icon="🔍",
   layout="wide",
   initial_sidebar_state="expanded"
)
# ==========================================
# CUSTOM CSS - CLEAN PROFESSIONAL UI
# ==========================================
st.markdown("""
<style>
   /* Main Theme */
   .stApp { background-color: #0d1117; color: #c9d1d9; }
   /* Typography */
   h1, h2, h3, h4, h5, h6 {
       color: #c9d1d9 !important;
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif !important;
       font-weight: 600 !important;
   }
   /* Cards */
   .css-1r6slb0, .css-1y4p8pa {
       background-color: #161b22;
       border: 1px solid #30363d;
       border-radius: 8px;
   }
   /* Buttons */
   .stButton > button {
       border-radius: 6px;
       font-weight: 500;
       transition: all 0.2s;
   }
   .stButton > button:hover { transform: translateY(-1px); }
   /* Primary Button */
   .stButton > button[kind="primary"] {
       background-color: #238636;
       color: white;
       border: 1px solid rgba(240,246,252,0.1);
   }
   .stButton > button[kind="primary"]:hover { background-color: #2ea043; }
   /* Secondary Button */
   .stButton > button[kind="secondary"] {
       background-color: #21262d;
       color: #c9d1d9;
       border: 1px solid #30363d;
   }
   /* Servo Control Buttons */
   .servo-btn {
       height: 70px !important;
       font-size: 16px !important;
       font-weight: 700 !important;
   }
   .servo-l { background-color: #ff6b6b !important; color: white !important; }
   .servo-c { background-color: #4ecdc4 !important; color: white !important; }
   .servo-r { background-color: #45b7d1 !important; color: white !important; }
   /* Status Badges */
   .badge {
       display: inline-block;
       padding: 4px 12px;
       border-radius: 12px;
       font-size: 12px;
       font-weight: 600;
       text-transform: uppercase;
   }
   .badge-defect { background: rgba(218,54,51,0.2); color: #da3633; border: 1px solid rgba(218,54,51,0.3); }
   .badge-ok { background: rgba(35,134,54,0.2); color: #238636; border: 1px solid rgba(35,134,54,0.3); }
   .badge-idle { background: rgba(110,118,129,0.2); color: #6e7681; border: 1px solid rgba(110,118,129,0.3); }
   /* Connection Status */
   .conn-connected {
       background: rgba(35,134,54,0.1);
       border: 1px solid #238636;
       color: #238636;
       padding: 10px;
       border-radius: 6px;
       font-family: monospace;
   }
   .conn-disconnected {
       background: rgba(218,54,51,0.1);
       border: 1px solid #da3633;
       color: #da3633;
       padding: 10px;
       border-radius: 6px;
       font-family: monospace;
   }
   /* Input Fields */
   .stTextInput > div > div > input, .stNumberInput > div > div > input {
       background-color: #21262d;
       color: #c9d1d9;
       border: 1px solid #30363d;
       border-radius: 6px;
   }
   /* Sliders */
   .stSlider > div > div > div { background-color: #238636; }
   /* Expander */
   .streamlit-expanderHeader {
       background-color: #161b22;
       border: 1px solid #30363d;
       border-radius: 6px;
   }
   /* Code Blocks */
   .stCodeBlock { background-color: #161b22 !important; }
   /* Servo Gauge */
   .servo-track {
       background: #21262d;
       height: 24px;
       border-radius: 12px;
       overflow: hidden;
       border: 1px solid #30363d;
   }
   .servo-fill {
       height: 100%;
       transition: width 0.3s ease, background-color 0.3s;
       display: flex;
       align-items: center;
       justify-content: center;
       font-size: 12px;
       font-weight: bold;
       color: white;
   }
   /* Debug Panel */
   .debug-panel {
       background: #161b22;
       border: 1px solid #30363d;
       border-radius: 8px;
       padding: 12px;
       margin: 8px 0;
   }
</style>
""", unsafe_allow_html=True)
# ==========================================
# SESSION STATE
# ==========================================
def init_session_state():
   defaults = {
       "running": False,
       "conf": config.DEFAULT_CONF_THRESHOLD,
       "iou": config.DEFAULT_IOU_THRESHOLD,
       "confirm_time": config.CONFIRMATION_TIME,
       "camera_index": config.CAMERA_INDEX,
       "fps_target": config.FPS_TARGET,
       "video_width": config.VIDEO_WIDTH,
       "video_height": config.VIDEO_HEIGHT,
       "confirmed_output": "IDLE",
       "pending_output": None,
       "pending_since": None,
       "output_history": deque(maxlen=100),
       "ser": None,
       "connected": False,
       "last_cmd": None,
       "last_response": None,
       "servo_pos": 90,
       "frame_count": 0,
       "detection_stats": {"defect": 0, "no_defect": 0, "idle": 0, "total": 0},
   }
   for key, value in defaults.items():
       if key not in st.session_state:
           st.session_state[key] = value
init_session_state()
# ==========================================
# ARDUINO FUNCTIONS
# ==========================================
def connect_arduino(port, baud):
   try:
       if st.session_state.ser and st.session_state.ser.is_open:
           st.session_state.ser.close()
       ser = serial.Serial(port, baud, timeout=1)
       time.sleep(2.5)
       st.session_state.ser = ser
       st.session_state.connected = True
       st.session_state.last_response = f"Connected to {port} @ {baud}"
       return True
   except Exception as e:
       st.session_state.last_response = f"Error: {str(e)}"
       return False
def disconnect_arduino():
   if st.session_state.ser and st.session_state.ser.is_open:
       st.session_state.ser.close()
   st.session_state.ser = None
   st.session_state.connected = False
   st.session_state.last_response = "Disconnected"
def send_servo_cmd(cmd):
   if not st.session_state.ser or not st.session_state.ser.is_open:
       st.session_state.last_response = "Not connected"
       return False
   try:
       st.session_state.ser.write(cmd.encode())
       st.session_state.ser.flush()
       st.session_state.last_cmd = f"{datetime.now().strftime('%H:%M:%S')} | {cmd}"
       # Update position tracking
       if cmd == 'L': st.session_state.servo_pos = 0
       elif cmd == 'C': st.session_state.servo_pos = 90
       elif cmd == 'R': st.session_state.servo_pos = 180
       # Read response
       time.sleep(0.1)
       if st.session_state.ser.in_waiting:
           resp = st.session_state.ser.readline().decode('utf-8', errors='ignore').strip()
           if resp: st.session_state.last_response = resp
       return True
   except Exception as e:
       st.session_state.last_response = f"Send failed: {e}"
       return False
def list_serial_ports():
   return [p.device for p in serial.tools.list_ports.comports()]
# ==========================================
# DETECTION LOGIC
# ==========================================
def check_confirmation(detections, conf_threshold, iou_threshold, confirm_time):
   now = time.time()
   if not detections:
       current = "IDLE"
       info = {"label": "NO DETECTION", "badge": "badge-idle", "cmd": "C", "angle": 90, "color": "#6e7681"}
   else:
       has_defect = any(d["class_id"] in config.DEFECT_CLASSES for d in detections)
       if has_defect:
           current = "DEFECT"
           info = {"label": "DEFECT DETECTED", "badge": "badge-defect", "cmd": "L", "angle": 0, "color": "#da3633"}
       else:
           current = "NO_DEFECT"
           info = {"label": "NO DEFECT", "badge": "badge-ok", "cmd": "C", "angle": 90, "color": "#238636"}
   # Stability check
   if st.session_state.pending_output == current:
       if st.session_state.pending_since and (now - st.session_state.pending_since) >= confirm_time:
           if st.session_state.confirmed_output != current:
               st.session_state.confirmed_output = current
               send_servo_cmd(info["cmd"])
               return current, True, info
           return current, False, info
   else:
       st.session_state.pending_output = current
       st.session_state.pending_since = now
   return st.session_state.confirmed_output, False, info
# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_model(model_path, device):
   try:
       model = YOLO(model_path)
       model.to(device)
       return model
   except Exception as e:
       st.error(f"Model load failed: {e}")
       return None
# ==========================================
# MAIN UI
# ==========================================
st.markdown("""
<div style="border-bottom: 1px solid #30363d; padding-bottom: 16px; margin-bottom: 24px;">
<h1 style="margin: 0; font-size: 28px;">🔍 Real-Time Defect Detection System</h1>
<p style="color: #8b949e; margin: 8px 0 0 0; font-size: 13px;">
       AI-Powered Visual Inspection with Arduino Servo Control
</p>
</div>
""", unsafe_allow_html=True)
# ==========================================
# SIDEBAR - ALL INPUT CONTROLS
# ==========================================
with st.sidebar:
   st.markdown("## ⚙️ System Configuration")
   # ==========================================
   # 1. CAMERA SETTINGS
   # ==========================================
   with st.expander("📷 Camera Settings", expanded=True):
       cam_col1, cam_col2 = st.columns(2)
       with cam_col1:
           camera_index = st.number_input(
               "Camera Index",
               min_value=0,
               max_value=10,
               value=st.session_state.camera_index,
               help="0 for default webcam, 1 for external USB camera"
           )
       with cam_col2:
           fps_target = st.slider(
               "Target FPS",
               min_value=1,
               max_value=30,
               value=st.session_state.fps_target,
               help="Higher FPS = smoother video but more CPU/GPU usage"
           )
       # Video Resolution Selection
       st.markdown("**Video Resolution:**")
       res_col1, res_col2 = st.columns(2)
       with res_col1:
           if st.button("520x390 (Small)", use_container_width=True):
               st.session_state.video_width = 520
               st.session_state.video_height = 390
       with res_col2:
           if st.button("1040x780 (Large)", use_container_width=True, type="primary"):
               st.session_state.video_width = 1040
               st.session_state.video_height = 780
       st.info(f"Current: **{st.session_state.video_width} × {st.session_state.video_height}**")
       st.session_state.camera_index = camera_index
       st.session_state.fps_target = fps_target
   # ==========================================
   # 2. MODEL SETTINGS
   # ==========================================
   with st.expander("🧠 Model Settings", expanded=True):
       model_path = st.text_input(
           "Model Path",
           value=config.MODEL_PATH,
           help="Path to YOLO model file (.pt)"
       )
       device = st.selectbox(
           "Inference Device",
           options=["cuda", "cpu"],
           index=0 if config.DEVICE == "cuda" else 1,
           help="cuda = GPU (faster), cpu = CPU (slower)"
       )
       st.markdown(f"<small>Model: `{model_path}` | Device: `{device}`</small>", unsafe_allow_html=True)
   # ==========================================
   # 3. DETECTION THRESHOLDS
   # ==========================================
   with st.expander("🎯 Detection Thresholds", expanded=True):
       conf_threshold = st.slider(
           "Confidence Threshold",
           min_value=0.01,
           max_value=1.0,
           value=st.session_state.conf,
           step=0.01,
           help="Minimum confidence to consider a detection valid"
       )
       iou_threshold = st.slider(
           "IOU Threshold (NMS)",
           min_value=0.01,
           max_value=1.0,
           value=st.session_state.iou,
           step=0.01,
           help="Overlap threshold for removing duplicate detections"
       )
       confirm_time = st.slider(
           "Confirmation Time (seconds)",
           min_value=0.5,
           max_value=5.0,
           value=st.session_state.confirm_time,
           step=0.5,
           help="How long detection must be stable before servo moves"
       )
       st.session_state.conf = conf_threshold
       st.session_state.iou = iou_threshold
       st.session_state.confirm_time = confirm_time
   # ==========================================
   # 4. DISPLAY OPTIONS
   # ==========================================
   with st.expander("🖼️ Display Options"):
       show_boxes = st.checkbox("Show Bounding Boxes", value=config.SHOW_BOUNDING_BOXES)
       show_labels = st.checkbox("Show Class Labels", value=config.SHOW_LABELS)
       show_conf = st.checkbox("Show Confidence Scores", value=config.SHOW_CONFIDENCE)
   # ==========================================
   # 5. ARDUINO / SERVO CONTROL
   # ==========================================
   st.markdown("## 🔌 Arduino Servo Control")
   with st.expander("Connection", expanded=True):
       # Connection Status
       if st.session_state.connected:
           st.markdown(f'<div class="conn-connected">● CONNECTED<br><small>{st.session_state.ser.port if st.session_state.ser else "Unknown"} @ {config.SERIAL_BAUDRATE} baud</small></div>', unsafe_allow_html=True)
       else:
           st.markdown(f'<div class="conn-disconnected">● DISCONNECTED<br><small>Click Connect to establish link</small></div>', unsafe_allow_html=True)
       # Port Selection
       available_ports = list_serial_ports()
       if available_ports:
           selected_port = st.selectbox("Serial Port", available_ports, index=available_ports.index(config.SERIAL_PORT) if config.SERIAL_PORT in available_ports else 0)
       else:
           selected_port = st.text_input("Serial Port", value=config.SERIAL_PORT)
           st.warning("No ports auto-detected")
       baud_rate = st.selectbox("Baud Rate", [9600, 115200], index=0)
       # Connect/Disconnect Buttons
       conn_col1, conn_col2 = st.columns(2)
       with conn_col1:
           if st.button("🔗 Connect", use_container_width=True, type="primary"):
               if connect_arduino(selected_port, baud_rate):
                   st.success("Connected!")
                   st.rerun()
               else:
                   st.error("Connection failed")
       with conn_col2:
           if st.button("❌ Disconnect", use_container_width=True):
               disconnect_arduino()
               st.rerun()
   # ==========================================
   # 6. SERVO DEBUGGING PANEL
   # ==========================================
   with st.expander("🎛️ Servo Debugging", expanded=True):
       # Current Position Gauge
       pos = st.session_state.servo_pos
       pos_percent = (pos / 180) * 100
       pos_color = "#ff6b6b" if pos == 0 else "#4ecdc4" if pos == 90 else "#45b7d1"
       pos_label = "LEFT (0°)" if pos == 0 else "CENTER (90°)" if pos == 90 else "RIGHT (180°)"
       st.markdown(f"""
<div style="margin-bottom: 8px;">
<span style="color: #8b949e; font-size: 11px;">CURRENT POSITION</span>
<div class="servo-track">
<div class="servo-fill" style="width: {pos_percent}%; background: {pos_color};">
                   {pos}°
</div>
</div>
<div style="text-align: center; color: {pos_color}; font-size: 12px; margin-top: 4px;">
               {pos_label}
</div>
</div>
       """, unsafe_allow_html=True)
       # Manual Control Buttons
       st.markdown("**Manual Control:**")
       mc1, mc2, mc3 = st.columns(3)
       with mc1:
           if st.button("⬅️ L\n0°", key="man_l", use_container_width=True, disabled=not st.session_state.connected):
               if send_servo_cmd('L'):
                   st.toast("Servo moved to 0° (Left/Defect)", icon="⬅️")
                   st.rerun()
       with mc2:
           if st.button("⬆️ C\n90°", key="man_c", use_container_width=True, disabled=not st.session_state.connected):
               if send_servo_cmd('C'):
                   st.toast("Servo moved to 90° (Center)", icon="⬆️")
                   st.rerun()
       with mc3:
           if st.button("➡️ R\n180°", key="man_r", use_container_width=True, disabled=not st.session_state.connected):
               if send_servo_cmd('R'):
                   st.toast("Servo moved to 180° (Right)", icon="➡️")
                   st.rerun()
       # Command Log
       st.markdown("**Command Log:**")
       st.code(f"TX: {st.session_state.last_cmd or 'None'}", language=None)
       st.code(f"RX: {st.session_state.last_response or 'None'}", language=None)
       # Quick Test Sequence
       st.markdown("**Auto Test Sequence:**")
       if st.button("▶️ Run L-C-R Test", use_container_width=True, disabled=not st.session_state.connected):
           with st.spinner("Testing servo..."):
               send_servo_cmd('L')
               time.sleep(0.5)
               send_servo_cmd('C')
               time.sleep(0.5)
               send_servo_cmd('R')
               time.sleep(0.5)
               send_servo_cmd('C')
               st.success("Test complete!")
               st.rerun()
   # ==========================================
   # 7. TROUBLESHOOTING GUIDE
   # ==========================================
   with st.expander("🔧 Troubleshooting"):
       st.markdown("""
       **Servo Not Moving?**
       1. **Check Wiring:**
          - Servo Orange/Yellow → Pin 9
          - Servo Red → 5V
          - Servo Brown/Black → GND
       2. **Check Power:**
          - USB power may be insufficient for large servos
          - Use external 5V power supply if needed
       3. **Check Serial:**
          - Close Arduino IDE Serial Monitor
          - Verify port: `ls /dev/ttyACM*`
          - Check permissions: `sudo usermod -a -G dialout $USER`
       4. **Test Standalone:**
          ```bash
          python -c "import serial; s=serial.Serial('/dev/ttyACM0', 9600); s.write(b'L'); s.close()"
          ```
       5. **Arduino Code:**
          - Must use single char commands (L, C, R)
          - Baud rate must match (9600)
       **Camera Not Found?**
       - Try index 0, 1, 2 in Camera Settings
       - Check `ls /dev/video*`
       **Model Not Loading?**
       - Verify `best.pt` exists in working directory
       - Check CUDA is available: `nvidia-smi`
       """)
# Load model
model = load_model(model_path, device)
# ==========================================
# MAIN DISPLAY AREA
# ==========================================
# Top Row: Video + Status
video_col, status_col = st.columns([2, 1])
with video_col:
   st.markdown(f"### 📹 Live Video Stream")
   video_placeholder = st.empty()
with status_col:
   st.markdown("### 🎯 Detection Status")
   # Main Status Card
   status_card = st.empty()
   # Servo Status Sub-card
   servo_status = st.empty()
   # Metrics
   m1, m2, m3 = st.columns(3)
   with m1:
       fps_metric = st.empty()
   with m2:
       det_metric = st.empty()
   with m3:
       conf_metric = st.empty()
# Middle: Detection History Table
st.markdown("---")
st.markdown("### 📊 Detection History")
tbl_col, stat_col = st.columns([3, 1])
with tbl_col:
   table_placeholder = st.dataframe(
       pd.DataFrame(columns=["Time", "Status", "Confidence", "Objects", "Servo", "Angle"]),
       use_container_width=True,
       height=250
   )
with stat_col:
   stats_placeholder = st.empty()
# Bottom: Main Controls
st.markdown("---")
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
with ctrl_col1:
   if st.button("▶️ START DETECTION", use_container_width=True, type="primary", disabled=st.session_state.running):
       st.session_state.running = True
       st.rerun()
with ctrl_col2:
   if st.button("⏹️ STOP DETECTION", use_container_width=True, disabled=not st.session_state.running):
       st.session_state.running = False
       st.rerun()
with ctrl_col3:
   # System Status Bar
   sys_status = "🟢 Running" if st.session_state.running else "⚪ Stopped"
   ard_status = "🟢 Arduino" if st.session_state.connected else "🔴 Arduino"
   st.info(f"{sys_status} | {ard_status} | Res: {st.session_state.video_width}×{st.session_state.video_height}")
# ==========================================
# MAIN PROCESSING LOOP
# ==========================================
if st.session_state.running and model:
   cap = cv2.VideoCapture(st.session_state.camera_index)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, st.session_state.video_width)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, st.session_state.video_height)
   if not cap.isOpened():
       st.error("❌ Failed to open camera")
       st.session_state.running = False
   else:
       while st.session_state.running:
           loop_start = time.time()
           ret, frame = cap.read()
           if not ret:
               st.warning("⚠️ Frame read failed")
               break
           # Resize to configured resolution
           frame = cv2.resize(frame, (st.session_state.video_width, st.session_state.video_height))
           # Inference
           results = model.predict(
               frame,
               conf=st.session_state.conf,
               iou=st.session_state.iou,
               device=device,
               verbose=False
           )
           result = results[0]
           # Parse detections
           detections = []
           if result.boxes:
               for box in result.boxes:
                   detections.append({
                       "class_id": int(box.cls[0]),
                       "confidence": float(box.conf[0])
                   })
           # Check confirmation and send to servo
           confirmed, is_new, info = check_confirmation(
               detections,
               st.session_state.conf,
               st.session_state.iou,
               st.session_state.confirm_time
           )
           # Draw video
           annotated = result.plot(boxes=show_boxes, labels=show_labels, conf=show_conf, line_width=2)
           annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
           video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=False, width=st.session_state.video_width)
           # Update Status Card
           status_card.markdown(f"""
<div style="background: #161b22; border: 2px solid {info['color']}; border-radius: 10px; padding: 20px; text-align: center;">
<div class="{info['badge']}" style="margin-bottom: 12px;">{info['label']}</div>
<div style="color: {info['color']}; font-size: 32px; font-weight: 700; margin: 8px 0;">
                   {confirmed}
</div>
<div style="color: #8b949e; font-size: 12px; margin-top: 8px;">
                   Confidence: {np.mean([d['confidence'] for d in detections]):.2f}" if detections else "0.00"
</div>
</div>
           """, unsafe_allow_html=True)
           # Update Servo Status
           servo_pos = st.session_state.servo_pos
           servo_color = "#ff6b6b" if servo_pos == 0 else "#4ecdc4" if servo_pos == 90 else "#45b7d1"
           servo_status.markdown(f"""
<div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-top: 12px;">
<div style="font-size: 11px; color: #8b949e; margin-bottom: 4px;">SERVO POSITION</div>
<div style="display: flex; align-items: center; gap: 8px;">
<div style="background: {servo_color}; width: 12px; height: 12px; border-radius: 50%;"></div>
<span style="color: {servo_color}; font-size: 20px; font-weight: bold;">{servo_pos}°</span>
<span style="color: #6e7681; font-size: 11px;">
                       ({ "Defect/Left" if servo_pos == 0 else "Good/Center" if servo_pos == 90 else "Right" })
</span>
</div>
<div style="font-size: 10px; color: #6e7681; margin-top: 4px;">
                   Last: {st.session_state.last_cmd or 'None'}
</div>
</div>
           """, unsafe_allow_html=True)
           # Metrics
           fps = 1.0 / (time.time() - loop_start) if loop_start else 0
           avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0.0
           fps_metric.metric("FPS", f"{fps:.1f}")
           det_metric.metric("Detections", len(detections))
           conf_metric.metric("Avg Conf", f"{avg_conf:.2f}")
           # Update History
           if is_new or st.session_state.frame_count % 5 == 0:
               entry = {
                   "Time": datetime.now().strftime("%H:%M:%S"),
                   "Status": confirmed,
                   "Confidence": f"{avg_conf:.2f}",
                   "Objects": len(detections),
                   "Servo": info["cmd"],
                   "Angle": info["angle"]
               }
               st.session_state.output_history.append(entry)
               df = pd.DataFrame(list(st.session_state.output_history))
               table_placeholder.dataframe(df.tail(15), use_container_width=True)
               if is_new:
                   key = {"DEFECT": "defect", "NO_DEFECT": "no_defect", "IDLE": "idle"}.get(confirmed, "idle")
                   st.session_state.detection_stats[key] += 1
                   st.session_state.detection_stats["total"] += 1
           # Statistics
           total = st.session_state.detection_stats["total"]
           if total > 0:
               stats_placeholder.markdown(f"""
<div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px;">
<div style="margin-bottom: 8px; font-size: 11px; color: #8b949e;">DETECTION STATISTICS</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
<span style="color: #c9d1d9;">Total:</span>
<span style="color: #58a6ff; font-weight: bold;">{total}</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
<span style="color: #da3633;">Defects:</span>
<span style="color: #da3633; font-weight: bold;">{st.session_state.detection_stats["defect"]}</span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
<span style="color: #238636;">Good:</span>
<span style="color: #238636; font-weight: bold;">{st.session_state.detection_stats["no_defect"]}</span>
</div>
<div style="display: flex; justify-content: space-between;">
<span style="color: #6e7681;">Idle:</span>
<span style="color: #6e7681; font-weight: bold;">{st.session_state.detection_stats["idle"]}</span>
</div>
</div>
               """, unsafe_allow_html=True)
           st.session_state.frame_count += 1
           # FPS limiting
           elapsed = time.time() - loop_start
           sleep_time = (1.0 / st.session_state.fps_target) - elapsed
           if sleep_time > 0:
               time.sleep(sleep_time)
       cap.release()
elif not st.session_state.running:
   # Stopped state
   video_placeholder.markdown(f"""
<div style="width: {st.session_state.video_width}px; height: {st.session_state.video_height}px;
               background: #161b22; border: 2px dashed #30363d; border-radius: 10px;
               display: flex; align-items: center; justify-content: center; color: #8b949e;
               max-width: 100%;">
<div style="text-align: center;">
<div style="font-size: 48px; margin-bottom: 16px;">📷</div>
<div style="font-size: 16px; margin-bottom: 8px;">Press <b>▶️ START DETECTION</b> to begin</div>
<div style="font-size: 12px; color: #6e7681;">
               Camera: {st.session_state.camera_index} |
               Resolution: {st.session_state.video_width}×{st.session_state.video_height}
</div>
</div>
</div>
   """, unsafe_allow_html=True)
   status_card.info("System Ready - Configure settings in sidebar and click Start")
# Footer
st.markdown("---")
st.caption(f"""
System: {config.MODEL_PATH} | Device: {device} |
Video: {st.session_state.video_width}×{st.session_state.video_height} @ {st.session_state.fps_target}fps |
Arduino: {config.SERIAL_PORT} @ {config.SERIAL_BAUDRATE} baud |
Protocol: L/C/R (0°/90°/180°)
""")