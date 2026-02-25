"""
Configuration file for Real-Time Defect Detection System
All tunable parameters centralized here
"""
# ==========================================
# MODEL CONFIGURATION
# ==========================================
MODEL_PATH = "yolo11m-seg.pt"          # Path to YOLO model
DEVICE = "cuda"                 # Use "cuda" for GPU or "cpu" for CPU
IMG_SIZE = 640                  # Input image size for model
# ==========================================
# DETECTION THRESHOLDS
# ==========================================
DEFAULT_CONF_THRESHOLD = 0.25   # Minimum confidence for detection (0.0 - 1.0)
DEFAULT_IOU_THRESHOLD = 0.45    # IOU threshold for NMS (0.0 - 1.0)
CONFIRMATION_TIME = 2.0         # Seconds to confirm stable output (0.5 - 5.0)
FPS_TARGET = 15                 # Target frames per second (1 - 30)
# ==========================================
# VIDEO STREAM CONFIGURATION
# ==========================================
VIDEO_WIDTH = 1040              # Video width in pixels (520 or 1040)
VIDEO_HEIGHT = 780              # Video height in pixels (390 or 780)
CAMERA_INDEX = 1               # Camera device index (0, 1, 2, etc.)
# ==========================================
# DISPLAY OPTIONS
# ==========================================
SHOW_BOUNDING_BOXES = True      # Show detection boxes
SHOW_LABELS = True              # Show class labels
SHOW_CONFIDENCE = True          # Show confidence scores
# ==========================================
# SERIAL COMMUNICATION (ARDUINO)
# ==========================================
SERIAL_PORT = "/dev/ttyACM0"    # Serial port (Windows: COM3, Linux/Mac: /dev/ttyACM0)
SERIAL_BAUDRATE = 9600          # Baud rate (must match Arduino sketch)
SERIAL_TIMEOUT = 1              # Serial timeout in seconds
# Servo control commands (Single character protocol)
SERVO_DEFECT_CMD = "L"          # L = 0° (Defect position)
SERVO_NO_DEFECT_CMD = "C"       # C = 90° (Good/No Defect position)
SERVO_IDLE_CMD = "I"            # C = 90° (Idle position)
# ==========================================
# CLASS CONFIGURATION
# ==========================================
CLASS_NAMES = {
   0: "Defect",
   1: "No Defect"
}
DEFECT_CLASSES = [0]            # Class indices that trigger defect servo position
# ==========================================
# UI THEME (GitHub Dark)
# ==========================================
THEME = {
   "bg_primary": "#0d1117",
   "bg_secondary": "#161b22",
   "bg_tertiary": "#21262d",
   "border": "#30363d",
   "text_primary": "#c9d1d9",
   "text_secondary": "#8b949e",
   "accent": "#58a6ff",
   "success": "#238636",
   "danger": "#da3633",
   "warning": "#d29922",
   "info": "#1f6feb",
   "idle": "#6e7681",
}