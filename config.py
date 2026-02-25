"""LabelInspect — Configuration"""

# Model
MODEL_PATH  = "yolo11m-seg.pt"
DEVICE      = "cuda"
IMG_SIZE    = 640

# Detection
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD  = 0.45
CONFIRMATION_TIME      = 2.0
FPS_TARGET             = 15

# Video
VIDEO_WIDTH   = 960
VIDEO_HEIGHT  = 540
CAMERA_INDEX  = 1

# Serial
SERIAL_PORT     = "/dev/ttyACM0"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT  = 1

# Servo commands
SERVO_DEFECT_CMD = "L"
SERVO_OK_CMD     = "C"
SERVO_IDLE_CMD   = "I"

# IR gate trigger char (Arduino sends this when belt tag detected)
IR_TRIGGER_CHAR = "T"

# Classes
CLASS_NAMES    = {0: "Defect", 1: "No Defect"}
DEFECT_CLASSES = [0]

THEME = {
    "bg_void":       "#060a10",
    "bg_deep":       "#0d1117",
    "bg_panel":      "#111720",
    "bg_card":       "#161d28",
    "bg_raised":     "#1c2433",
    "border_dim":    "#1e2d40",
    "border_mid":    "#2a3f58",
    "border_bright": "#3d5a7a",
    "text_white":    "#f0f6fc",
    "text_primary":  "#cdd9e5",
    "text_secondary":"#768390",
    "text_muted":    "#444c56",
    "blue":          "#4da6ff",
    "blue_dim":      "#1f78d1",
    "cyan":          "#39d0d8",
    "green":         "#3fb950",
    "red":           "#f85149",
    "orange":        "#d29922",
}