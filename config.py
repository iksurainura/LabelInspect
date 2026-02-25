"""
LabelInspect — Configuration
Arduino protocol: single char  'O' = OK,  'D' = DEFECT,  'R' = Reset
"""

# ── Model ────────────────────────────────────────────────────
MODEL_PATH  = "yolo11m-seg.pt"
DEVICE      = "cuda"            # "cuda" | "cpu"
IMG_SIZE    = 640

# ── Detection ────────────────────────────────────────────────
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD  = 0.45
CONFIRMATION_TIME      = 2.0    # seconds stable before state commits
FPS_TARGET             = 15

# ── Video ────────────────────────────────────────────────────
VIDEO_WIDTH   = 960
VIDEO_HEIGHT  = 540
CAMERA_INDEX  = 1

# ── Serial / Arduino ─────────────────────────────────────────
SERIAL_PORT     = "/dev/ttyACM0"    # Windows: "COM3"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT  = 1

# ── Arduino Commands (single char) ───────────────────────────
# 'O' → labelType=1 → arms irOK  (pin 2) → sweeps servoOK  (pin 9)
# 'D' → labelType=0 → arms irDEFECT (pin 3) → sweeps servoDEFECT (pin 10)
# 'R' → reset: conveyor ON, clears labelType
CMD_OK     = "O"
CMD_DEFECT = "D"
CMD_RESET  = "R"

# ── Classes ──────────────────────────────────────────────────
CLASS_NAMES    = {0: "Defect", 1: "No Defect"}
DEFECT_CLASSES = [0]

# ── Palette  (ALL keys used in app.py live here) ─────────────
THEME = {
    # backgrounds
    "void":          "#060a10",
    "deep":          "#0d1117",
    "panel":         "#0f1621",
    "card":          "#131a24",
    "raised":        "#18212e",
    "hover":         "#1d2a3a",
    # borders
    "b_dim":         "#1a2840",
    "b_mid":         "#243650",
    "b_bright":      "#2e4a6e",
    # text
    "white":         "#f0f6fc",
    "t_primary":     "#cdd9e5",
    "t_secondary":   "#6b7d8f",
    "t_muted":       "#394655",
    # accents
    "blue":          "#4da6ff",
    "blue_dim":      "#1f78d1",
    "cyan":          "#39d0d8",
    "green":         "#3fb950",
    "red":           "#f85149",
    "orange":        "#d29922",
}