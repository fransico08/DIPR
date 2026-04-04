# =============================================================
#  config.py — single source of truth for all constants
#  No magic numbers should appear anywhere else in the codebase.
# =============================================================

# ── Model ─────────────────────────────────────────────────────
MODEL_PATH       = "yolov8n.pt"
CONF_THRESHOLD   = 0.25
IOU_THRESHOLD    = 0.45
YOLO_IMGSZ       = 640

# ── Detection & tracking ──────────────────────────────────────
DETECT_EVERY     = 3      # run YOLO every N frames
MAX_AGE          = 10     # DeepSORT: frames to keep a lost track
N_INIT           = 5      # DeepSORT: frames before track is confirmed
MAX_IOU_DIST     = 0.7    # DeepSORT: max IoU distance for association
MAX_COSINE_DIST  = 0.4    # DeepSORT: max cosine distance for re-id

# ── Speed estimation ──────────────────────────────────────────
SPEED_WINDOW     = 12      # sliding-window size (frames)
MIN_HISTORY      = 6      # minimum samples before reporting speed
SPEED_SMOOTH     = 0.35   # EMA alpha  (0=no update, 1=no smoothing)

# ── Homography calibration defaults ───────────────────────────
ROAD_WIDTH_M     = 3.5    # default real-world width  (metres)
ROAD_LENGTH_M    = 7.0    # default real-world length (metres)
HOMOGRAPHY_RANSAC_THRESH = 5.0   # RANSAC reprojection threshold (px)
DEFAULT_IMG_PTS  = [[200, 360], [600, 360], [100, 620], [700, 620]]

# ── HSV histogram embedder ────────────────────────────────────
EMBED_BINS_H     = 32     # hue bins
EMBED_BINS_S     = 32     # saturation bins
EMBED_BINS_V     = 32     # value bins
EMBED_DIM        = EMBED_BINS_H + EMBED_BINS_S + EMBED_BINS_V  # = 96

# ── Vehicle classes (COCO) ────────────────────────────────────
VEHICLE_CLASSES  = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ── Display ───────────────────────────────────────────────────
TOPBAR_H         = 40
SIDEBAR_W        = 280    # sidebar width in pixels
SCREEN_MARGIN    = 80     # pixels kept free around the window
CAL_SCREEN_MARGIN = 120    # margin used during calibration window

# ── FPS smoothing ─────────────────────────────────────────────
FPS_BUF_SIZE     = 60     # rolling window size for FPS measurement
FPS_EPSILON      = 1e-9   # prevents division-by-zero in FPS calc

# ── Seekbar step sizes ────────────────────────────────────────
SEEK_STEP_SHORT  = 5      # seconds
SEEK_STEP_LONG   = 10     # seconds

# ── Calibration window UI ────────────────────────────────────
CAL_NEEDED_POINTS    = 4
CAL_POINT_RADIUS     = 8
CAL_POINT_COLOR      = (0, 255, 100)
CAL_LINE_COLOR       = (0, 210, 255)
CAL_LINE_THICKNESS   = 2
CAL_STATUS_COLOR     = (100, 255, 180)
CAL_TEXT_FONT        = 0          # cv2.FONT_HERSHEY_SIMPLEX (avoid cv2 import here)
CAL_TEXT_SCALE_MAIN  = 0.8
CAL_TEXT_SCALE_STATUS = 0.55
CAL_TEXT_THICKNESS   = 2
CAL_TEXT_OFFSET_X    = 10
CAL_TEXT_OFFSET_Y    = -10
CAL_STATUS_MARGIN_BOTTOM = 12

# ── Drawing / HUD ─────────────────────────────────────────────
HUD_RECT_X2      = 320
HUD_RECT_Y2      = 90
HUD_OVERLAY_ALPHA = 0.55   # background alpha for HUD overlay
ROI_COLOR        = (0, 255, 180)
ROI_DOT_RADIUS   = 5
TRACK_LABEL_FONT_SCALE = 0.46
TRACK_DOT_RADIUS = 4
TRACK_DOT_BORDER = (255, 255, 255)

# ── Sidebar UI colours ────────────────────────────────────────
COLOR_SIDEBAR_BG     = "#1a1a2e"
COLOR_HEADER_BG      = "#0f3460"
COLOR_HEADER_FG      = "#00d4ff"
COLOR_FRAME_FG       = "#c0c0c0"
COLOR_LABEL_FG       = "#e8e8e8"
COLOR_ENTRY_BG       = "#2c3e50"
COLOR_BTN_DEFAULT    = "#548eb3"
COLOR_BTN_DANGER     = "#9b2226"
COLOR_BTN_SEEK       = "#2c3e50"
COLOR_STATUS_WARN    = "#f4a261"

# ── Fonts ─────────────────────────────────────────────────────
FONT_TITLE   = ("Consolas", 13, "bold")
FONT_NORMAL  = ("Consolas", 10)
FONT_BOLD    = ("Consolas", 10, "bold")
FONT_SMALL   = ("Consolas", 9)

# ── Color palette for tracks ──────────────────────────────────
COLOR_PALETTE = [
    (255, 100, 100), (100, 255, 100), (100, 180, 255), (255, 200,  50),
    ( 50, 220, 220), (220,  50, 220), (255, 150,  50), ( 50, 255, 150),
    (180,  50, 255),
]

# ── Speed conversion ──────────────────────────────────────────
MS_TO_KMH            = 3.6     # multiply m/s by this to get km/h
TS_MS_TO_S           = 1000.0  # divide timestamp ms by this to get seconds

# ── HSV channel ranges ────────────────────────────────────────
HSV_HUE_MAX          = 180
HSV_SAT_MAX          = 256
HSV_VAL_MAX          = 256

# ── Frame reader ──────────────────────────────────────────────
FRAME_QUEUE_SIZE     = 6
FRAME_QUEUE_SLEEP_S  = 0.001   # sleep when queue is full
FRAME_READ_TIMEOUT_S = 3.0     # max wait for next frame
FALLBACK_FPS         = 30.0    # used when video reports 0 fps

# ── HUD positions and colours ─────────────────────────────────
HUD_ORIGIN_X         = 8
HUD_ORIGIN_Y         = 8
HUD_BG_COLOR         = (10, 10, 10)
HUD_LINE1_Y          = 28
HUD_LINE2_Y          = 48
HUD_LINE3_Y          = 66
HUD_LINE4_Y          = 84
HUD_TEXT_X           = 16
HUD_TEXT_SCALE_TITLE = 0.5
HUD_TEXT_SCALE_BODY  = 0.43
HUD_TITLE_COLOR      = (80, 210, 255)
HUD_BODY_COLOR       = (200, 200, 200)
HUD_CAL_OK_COLOR     = (80,  255,  80)
HUD_CAL_NO_COLOR     = (60,   60, 255)
HUD_RUNNING_COLOR    = (80,  255,  80)
HUD_PAUSED_COLOR     = ( 0,  200, 255)

# ── ROI label offset ──────────────────────────────────────────
ROI_LABEL_OFFSET_X   = 6
ROI_LABEL_OFFSET_Y   = -6
ROI_LABEL_SCALE      = 0.4

# ── Track label ───────────────────────────────────────────────
TRACK_LABEL_PAD      = 3
TRACK_LABEL_TEXT_COLOR = (15, 15, 15)

# ── UI layout padding ─────────────────────────────────────────
UI_HEADER_PADY       = 10
UI_SECTION_PADX      = 6
UI_SECTION_PADY      = 4
UI_SECTION_PACK_PADX = 8
UI_SECTION_PACK_PADY = 4
UI_BTN_PADX          = 4
UI_BTN_PADY          = 5
UI_BTN_PACK_PADY     = 2
UI_STATUS_WRAP_PX    = 220
UI_STATUS_PACK_PADY  = 4
UI_SEEKBAR_PADY      = (4, 2)

# ── Calibration canvas status label ──────────────────────────
CAL_STATUS_POS_X     = 10

# ── Calibration window status label (canvas) ─────────────────
CAL_STATUS_X             = 10    # x position of "Points: N/4" label

# ── Calibration sidebar status widget ────────────────────────
UI_STATUS_FONT_SIZE      = 8     # font size for the warning/info label
UI_STATUS_WRAP_PX        = 220   # wraplength in pixels
UI_STATUS_BOTTOM_PADY    = 4     # padding below status label

# ── Speed stabilisation ───────────────────────────────────────
SPEED_EMA_ALPHA      = 0.08   # heavy smoothing EMA (lower = smoother)
SPEED_DEADBAND_KMH   = 5.0    # ignore changes smaller than this (km/h)
SPEED_SPIKE_RATIO    = 1.5    # reject samples > this × current speed
SPEED_MIN_VALID      = 0.5    # below this (km/h) treat as stationary

# ── Camera-based auto-calibration ────────────────────────────
CAM_HEIGHT_M_DEFAULT  = 6.0   # camera mounting height above ground (m)
CAM_TILT_DEG_DEFAULT  = 30.0  # tilt below horizontal (degrees, 0=horizontal)
CAM_FOV_H_DEG_DEFAULT = 60.0  # horizontal field of view (degrees)
CAM_ROAD_WIDTH_DEFAULT = 7.0  # visible road width at calibration depth (m)
CAM_ROAD_DEPTH_DEFAULT = 20.0 # distance from camera base to far edge (m)

# ── Per-video config file ─────────────────────────────────────
VIDEO_CONFIG_SUFFIX   = "_cal.json"  # appended to video stem for save path

# ── Camera-based auto-calibration (additions) ────────────────
CAM_SLOPE_DEG_DEFAULT  = 0.0   # road slope: positive = uphill away from cam (degrees)
CAM_NEAR_FACTOR        = 0.3   # near_y = cam_height * this factor (min 1.0 m)

# ── ROI drawing (shared between calibration preview and live HUD) ──
ROI_QUAD_ORDER     = (0, 1, 3, 2)   # TL,TR,BR,BL index order to close polygon
ROI_LABEL_NAMES    = ("TL", "TR", "BL", "BR")
ROI_LABEL_FONT_SCALE_CAL = 0.6      # font scale used in calibration window
ROI_DOT_RADIUS_CAL = 6              # dot radius in calibration window
