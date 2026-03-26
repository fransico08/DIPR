MODEL_PATH = "yolov8n.pt"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
YOLO_IMGSZ = 640
DETECT_EVERY = 3
MAX_AGE = 10
N_INIT = 5
MAX_IOU_DIST = 0.7
SPEED_WINDOW = 8
MIN_HISTORY = 4
SPEED_SMOOTH = 0.35

ROAD_WIDTH_M = 3.0
ROAD_LENGTH_M = 7.0

# --- Camera geometry for auto-calibration ---
DEFAULT_CAM_HEIGHT_M   = 6.0    # metres above road surface
DEFAULT_CAM_TILT_DEG   = 30.0   # depression angle below horizontal (positive = looking down)
DEFAULT_CAM_FOV_H_DEG  = 60.0   # horizontal field of view in degrees
DEFAULT_ROAD_SLOPE_DEG = 0.0    # road slope, positive = uphill away from camera
AUTO_ROI_TOP    = 0.35          # top boundary of ROI (fraction of frame height)
AUTO_ROI_BOT    = 0.90          # bottom boundary of ROI (fraction of frame height)

DEFAULT_IMG_PTS = [[200,360],[600,360],[100,620],[700,620]]

VEHICLE_CLASSES = {2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}