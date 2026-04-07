# core/calibration.py — Homography calibration (camera params method)

import cv2
import math
import numpy as np

from config import (
    DEFAULT_IMG_PTS, ROAD_WIDTH_M, ROAD_LENGTH_M,
    HOMOGRAPHY_RANSAC_THRESH, CAM_NEAR_FACTOR,
)


class HomographyCalibrator:
    """
    Maps image pixels → real-world metres via a perspective homography.

    Method: camera parameters (no manual clicking).
        cal.apply_camera_params(img_w, img_h,
            cam_height_m, cam_tilt_deg, fov_h_deg,
            road_width_m, road_depth_m, road_slope_deg)

    Legacy manual method still available:
        cal.apply_points([[x1,y1],...], width_m, length_m)
    """

    def __init__(self):
        self.image_points: list | None = None
        self._H:           np.ndarray | None = None
        self.real_w:       float = ROAD_WIDTH_M
        self.real_l:       float = ROAD_LENGTH_M
        self.cam_height_m:  float | None = None
        self.cam_tilt_deg:  float | None = None
        self.fov_h_deg:     float | None = None
        self.road_slope_deg: float = 0.0

    def is_calibrated(self) -> bool:
        return self._H is not None

    def pixel_to_world(self, point) -> np.ndarray:
        if self._H is None:
            return np.array([0.0, 0.0])
        src = np.array([[point]], dtype=np.float32)
        return cv2.perspectiveTransform(src, self._H)[0][0]

    # ── Method A: manual points ───────────────────────────────

    def apply_points(self, clicked: list, w_m: float, l_m: float) -> None:
        self.real_w, self.real_l = w_m, l_m
        real_pts = np.float32([
            [0,   0  ], [w_m, 0  ],
            [0,   l_m], [w_m, l_m],
        ])
        self._compute(np.float32(clicked), real_pts)

    # ── Method B: camera parameters ──────────────────────────

    def apply_camera_params(self,
                            img_w: int, img_h: int,
                            cam_height_m: float,
                            cam_tilt_deg: float,
                            fov_h_deg: float,
                            road_width_m: float,
                            road_depth_m: float,
                            road_slope_deg: float = 0.0) -> None:
        """
        Phiên bản ĐÚNG - Sử dụng rotation matrix chuẩn cho camera tilt xuống.
        Trapezoid sẽ nằm gọn trong vùng dưới của video.
        """
        self.cam_height_m   = float(cam_height_m)
        self.cam_tilt_deg   = float(cam_tilt_deg)
        self.fov_h_deg      = float(fov_h_deg)
        self.real_w         = float(road_width_m)
        self.real_l         = float(road_depth_m)
        self.road_slope_deg = float(road_slope_deg)

        tilt_rad  = math.radians(cam_tilt_deg)
        slope_rad = math.radians(road_slope_deg)
        fov_h_rad = math.radians(fov_h_deg)

        fx = 1.0 / math.tan(fov_h_rad / 2)
        aspect = img_w / float(img_h)
        fov_v_rad = 2 * math.atan(math.tan(fov_h_rad / 2) / aspect)
        fy = 1.0 / math.tan(fov_v_rad / 2)

        def world_to_pixel(world_x: float, world_y: float) -> tuple[int, int]:
            # Z của điểm trên đường
            road_z = world_y * math.tan(slope_rad)
            dx = world_x
            dy = world_y
            dz = road_z - cam_height_m   # camera ở trên đường

            # ====================== ROTATION MATRIX CHUẨN ======================
            # Positive tilt = camera nhìn xuống đường
            cam_x = dx
            cam_y = dy * math.sin(tilt_rad) + dz * math.cos(tilt_rad)
            cam_z = dy * math.cos(tilt_rad) - dz * math.sin(tilt_rad)
            # ===================================================================

            if cam_z < 0.5:
                # Điểm nằm sau camera → đẩy về phía dưới ảnh (near)
                scale = 1200.0
                ndx = cam_x * scale
                ndy = cam_y * scale
            else:
                ndx = cam_x / cam_z
                ndy = cam_y / cam_z

            u = int(img_w / 2 + ndx * fx * (img_w / 2))
            v = int(img_h / 2 - ndy * fy * (img_h / 2))

            u = max(0, min(img_w - 1, u))
            v = max(0, min(img_h - 1, v))
            return u, v

        # Tính near_y và far_y
        near_y = max(2.0, cam_height_m * 0.5)
        far_y  = max(near_y + 12.0, road_depth_m)

        half_w = road_width_m / 2.0

        # Far = xa camera → thường nằm phía trên ảnh
        tl = world_to_pixel(-half_w, far_y)
        tr = world_to_pixel( half_w, far_y)
        bl = world_to_pixel(-half_w, near_y)
        br = world_to_pixel( half_w, near_y)

        self.image_points = [list(tl), list(tr), list(bl), list(br)]

        img_pts  = np.float32([tl, tr, bl, br])
        real_pts = np.float32([
            [0,            far_y ],
            [road_width_m, far_y ],
            [0,            near_y],
            [road_width_m, near_y],
        ])

        self._H, _ = cv2.findHomography(img_pts, real_pts, cv2.RANSAC, HOMOGRAPHY_RANSAC_THRESH)

        print(f"[Cal] Applied → h={cam_height_m:.1f}m tilt={cam_tilt_deg:.1f}° fov={fov_h_deg:.1f}° "
              f"slope={road_slope_deg:.1f}° road={road_width_m:.1f}x{road_depth_m:.1f}m")
        print(f"[Cal] near_y={near_y:.1f} | far_y={far_y:.1f}")
        print(f"[Cal] Corners → TL{tl} TR{tr} BL{bl} BR{br}")    # ── Persistence ───────────────────────────────────────────

    def save(self, path: str) -> None:
        data: dict = dict(
            image_points = np.float32(self.image_points or [[0,0]]*4),
            H            = self._H if self._H is not None else np.eye(3, np.float32),
            real_w       = self.real_w,
            real_l       = self.real_l,
            road_slope_deg = self.road_slope_deg,
        )
        if self.cam_height_m is not None:
            data.update(cam_height_m=self.cam_height_m,
                        cam_tilt_deg=self.cam_tilt_deg,
                        fov_h_deg=self.fov_h_deg)
        np.savez(path, **data)
        print(f"[Calibration] Saved -> {path}")

    def load(self, path: str) -> bool:
        try:
            d = np.load(path)
            self.image_points   = d["image_points"].tolist()
            self._H             = d["H"]
            self.real_w         = float(d.get("real_w",  ROAD_WIDTH_M))
            self.real_l         = float(d.get("real_l",  ROAD_LENGTH_M))
            self.road_slope_deg = float(d["road_slope_deg"]) if "road_slope_deg" in d else 0.0
            self.cam_height_m   = float(d["cam_height_m"]) if "cam_height_m" in d else None
            self.cam_tilt_deg   = float(d["cam_tilt_deg"]) if "cam_tilt_deg" in d else None
            self.fov_h_deg      = float(d["fov_h_deg"])    if "fov_h_deg"    in d else None
            print(f"[Calibration] Loaded -> {path}")
            return True
        except Exception:
            return False

    def set_default(self) -> None:
        real_pts = np.float32([
            [0, 0], [ROAD_WIDTH_M, 0],
            [0, ROAD_LENGTH_M], [ROAD_WIDTH_M, ROAD_LENGTH_M],
        ])
        self._compute(np.float32(DEFAULT_IMG_PTS), real_pts)
        print("[Calibration] Using default homography.")

    def _compute(self, img_pts: np.ndarray, real_pts: np.ndarray) -> None:
        self.image_points = [list(p) for p in img_pts]
        self._H, _ = cv2.findHomography(
            img_pts, real_pts, cv2.RANSAC, HOMOGRAPHY_RANSAC_THRESH)
        if self._H is None:
            print("[Calibration] findHomography failed — using default.")
            self.set_default()
