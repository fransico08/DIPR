import cv2

COLOR_PALETTE = [
    (255,100,100), (100,255,100), (100,180,255), (255,200,50),
    (50,220,220), (220,50,220), (255,150,50), (50,255,150), (180,50,255),
]

_cmap = {}

def get_color(tid):
    k = int(tid)
    if k not in _cmap:
        _cmap[k] = COLOR_PALETTE[k % len(COLOR_PALETTE)]
    return _cmap[k]

def draw_track(frame, l, t, r, b, tid, cls_name, speed, color):
    cv2.rectangle(frame,(l, t),(r, b), color, 2)
    label = f"ID{int(tid)} {cls_name} {speed:.1f}km/h"
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
    p = 3
    cv2.rectangle(frame,(l, t - th - 2 * p - bl), (l + tw + 2 * p, t), color, -1)
    cv2.putText(frame, label, (l + p, t - p - bl),
                cv2.FONT_HERSHEY_DUPLEX, 0.46, (15, 15, 15), 1, cv2.LINE_AA)
    cx, cy = (l + r) // 2, b
    cv2.circle(frame, (cx, cy), 4, color, -1)
    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 1)

def draw_roi(frame, pts):
    if not pts or len(pts) < 2:
        return
    ip = [list(map(int, p)) for p in pts]
    
    for i in range(len(ip)):
        cv2.line(frame, tuple(ip[i]), tuple(ip[(i + 1) % len(ip)]), (0, 255, 180), 1, cv2.LINE_AA)
        
    for i, pt in enumerate(ip):
        cv2.circle(frame, tuple(pt), 5, (0, 255, 180), -1)
        cv2.putText(frame, str(i + 1), (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 180), 1)

def draw_hud(frame, fps, tracked, calibrated, paused):
    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (320, 90), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "VEHICLE SPEED ESTIMATION",
                (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.5, (80, 210, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}   Tracked: {tracked}",
                (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1, cv2.LINE_AA)
    cc = (80, 255, 80) if calibrated else (60, 60, 255)
    cv2.putText(frame, f"Homography: {'Calibrated' if calibrated else 'Default'}",
                (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.43, cc, 1, cv2.LINE_AA)
    sc = (0, 200, 255) if paused else (80, 255, 80)
    cv2.putText(frame, "PAUSED" if paused else "RUNNING",
                (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.43, sc, 1, cv2.LINE_AA)
