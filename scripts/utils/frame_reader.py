# utils/frame_reader.py — Background frame-reading thread

import cv2
import time
import threading
import queue

from config import FRAME_QUEUE_SIZE, FRAME_QUEUE_SLEEP_S, FRAME_READ_TIMEOUT_S, FALLBACK_FPS


class FrameReader(threading.Thread):
    def __init__(self, video_path: str, queue_size: int = FRAME_QUEUE_SIZE):
        super().__init__(daemon=True)
        self.cap          = cv2.VideoCapture(video_path)
        self.q            = queue.Queue(maxsize=queue_size)
        self.stopped      = False
        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_src      = self.cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
        self._seek_req: int | None = None
        self._lock        = threading.Lock()

    def run(self) -> None:
        while not self.stopped:
            with self._lock:
                req = self._seek_req; self._seek_req = None
            if req is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, req)
                with self.q.mutex: self.q.queue.clear()
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret: self.stopped = True; break
                self.q.put((frame,
                            self.cap.get(cv2.CAP_PROP_POS_MSEC),
                            int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))))
            else:
                time.sleep(FRAME_QUEUE_SLEEP_S)
        self.cap.release()

    def seek(self, frame_no: int) -> None:
        with self._lock:
            self._seek_req = max(0, min(frame_no, self.total_frames - 1))

    def read(self) -> tuple:
        return self.q.get(timeout=FRAME_READ_TIMEOUT_S)

    def is_done(self) -> bool:
        return self.stopped and self.q.empty()

    def stop(self) -> None:
        self.stopped = True
