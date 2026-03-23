import cv2
import time
import threading
import queue

class FrameReader(threading.Thread):
    def __init__(self, video_path, queue_size=6):
        super().__init__(daemon=True)
        self.cap          = cv2.VideoCapture(video_path)
        self.q            = queue.Queue(maxsize=queue_size)
        self.stopped      = False
        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_src      = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._seek_req    = None
        self._lock        = threading.Lock()

    def run(self):
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
                time.sleep(0.001)
        self.cap.release()

    def seek(self, frame_no):
        with self._lock:
            self._seek_req = max(0, min(frame_no, self.total_frames - 1))

    def read(self):
        return self.q.get(timeout=3.0)
    
    def is_done(self):
        return self.stopped and self.q.empty()
    
    def stop(self):
        self.stopped = True