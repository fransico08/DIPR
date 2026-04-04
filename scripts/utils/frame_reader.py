# utils/frame_reader.py — Background frame-reading thread

import cv2
import time
import threading
import queue

from config import FRAME_QUEUE_SIZE, FRAME_QUEUE_SLEEP_S, FRAME_READ_TIMEOUT_S, FALLBACK_FPS


class FrameReader(threading.Thread):
    def __init__(self, video_path: str, queue_size: int = FRAME_QUEUE_SIZE):
        super().__init__(daemon=True)
        self._video_path  = video_path
        self.q            = queue.Queue(maxsize=queue_size)
        self.stopped      = False
        self._lock        = threading.Lock()
        self._seek_req: int | None = None
        self._reopen_req  = False   # FIX: flag to reopen cap (for replay)

        cap = cv2.VideoCapture(video_path)
        self.width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_src      = cap.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
        cap.release()

    def run(self) -> None:
        cap = cv2.VideoCapture(self._video_path)

        while not self.stopped:
            with self._lock:
                req    = self._seek_req;   self._seek_req   = None
                reopen = self._reopen_req; self._reopen_req = False

            if reopen:
                # FIX: fully reopen the file so cv2 is not stuck at EOF
                cap.release()
                cap = cv2.VideoCapture(self._video_path)
                req = 0   # seek to start after reopen

            if req is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, req)
                with self.q.mutex:
                    self.q.queue.clear()

            if not self.q.full():
                ret, frame = cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.q.put((frame,
                            cap.get(cv2.CAP_PROP_POS_MSEC),
                            int(cap.get(cv2.CAP_PROP_POS_FRAMES))))
            else:
                time.sleep(FRAME_QUEUE_SLEEP_S)

        cap.release()

    def seek(self, frame_no: int) -> None:
        with self._lock:
            self._seek_req = max(0, min(frame_no, self.total_frames - 1))

    def replay(self) -> None:
        """
        Reset to frame 0 for replay — works even after the video has finished.
        Strategy: set reopen flag (run() will reopen the cap from scratch)
        and, if the thread is no longer alive (finished naturally), spawn a
        fresh daemon thread that runs the same _run logic.
        """
        with self._lock:
            self._reopen_req = True
            self._seek_req   = None
        self.stopped = False  # allow loop to continue / new thread to enter

        if not self.is_alive():
            # Thread exited because the video ended; start a brand-new one.
            t = threading.Thread(target=self.run, daemon=True)
            t.start()

    def read(self) -> tuple:
        return self.q.get(timeout=FRAME_READ_TIMEOUT_S)

    def is_done(self) -> bool:
        return self.stopped and self.q.empty()

    def stop(self) -> None:
        self.stopped = True
