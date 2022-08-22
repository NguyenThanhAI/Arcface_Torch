import queue
import threading
import time

import cv2


class QueuedStream:

    def __init__(self, uri, drop=True, fps=25):
        self.uri = uri
        self.queue = queue.Queue(maxsize=1)
        self.lock_started = threading.Lock()
        self.fps = fps
        self.opened = False
        self.stopped = False
        self.drop = drop

    def start(self):
        self.lock_started.acquire()
        self.th = threading.Thread(target=self._thread_func)
        self.th.daemon = True
        self.th.start()
        self.lock_started.acquire()

    def read(self):
        if not self.stopped:
            frame, frame_id = self.queue.get(True)
            if frame is None:
                return (False, None, None)
            return (True, frame, frame_id)
        else:
            return (False, None, None)

    def stop(self):
        if not self.stopped:
            self.stopped = True
            try:
                self.queue.get(False)
            except Exception:
                pass
            self.th.join()

    def isOpened(self):
        return self.opened

    def release(self):
        self.stop()

    def estimate_framerate(self):
        return self.fps

    def _thread_func(self):
        '''keep looping infinitely'''
        global IMAGE_W, IMAGE_H, SCALE_RATIO
        if len(self.uri) == 0:
            stream = cv2.VideoCapture(0)
            estimate_fps = True
        else:
            stream = cv2.VideoCapture(self.uri)
            if self.uri.startswith('rtsp://'):
                estimate_fps = True
            else:
                estimate_fps = False
        time.sleep(0.1)
        self.opened = stream.isOpened()

        self.lock_started.release()

        if not self.opened:
            stream.release()
            return
        start_time = time.time()
        frame_id = 0

        while not self.stopped:
            grabbed, frame = stream.read()
            if not grabbed:
                frame = None
                frame_id = None
            else:
                frame_id += 1

            if self.drop:
                try:
                    self.queue.get(False)
                except Exception:
                    pass
                self.queue.put((frame, frame_id))
            else:  # not drop
                self.queue.put((frame, frame_id))

            if frame is None:
                stream.release()
                return

            if not estimate_fps:
                time.sleep(1.0 / self.fps)
            else:
                if frame_id > 25 and self.drop:
                    self.fps = frame_id / (time.time() - start_time)
                elif frame_id > 5 and self.drop:
                    estimate = frame_id / (time.time() - start_time)
                    self.fps = (self.fps + estimate) / 2.0
        # stopped
        if frame is not None:  # stopped.value == True
            try:
                self.queue.get(True, 0.5)
            except Exception:
                pass
        else:
            self.stopped = True
        stream.release()


'''stream = QueuedStream(uri=[], fps=30)
stream.start()

while True:
    ret, frame, frame_id = stream.read()

    if not ret:
        break

    cv2.imshow("", frame)
    cv2.waitKey(1)'''