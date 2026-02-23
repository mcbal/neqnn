import cv2
import numpy as np

from neqnn.adapters import Adapter


class WebcamAdapter(Adapter):
    def __init__(self, device=0, resize=(320, 240)):
        self.cap = cv2.VideoCapture(device)
        self.resize = resize
        super().__init__("webcam")

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            if self.resize:
                frame = cv2.resize(frame, self.resize)

            # Explicitly document dtype
            assert frame.dtype == np.uint8
            self.buffer.push(frame)
