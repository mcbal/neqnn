import sounddevice as sd
import time

from neqnn.adapters import Adapter


class AudioAdapter(Adapter):
    def __init__(self, samplerate=16000, blocksize=1024):
        self.samplerate = samplerate
        self.blocksize = blocksize
        super().__init__("audio")

    def _callback(self, indata, frames, time_info, status):
        if status:
            return
        self.buffer.push(indata.copy())

    def _run(self):
        with sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            callback=self._callback,
        ):
            while self.running:
                time.sleep(0.1)
