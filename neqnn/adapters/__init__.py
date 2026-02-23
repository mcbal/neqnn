import threading
import time
from abc import ABC, abstractmethod
from collections import deque


class TimeBuffer:
    def __init__(self, maxlen=32):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def push(self, value):
        with self.lock:
            self.buffer.append((time.time(), value))

    def get_latest_before(self, t, max_age=None):
        with self.lock:
            candidates = [x for x in self.buffer if x[0] <= t]
            if not candidates:
                return None
            ts, val = candidates[-1]
            if max_age is not None and (t - ts) > max_age:
                return None
            return ts, val


class Adapter(ABC):
    def __init__(self, name, buffer_size=32):
        self.name = name
        self.buffer = TimeBuffer(maxlen=buffer_size)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @abstractmethod
    def _run(self):
        pass


class AdapterHub:
    def __init__(self, adapters, reference="webcam", max_age=0.2):
        self.adapters = {a.name: a for a in adapters}
        self.reference = reference
        self.max_age = max_age

    def snapshot(self):
        ref_adapter = self.adapters[self.reference]

        with ref_adapter.buffer.lock:
            if not ref_adapter.buffer.buffer:
                return None
            ref_ts, _ = ref_adapter.buffer.buffer[-1]

        aligned = {}
        for name, adapter in self.adapters.items():
            item = adapter.buffer.get_latest_before(ref_ts, max_age=self.max_age)
            if item is None:
                return None
            ts, val = item
            aligned[name] = {"value": val, "timestamp": ts}

        return aligned
