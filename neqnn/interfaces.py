import time
from torch.utils.data import IterableDataset


class OnlineAlignedSensorsDataset(IterableDataset):
    def __init__(self, hub, poll_interval=0.05):
        self.hub = hub
        self.poll_interval = poll_interval

    def __iter__(self):
        while True:
            snap = self.hub.snapshot()
            if snap is not None:
                yield snap
            time.sleep(self.poll_interval)
