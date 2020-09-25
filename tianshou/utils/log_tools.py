from torch.utils import tensorboard
import threading


class SummaryWriter(tensorboard.SummaryWriter):
    _mutex_lock = threading.Lock()

    @classmethod
    def get_instance(cls, *args, **kwargs):
        with SummaryWriter._mutex_lock:
            if not hasattr(SummaryWriter, "_instance"):
                SummaryWriter._instance = SummaryWriter(*args, **kwargs)
        return SummaryWriter._instance
