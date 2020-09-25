from torch.utils import tensorboard
import threading


class SummaryWriter(tensorboard.SummaryWriter):
    """
    A more convenient Summary Writer, you can get the same instance of summary
    writer every where after you have create one.

    """
    _mutex_lock = threading.Lock()
    _default_key = None

    @classmethod
    def get_instance(cls, key: str = 'default', *args, **kwargs):
        with SummaryWriter._mutex_lock:
            if not hasattr(SummaryWriter, "_instance"):
                SummaryWriter._instance = {
                    key: SummaryWriter(*args, **kwargs)
                    }
                SummaryWriter._default_key = key
            elif key == 'default':
                key = SummaryWriter._default_key
            elif key not in SummaryWriter._instance.keys():
                writer = SummaryWriter(*args, **kwargs)
                SummaryWriter._instance.update({key: writer})
        return SummaryWriter._instance[key]
