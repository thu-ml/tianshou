import threading
from torch.utils import tensorboard
from typing import Any, Dict, Optional


class SummaryWriter(tensorboard.SummaryWriter):
    """A more convenient Summary Writer.

    You can get the same instance of summary writer everywhere after you
    created one.
    """

    _mutex_lock = threading.Lock()
    _default_key: str
    _instance: Optional[Dict[str, "SummaryWriter"]] = None

    @classmethod
    def get_instance(
        cls, key: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> "SummaryWriter":
        with SummaryWriter._mutex_lock:
            if key is None:
                key = SummaryWriter._default_key
            if SummaryWriter._instance is None:
                SummaryWriter._instance = {
                    key: SummaryWriter(*args, **kwargs)
                }
                SummaryWriter._default_key = key
            elif key not in SummaryWriter._instance.keys():
                writer = SummaryWriter(*args, **kwargs)
                SummaryWriter._instance.update({key: writer})
        return SummaryWriter._instance[key]
