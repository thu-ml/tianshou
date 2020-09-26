import threading
from torch.utils import tensorboard
from typing import Dict, Optional


class SummaryWriter(tensorboard.SummaryWriter):
    """A more convenient Summary Writer.

    You can get the same instance of summary writer everywhere after you
    created one.
    """

    _mutex_lock = threading.Lock()
    _default_key: str
    _instance: Optional[Dict[str, "SummaryWriter"]] = None

    @classmethod
    def get_instance(cls,
                     key: Optional[str] = None,
                     log_dir: Optional[str] = None,
                     comment: str = '',
                     purge_step: Optional[int] = None,
                     max_queue: int = 10,
                     flush_secs: int = 120,
                     filename_suffix: str = '') -> "SummaryWriter":
        with SummaryWriter._mutex_lock:
            if key is None:
                key = SummaryWriter._default_key
            if SummaryWriter._instance is None:
                writer = SummaryWriter(log_dir=None,
                                       comment='',
                                       purge_step=None,
                                       max_queue=10,
                                       flush_secs=120,
                                       filename_suffix='')
                SummaryWriter._instance = {key: writer}
                SummaryWriter._default_key = key
            elif key not in SummaryWriter._instance.keys():
                writer = SummaryWriter(log_dir, comment, purge_step, max_queue,
                                       flush_secs, filename_suffix)
                SummaryWriter._instance.update({key: writer})
        return SummaryWriter._instance[key]
