import threading
from numbers import Number
import numpy as np
from torch.utils import tensorboard
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class SummaryWriter(tensorboard.SummaryWriter):
    """A more convenient Summary Writer(`tensorboard.SummaryWriter`).

    You can get the same instance of summary writer everywhere after you
    created one.
    ::

        >>> writer1 = SummaryWriter.get_instance(
                key="first", log_dir="log/test_sw/first")
        >>> writer2 = SummaryWriter.get_instance()
        >>> writer1 is writer2
        True
        >>> writer4 = SummaryWriter.get_instance(
                key="second", log_dir="log/test_sw/second")
        >>> writer5 = SummaryWriter.get_instance(key="second")
        >>> writer1 is not writer4
        True
        >>> writer4 is writer5
        True
    """

    _mutex_lock = threading.Lock()
    _default_key: str
    _instance: Optional[Dict[str, "SummaryWriter"]] = None

    @classmethod
    def get_instance(
        cls,
        key: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "SummaryWriter":
        """Get instance of torch.utils.tensorboard.SummaryWriter by key."""
        with SummaryWriter._mutex_lock:
            if key is None:
                key = SummaryWriter._default_key
            if SummaryWriter._instance is None:
                SummaryWriter._instance = {}
                SummaryWriter._default_key = key
            if key not in SummaryWriter._instance.keys():
                SummaryWriter._instance[key] = SummaryWriter(*args, **kwargs)
        return SummaryWriter._instance[key]

class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer."""
    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    @abstractmethod
    def write(self, key: str,
              x: Union[Number, np.number, np.ndarray],
              y: Union[Number, np.number, np.ndarray],
              **kwargs: Any) -> None:
        """Specifies how writer is used to log data.

            :param key: namespace which the input data tuple belongs to.
            :param x: stands for the ordinate of the input data tuple.
            :param x: stands for the abscissa of the input data tuple.
        """
        pass

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

            :param collect_result: a dict containing information of data collected
                in training stage, e.g. returns of collect() method in collector.
        """
        pass

    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

            :param update_result: a dict containing information of data collected
                in updating stage, e.g. returns of update() method in policy.
        """
        pass

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

            :param collect_result: a dict containing information of data collected
                in evaluating stage, e.g. returns of collect() method in collector.
        """
        pass

    def set_checkpoint(self, **kwargs: Any) -> None:
        """Set up checkpoint during training by saving policy, rendering, etc."""
        pass

class BasicLogger(BaseLogger):
    """A loggger that relies on tensorboard.SummaryWriter by default to visualise
    and log statistics. You can also rewrite write() func to use your own writer."""
    def __init__(self, writer,
                 train_interval = 1, test_interval = 1, update_interval = 1000,
                 save_path = None):  
        super().__init__(writer)
        self.n_trainlog = 0
        self.n_testlog = 0
        self.n_updatelog = 0
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.save_path = save_path
        
    def write(self, key, x, y):
        self.writer.add_scalar(key, y, global_step=x)

    def log_train_data(self, collect_result, step):
        if collect_result["n/ep"] > 0:
            if 'rew' not in collect_result:
                collect_result['rew'] = collect_result['rews'].mean()
            if 'len' not in collect_result:
                collect_result['len'] = collect_result['lens'].mean()
            if step - self.last_log_train_step >= self.train_interval:
                self.write("train/n/ep", step, collect_result["n/ep"])  
                self.write("train/rew", step, collect_result["rew"])
                self.write("train/len", step, collect_result["len"])
                self.last_log_train_step = step
                self.n_trainlog += 1

    def log_test_data(self, collect_result, step):
        assert(collect_result["n/ep"] > 0)
        if 'rew' not in collect_result:
            collect_result['rew'] = collect_result['rews'].mean()
        if 'len' not in collect_result:
            collect_result['len'] = collect_result['lens'].mean()
        if 'rew_std' not in collect_result:
            collect_result['rew_std'] = collect_result['rews'].std()
        if 'len_std' not in collect_result:
            collect_result['len_std'] = collect_result['lens'].std()
        if step - self.last_log_test_step >= self.test_interval:
            self.write("test/rew", step, collect_result["rew"])
            self.write("test/len", step, collect_result["len"])
            self.write("test/rew_std", step, collect_result["rew_std"])
            self.write("test/len_std", step, collect_result["len_std"])
            self.last_log_test_step = step
        self.n_testlog += 1        

    def log_update_data(self, update_result, step):
        if step - self.last_log_update_step >= self.update_interval:
            for k, v in update_result.items():
                self.write(k, step, v)
            self.last_log_update_step = step
            self.n_updatelog += 1

    def global_log(self, **kwargs):
        if 'policy' in kwargs and self.save_path:
            import torch
            torch.save(kwargs['policy'].state_dict(), self.save_path)

class LazyLogger(BasicLogger):
    """A loggger that does nothing. Used as placeholder in trainer."""
    def __init__(self):
        super().__init__(None)

    def write(self, key, x, y):
        pass