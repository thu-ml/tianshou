

class ReplayBufferBase(object):
    """
    base class for replay buffer.
    """
    def add(self, frame):
        raise NotImplementedError()

    def remove(self):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()