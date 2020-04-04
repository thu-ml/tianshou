import cloudpickle


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in :class:`~tianshou.env.SubprocVectorEnv`"""

    def __init__(self, data):
        self.data = data

    def __getstate__(self):
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data):
        self.data = cloudpickle.loads(data)
