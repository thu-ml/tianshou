class dummy_tqdm:
    """A dummy tqdm class that keeps the total and n stats, but shows
    no progress bar.

    It supports __enter__ and __exit__, update and a dummy set_postfix:
    the interface that trainers use.
    """

    def __init__(self, total, **kwargs):
        self.total = total
        self.n = 0

    def set_postfix(self, **kwargs):
        pass

    def update(self, n=1):
        self.n += n

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
    