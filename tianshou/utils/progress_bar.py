from typing import Any

tqdm_config = {
    "dynamic_ncols": True,
    "ascii": True,
}


class DummyTqdm:
    """A dummy tqdm class that keeps stats but without progress bar.

    It supports ``__enter__`` and ``__exit__``, update and a dummy
    ``set_postfix``, which is the interface that trainers use.

    .. note::

        Using ``disable=True`` in tqdm config results in infinite loop, thus
        this class is created. See the discussion at #641 for details.
    """

    def __init__(self, total: int, **kwargs: Any):
        self.total = total
        self.n = 0

    def set_postfix(self, **kwargs: Any) -> None:
        pass

    def update(self, n: int = 1) -> None:
        self.n += n

    def __enter__(self) -> "DummyTqdm":
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        pass
