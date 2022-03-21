import warnings

warnings.simplefilter("once", DeprecationWarning)


def deprecation(msg: str) -> None:
    """Deprecation warning wrapper."""
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
