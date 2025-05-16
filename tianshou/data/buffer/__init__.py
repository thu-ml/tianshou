def _backward_compatibility():
    import sys

    from . import buffer_base

    # backward compatibility with persisted buffers from v1 for determinism tests
    sys.modules["tianshou.data.buffer.base"] = buffer_base


_backward_compatibility()
