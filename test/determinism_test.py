import torch
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tianshou.utils.determinism import TraceDeterminismTest, TraceLoggerContext


class TorchDeterministicModeContext:
    def __init__(self, mode="default"):
        self.new_mode = mode
        self.original_mode = None

    def __enter__(self):
        self.original_mode = torch.get_deterministic_debug_mode()
        torch.set_deterministic_debug_mode(self.new_mode)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_deterministic_debug_mode(self.original_mode)


class AlgorithmDeterminismTest:
    def __init__(self, name: str, main_fn: Callable[[Namespace], Any], args: Namespace):
        self.determinism_test = TraceDeterminismTest(
            base_path=Path(__file__).parent / "resources" / "determinism",
        )
        self.name = name

        def set(attr: str, value: Any) -> None:
            old_value = getattr(args, attr)
            if old_value is None:
                raise ValueError(f"Attribute '{attr}' is not defined for args: {args}")
            setattr(args, attr, value)

        set("epoch", 3)
        set("step_per_epoch", 100)
        set("device", "cpu")
        set("training_num", 1)
        set("test_num", 1)

        # run the actual process
        with TraceLoggerContext() as trace:
            with TorchDeterministicModeContext():
                main_fn(args)
            self.log = trace.get_log()

    def run(self, update_snapshot: bool = False) -> None:
        self.determinism_test.check(self.log, self.name, create_reference_result=update_snapshot)
