from argparse import Namespace
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from tianshou.utils.determinism import TraceDeterminismTest, TraceLoggerContext


class TorchDeterministicModeContext:
    def __init__(self, mode: str | int = "default") -> None:
        self.new_mode = mode
        self.original_mode: str | int | None = None

    def __enter__(self) -> None:
        self.original_mode = torch.get_deterministic_debug_mode()
        torch.set_deterministic_debug_mode(self.new_mode)

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore
        assert self.original_mode is not None
        torch.set_deterministic_debug_mode(self.original_mode)


class AlgorithmDeterminismTest:
    """
    Represents a determinism test for Tianshou's RL algorithms.

    A test using this class should be added for every algorithm in Tianshou.
    Then, when making changes to one or more algorithms (e.g. refactoring), run the respective tests
    on the old branch (creating snapshots) and then on the new branch that contains the changes
    (comparing with the snapshots).

    Intended usage is therefore:

      1. On the old branch: Set ENABLED=True and FORCE_SNAPSHOT_UPDATE=True and run the tests.
      2. On the new branch: Set ENABLED=True and FORCE_SNAPSHOT_UPDATE=False and run the tests.
      3. Inspect determinism_tests.log
    """

    ENABLED = True
    """
    whether determinism tests are enabled.
    """
    FORCE_SNAPSHOT_UPDATE = False
    """
    whether to force the update/creation of snapshots for every test.
    Enable this when running on the "old" branch and you want to prepare the snapshots
    for a comparison with the "new" branch.
    """
    PASS_IF_CORE_MESSAGES_UNCHANGED = True
    """
    whether to pass the test if only the core messages are unchanged.
    If this is False, then the full log is required to be equivalent, whereas if it is True,
    only the core messages need to be equivalent.
    The core messages test whether the algorithm produces the same network parameters.
    """

    def __init__(
        self,
        name: str,
        main_fn: Callable[[Namespace], Any],
        args: Namespace,
        is_offline: bool = False,
        ignored_messages: Sequence[str] = (),
    ):
        """
        :param name: the (unique!) name of the test
        :param main_fn: the function to be called for the test
        :param args: the arguments to be passed to the main function (some of which are overridden
            for the test)
        :param is_offline: whether the algorithm being tested is an offline algorithm and therefore
            does not configure the number of training environments (`training_num`)
        :param ignored_messages: message fragments to ignore in the trace log (if any)
        """
        self.determinism_test = TraceDeterminismTest(
            base_path=Path(__file__).parent / "resources" / "determinism",
            log_filename="determinism_tests.log",
            core_messages=["Params"],
            ignored_messages=ignored_messages,
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
        if not is_offline:
            set("training_num", 1)
        set("test_num", 1)

        self.args = args
        self.main_fn = main_fn

    def run(self, update_snapshot: bool = False) -> None:
        """
        :param update_snapshot: whether to update to snapshot (may be centrally overridden by
            FORCE_SNAPSHOT_UPDATE)
        """
        if not self.ENABLED:
            pytest.skip("Algorithm determinism tests are disabled.")

        if self.FORCE_SNAPSHOT_UPDATE:
            update_snapshot = True

        # run the actual process
        with TraceLoggerContext() as trace:
            with TorchDeterministicModeContext():
                self.main_fn(self.args)
            log = trace.get_log()

        self.determinism_test.check(
            log,
            self.name,
            create_reference_result=update_snapshot,
            pass_if_core_messages_unchanged=self.PASS_IF_CORE_MESSAGES_UNCHANGED,
        )
