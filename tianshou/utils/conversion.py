from typing import overload

import torch


@overload
def to_optional_float(x: torch.Tensor) -> float:
    ...


@overload
def to_optional_float(x: float) -> float:
    ...


@overload
def to_optional_float(x: None) -> None:
    ...


def to_optional_float(x: torch.Tensor | float | None) -> float | None:
    """For the common case where one needs to extract a float from a scalar Tensor, which may be None."""
    if isinstance(x, torch.Tensor):
        return x.item()
    return x
