"""Helper functions for persistence/pickling, which have been copied from sensAI (specifically `sensai.util.pickle`)."""

from collections.abc import Iterable
from copy import copy
from typing import Any


def setstate(
    cls: type,
    obj: Any,
    state: dict[str, Any],
    renamed_properties: dict[str, str] | None = None,
    new_optional_properties: list[str] | None = None,
    new_default_properties: dict[str, Any] | None = None,
    removed_properties: list[str] | None = None,
) -> None:
    """Helper function for safe implementations of `__setstate__` in classes, which appropriately handles the cases where
    a parent class already implements `__setstate__` and where it does not. Call this function whenever you would actually
    like to call the super-class' implementation.
    Unfortunately, `__setstate__` is not implemented in `object`, rendering `super().__setstate__(state)` invalid in the general case.

    :param cls: the class in which you are implementing `__setstate__`
    :param obj: the instance of `cls`
    :param state: the state dictionary
    :param renamed_properties: a mapping from old property names to new property names
    :param new_optional_properties: a list of names of new property names, which, if not present, shall be initialized with None
    :param new_default_properties: a dictionary mapping property names to their default values, which shall be added if they are not present
    :param removed_properties: a list of names of properties that are no longer being used
    """
    # handle new/changed properties
    if renamed_properties is not None:
        for mOld, mNew in renamed_properties.items():
            if mOld in state:
                state[mNew] = state[mOld]
                del state[mOld]
    if new_optional_properties is not None:
        for mNew in new_optional_properties:
            if mNew not in state:
                state[mNew] = None
    if new_default_properties is not None:
        for mNew, mValue in new_default_properties.items():
            if mNew not in state:
                state[mNew] = mValue
    if removed_properties is not None:
        for p in removed_properties:
            if p in state:
                del state[p]
    # call super implementation, if any
    s = super(cls, obj)
    if hasattr(s, "__setstate__"):
        s.__setstate__(state)
    else:
        obj.__dict__ = state


def getstate(
    cls: type,
    obj: Any,
    transient_properties: Iterable[str] | None = None,
    excluded_properties: Iterable[str] | None = None,
    override_properties: dict[str, Any] | None = None,
    excluded_default_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper function for safe implementations of `__getstate__` in classes, which appropriately handles the cases where
    a parent class already implements `__getstate__` and where it does not. Call this function whenever you would actually
    like to call the super-class' implementation.
    Unfortunately, `__getstate__` is not implemented in `object`, rendering `super().__getstate__()` invalid in the general case.

    :param cls: the class in which you are implementing `__getstate__`
    :param obj: the instance of `cls`
    :param transient_properties: transient properties which shall be set to None in serializations
    :param excluded_properties: properties which shall be completely removed from serializations
    :param override_properties: a mapping from property names to values specifying (new or existing) properties which are to be set;
        use this to set a fixed value for an existing property or to add a completely new property
    :param excluded_default_properties: properties which shall be completely removed from serializations, if they are set
        to the given default value
    :return: the state dictionary, which may be modified by the receiver
    """
    s = super(cls, obj)
    d = s.__getstate__() if hasattr(s, "__getstate__") else obj.__dict__
    d = copy(d)
    if transient_properties is not None:
        for p in transient_properties:
            if p in d:
                d[p] = None
    if excluded_properties is not None:
        for p in excluded_properties:
            if p in d:
                del d[p]
    if override_properties is not None:
        for k, v in override_properties.items():
            d[k] = v
    if excluded_default_properties is not None:
        for p, v in excluded_default_properties.items():
            if p in d and d[p] == v:
                del d[p]
    return d
