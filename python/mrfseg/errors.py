"""The exceptions the core raises for its callers to show."""

from __future__ import annotations

import math


class InputError(ValueError):
    """A request that cannot be done as asked. The message is for the user; the server sends it
    as a 400."""


def is_number(value: object) -> bool:
    """A finite int or float: what JSON may send as a number, less bool and the non-finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def is_whole(value: object, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high
