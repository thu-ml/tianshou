from typing import Any


def set_numerical_fields_to_precision(data: dict[str, Any], precision: int = 3) -> dict[str, Any]:
    """Returns a copy of the given dictionary with all numerical values rounded to the given precision.

    Note: does not recurse into nested dictionaries.

    :param data: a dictionary
    :param precision: the precision to be used
    """
    result = {}
    for k, v in data.items():
        if isinstance(v, float):
            v = round(v, precision)
        result[k] = v
    return result
