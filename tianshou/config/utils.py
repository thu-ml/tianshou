from dataclasses import asdict, is_dataclass


def collect_configs(*confs):
    """
    Collect instances of dataclasses to a single dict mapping the
    classname to the values. If any of the passed objects is not a
    dataclass or if two instances of the same config class are passed,
    an error will be raised.

    :param confs: dataclasses
    :return: Dictionary mapping class names to their instances.
    """
    result = {}

    for conf in confs:
        if not is_dataclass(conf):
            raise ValueError(f"Object {conf.__class__.__name__} is not a dataclass.")

        if conf.__class__.__name__ in result:
            raise ValueError(f"Duplicate instance of {conf.__class__.__name__} found.")

        result[conf.__class__.__name__] = asdict(conf)

    return result
