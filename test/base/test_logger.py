from tianshou.utils.logger.base import BaseLogger


def depth(d):
    """Compute the depth of a nested dictionary."""
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0


def test_flatten_dict():
    # test case 1
    dictionary = {"a": 1, "b": {"c": 2, "d": 3}}
    expected = {"a": 1, "b/c": 2, "b/d": 3}
    flattened_dict = BaseLogger._flatten_dict(dictionary)
    assert flattened_dict == expected
    assert depth(flattened_dict) == 1
    # test case 2
    dictionary = {"a": 1, "b": {"c": 2, "d": 3}, "e": {"f": {"g": 4}}}
    expected = {"a": 1, "b/c": 2, "b/d": 3, "e/f/g": 4}
    flattened_dict = BaseLogger._flatten_dict(dictionary)
    assert flattened_dict == expected
    assert depth(flattened_dict) == 1


if __name__ == "__main__":
    test_flatten_dict()
