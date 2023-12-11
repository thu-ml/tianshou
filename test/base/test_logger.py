import numpy as np
import pytest

from tianshou.utils import BaseLogger


class TestBaseLogger:
    @staticmethod
    @pytest.mark.parametrize(
        "input_dict, expected_output",
        [
            ({"a": 1, "b": {"c": 2, "d": {"e": 3}}}, {"a": 1, "b/c": 2, "b/d/e": 3}),
            ({"a": {"b": {"c": 1}}}, {"a/b/c": 1}),
        ],
    )
    def test_flatten_dict_basic(input_dict, expected_output):
        result = BaseLogger.prepare_dict_for_logging(input_dict)
        assert result == expected_output

    @staticmethod
    @pytest.mark.parametrize(
        "input_dict, delimiter, expected_output",
        [
            ({"a": {"b": {"c": 1}}}, "|", {"a|b|c": 1}),
            ({"a": {"b": {"c": 1}}}, ".", {"a.b.c": 1}),
        ],
    )
    def test_flatten_dict_custom_delimiter(input_dict, delimiter, expected_output):
        result = BaseLogger.prepare_dict_for_logging(input_dict, delimiter=delimiter)
        assert result == expected_output

    @staticmethod
    @pytest.mark.parametrize(
        "input_dict, exclude_arrays, expected_output",
        [
            (
                {"a": np.array([1, 2, 3]), "b": {"c": np.array([4, 5, 6])}},
                False,
                {"a": np.array([1, 2, 3]), "b/c": np.array([4, 5, 6])},
            ),
            ({"a": np.array([1, 2, 3]), "b": {"c": np.array([4, 5, 6])}}, True, {}),
        ],
    )
    def test_flatten_dict_exclude_arrays(input_dict, exclude_arrays, expected_output):
        result = BaseLogger.prepare_dict_for_logging(input_dict, exclude_arrays=exclude_arrays)
        assert result.keys() == expected_output.keys()
        for val1, val2 in zip(result.values(), expected_output.values(), strict=True):
            assert np.all(val1 == val2)

    @staticmethod
    @pytest.mark.parametrize(
        "input_dict, expected_output",
        [
            ({"a": (1,), "b": {"c": "2", "d": {"e": 3}}}, {"b/d/e": 3}),
        ],
    )
    def test_flatten_dict_invalid_values_filtered_out(input_dict, expected_output):
        result = BaseLogger.prepare_dict_for_logging(input_dict)
        assert result == expected_output
