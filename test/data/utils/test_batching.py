import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset, random_split

from tianshou.data.utils.batching import BatchDataLoader, get_batch_boundaries


class TestGetBatchBoundaries:
    def test_drop_last(self):
        result = get_batch_boundaries(5, 22, "drop")
        expected = np.array([0, 5, 10, 15, 20])
        np.testing.assert_array_equal(result, expected)

    def test_merge_last(self):
        result = get_batch_boundaries(5, 22, "merge")
        expected = np.array([0, 5, 10, 15, 22])
        np.testing.assert_array_equal(result, expected)

    def test_keep_last(self):
        result = get_batch_boundaries(5, 22, "keep")
        expected = np.array([0, 5, 10, 15, 20, 22])
        np.testing.assert_array_equal(result, expected)

    def test_len_data_less_than_batch_size(self):
        result = get_batch_boundaries(5, 2, "keep")
        expected = np.array([0, 2])
        np.testing.assert_array_equal(result, expected)


class TestBatchDataLoader:
    def test_iter_keep_last(self):
        loader = BatchDataLoader(range(1, 9), batch_size=3, last_batch="keep")
        batches = list(loader)
        assert batches[0].tolist() == [1, 2, 3]
        assert batches[1].tolist() == [4, 5, 6]
        assert batches[2].tolist() == [7, 8]

    def test_iter_merge_last(self):
        loader = BatchDataLoader(range(1, 9), batch_size=3, last_batch="merge")
        batches = list(loader)
        assert batches[-1].tolist() == [4, 5, 6, 7, 8]

    def test_iter_drop_last(self):
        loader = BatchDataLoader(range(1, 9), batch_size=3, last_batch="drop")
        batches = list(loader)
        assert batches[-1].tolist() == [4, 5, 6]

    @pytest.mark.parametrize("array_type", [np.array, torch.tensor])
    def test_with_different_array_types(self, array_type):
        data_as_array = array_type(range(1, 9))
        loader = BatchDataLoader(data_as_array, batch_size=3)
        batches = list(loader)
        assert (batches[0] == array_type([1, 2, 3])).all()
        assert (batches[1] == array_type([4, 5, 6, 7, 8])).all()

    def test_with_tensor_dataset_and_subset(self):
        ds = TensorDataset(torch.arange(8), torch.arange(10, 18))
        loader = BatchDataLoader(ds, batch_size=3)
        batches = list(loader)
        assert (batches[0][0] == torch.tensor([0, 1, 2])).all()
        assert (batches[0][1] == torch.tensor([10, 11, 12])).all()
        assert (batches[1][0] == torch.tensor([3, 4, 5, 6, 7])).all()

        some_subset = random_split(ds, [3, 5])[0]
        subset_loader = BatchDataLoader(some_subset, batch_size=3, shuffle=True)
        # Just testing that it doesn't crash
        list(subset_loader)

    def test_shuffling(self):
        larger_data = range(100)
        loader = BatchDataLoader(larger_data, batch_size=3, shuffle=True)
        batches_1 = list(loader)
        batches_2 = list(loader)
        # It's very unlikely that the same batch is sampled twice
        assert (batches_1[0] != batches_2[0]).any()

        data_as_set = set(larger_data)
        assert {i for batch in batches_1 for i in batch} == data_as_set
        assert {i for batch in batches_2 for i in batch} == data_as_set
        
    def test_len(self):
        assert len(BatchDataLoader(range(10), batch_size=10)) == 1
        assert len(BatchDataLoader(range(10), batch_size=3, last_batch="merge")) == 3
        assert len(BatchDataLoader(range(10), batch_size=3, last_batch="drop")) == 3
        assert len(BatchDataLoader(range(10), batch_size=3, last_batch="keep")) == 4
