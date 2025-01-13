import os.path
import pytest

from torch.utils.data import Dataset

from m2.data import corrupt_mnist

@pytest.mark.skipif(not os.path.exists(os.path.join("data", "processed")), reason="No processed data found.")
def test_my_dataset() -> None:
    """Test the MyDataset class."""
    train_set, test_set = corrupt_mnist(os.path.join("data", "processed"))
    
    # Test correct data type
    assert isinstance(train_set, Dataset), "train_set is not a torch Dataset."
    assert isinstance(test_set, Dataset), "test_set is not a torch Dataset."
    
    # Test correct length
    assert len(train_set) == 30000, "train_set has incorrect length."
    assert len(test_set) == 5000, "test_set has incorrect length."
    
    # Test correct shape of data and correct target values
    for dataset in [train_set, test_set]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "Data has incorrect shape."
            assert y in range(10), "Target value is out of range."
    

def test_my_dataset_no_data():
    """Test the MyDataset class when no data is available."""
    with pytest.raises(FileNotFoundError):
        corrupt_mnist(os.path.join("data", "processed_fake"))
