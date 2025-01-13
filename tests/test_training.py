import os.path
import pytest

from hydra import initialize, compose
from src.m2.train import train

@pytest.mark.skipif(not os.path.exists(os.path.join("data", "processed")), reason="No processed data found.")
def test_train():
    with initialize(config_path=os.path.join(os.path.pardir, "configs"), job_name="test_app"):
        cfg = compose(config_name="default_config", overrides=["train=hpar1"])
        stats = train(cfg)
        
    accuracy = stats['train_accuracy']
    
    for acc in accuracy:
        assert acc >= 0.0, "Accuracy is less than 0."
        assert acc <= 1.0, "Accuracy is greater than 1."