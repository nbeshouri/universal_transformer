import os
import wandb
import pytest

from train import Trainer
from universal_transformer.wandb_utils import ConfigWrapper


@pytest.mark.parametrize("config_file", ["universal_test.yaml", "wmt_test.yaml"])
def test_train(config_file):
    os.environ["WANDB_MODE"] = "dryrun"
    os.environ["WANDB_CONFIG_PATHS"] = f"configs/{config_file}"
    run = wandb.init(project="testing", reinit=True, mode="disabled")
    print(run)
    config = ConfigWrapper(wandb.config)
    config.device = "cpu"
    trainer = Trainer(config, run)
    run_history = trainer.train()
    assert run_history["train"][0]["loss"] > run_history["train"][-1]["loss"]
