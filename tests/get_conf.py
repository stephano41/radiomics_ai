import pytest
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

CONFIG_NAME = "main"


@pytest.fixture(scope="function")
def cfg_main(tmp_path):
    with initialize(version_base='1.3', config_path="../conf"):
        config = compose(config_name=CONFIG_NAME, return_hydra_config=True, overrides=[f"output_root={tmp_path}"])

        yield config

    GlobalHydra.instance().clear()


def test_conf_build(cfg_tune):
    HydraConfig().set_config(cfg_tune)
