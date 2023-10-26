import os
from typing import List

import pytest
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

CONFIG_NAME = "main"
WORKING_DIRECTORY='/opt/project/'


@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch):

    monkeypatch.chdir(WORKING_DIRECTORY)
    os.chdir(WORKING_DIRECTORY)


@pytest.fixture(scope="function")
def cfg_tune(request, tmp_path):
    return _get_conf(CONFIG_NAME, tmp_path, request.param)

    # GlobalHydra.instance().clear()


def _get_conf(config_name, tmp_path, overrides):
    config_path = os.path.relpath(os.path.join(WORKING_DIRECTORY, "conf"), os.path.dirname(os.path.realpath(__file__)))
    if not isinstance(overrides, List):
        overrides = [overrides]
    with initialize(version_base='1.3', config_path=config_path):

        config = compose(config_name=config_name, return_hydra_config=True, overrides=[f"output_root={tmp_path}"]+overrides)

        return config


@pytest.fixture(scope="function")
def get_dummy_autoencoder(tmp_path):
    config = _get_conf(CONFIG_NAME, tmp_path, "+autoencoder=dummy_vae")
    return instantiate(config.preprocessing.autoencoder, _convert_='object')


def test_conf_build(cfg_tune):
    HydraConfig().set_config(cfg_tune)
