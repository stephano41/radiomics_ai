from hydra.utils import instantiate
from pytest import mark
from hydra.core.hydra_config import HydraConfig

@mark.slow
@mark.parametrize('cfg_tune', [['experiments=meningioma_autoencoder', 'autoencoder=dummy_vae', 'bootstrap.iters=5',
                                'optimizer.n_trials=5']], indirect=True)
def test_dryrun(cfg_tune):
    HydraConfig().set_config(cfg_tune)
    instantiate(cfg_tune.pipeline, cfg_tune)
