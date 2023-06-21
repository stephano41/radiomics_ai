import logging
import hydra
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='main', version_base='1.3')
def main(config):
    instantiate(config.pipeline, config)


if __name__ == '__main__':
    main()