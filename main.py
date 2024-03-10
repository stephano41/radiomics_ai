import hydra
from hydra.utils import instantiate
import logging

logger = logging.getLogger(__name__)

@hydra.main(config_path='conf', config_name='main', version_base='1.3')
def main(config):
    if config.get('notes',None) is not None:
        logger.info(config.notes)
    instantiate(config.pipeline, config)


if __name__ == '__main__':
    main()
