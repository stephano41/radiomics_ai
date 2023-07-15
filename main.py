import os

# os.environ['AUTORAD_RESULT_DIR'] = os.path.join(os.getcwd(), 'outputs/mlflow')
os.environ['AUTORAD_RESULT_DIR'] = '/app/app/outputs/'

print(f"0. {os.getenv('AUTORAD_RESULT_DIR')}")
import hydra
from hydra.utils import instantiate


os.environ["HYDRA_FULL_ERROR"] = '1'


@hydra.main(config_path='conf', config_name='main', version_base='1.3')
def main(config):
    print(f"1. {os.getenv('AUTORAD_RESULT_DIR')}")
    print(f"1. {os.getenv('HYDRA_FULL_ERROR')}")

    instantiate(config.pipeline, config)


if __name__ == '__main__':
    main()
