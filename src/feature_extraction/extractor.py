from autorad.feature_extraction import FeatureExtractor as OrigiFeatureExtractor
import mlflow

from autorad.utils import io, mlflow_utils
from autorad.config import config


class FeatureExtractor(OrigiFeatureExtractor):
    def save_config(self, mask_label):
        extraction_param_dict = io.load_yaml(self.extraction_params)
        if mask_label is not None:
            extraction_param_dict["label"] = mask_label
        run_config = {
            "feature_set": self.feature_set,
            "extraction_params": extraction_param_dict,
        }
        # got rid of 'file://' at front of path to avoid bug
        mlflow.set_tracking_uri('file:/' + config.MODEL_REGISTRY)
        mlflow.set_experiment("feature_extraction")
        with mlflow.start_run() as run:
            mlflow_utils.log_dict_as_artifact(run_config, "extraction_config")

        return run.info.run_id

