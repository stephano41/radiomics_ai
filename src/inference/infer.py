from src.utils.dataset import get_multi_paths_with_separate_folder_per_case
from src.utils.pipeline import extract_features
from autorad.utils.extraction_utils import filter_pyradiomics_names
from autorad.inference.infer_utils import get_pipeline_from_run, get_run_info_as_series
from autorad.config.type_definitions import PathLike
from autorad.utils import io
from collections.abc import Mapping
import os


class Inferrer:
    def __init__(self, run, image_stems=("image",), mask_stem="segmentation", extraction_config: PathLike | Mapping=None, n_jobs=1) -> None:
        self.run=run
        if isinstance(run, str):
            self.run = get_run_info_as_series(run)
        self.image_stems=image_stems,
        self.mask_stem=mask_stem,
        self.extraction_config=extraction_config
        self.n_jobs=n_jobs

    def extract_features(self, directory):
        paths_df = get_multi_paths_with_separate_folder_per_case(directory, image_stems=self.image_stems, mask_stem=self.mask_stem, relative=True)
        features_df = extract_features(image_stems=self.image_stems, 
                                       paths_df=paths_df, 
                                       data_dir=directory, 
                                       extraction_params=self.extraction_config, 
                                       n_jobs=self.n_jobs)
        # TODO multi image modality featuredataset
        radiomics_features = filter_pyradiomics_names(list(features_df.columns))

        return features_df[radiomics_features+self.get_additional_feature_columns()]

    def get_additional_feature_columns(self):
        artifact_uri = self.run.artifact_uri.removeprefix('file://')
        dataset_config = io.load_yaml(os.path.join(artifact_uri, 'feature_dataset/dataset_config.yaml'))

        return dataset_config.get('additional_features',[])
    
    def predict_proba(self, directory):
        feature_df = self.extract_features(directory)
        pipeline = get_pipeline_from_run(self.run)
        return pipeline.predict_proba(feature_df)
    
    def predict(self, directory):
        feature_df = self.extract_features(directory)
        pipeline = get_pipeline_from_run(self.run)
        return pipeline.predict(feature_df)