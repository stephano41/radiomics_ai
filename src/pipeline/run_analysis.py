import logging
import mlflow
import pandas as pd
from autorad.inference.infer_utils import get_last_run_from_experiment_name, load_dataset_artifacts
from src.analysis.shap import get_shap_values, plot_shap_bar, summate_shap_bar, plot_dependence_scatter_plot
from src.analysis.calibration_curve import plot_calibration_curve
import os

logger = logging.getLogger(__name__)


def run_analysis(config):
    if config.get('run_id', None) is not None:
        # convert this to the same format
        run = pd.Series(dict(mlflow.get_run(config.run_id).info))
    else:
        logger.info(f'no run speicified in config, getting the last run from {config.name} instead')
        run = get_last_run_from_experiment_name(config.name)

    logger.info(f'analysing {run.run_id}')
    output_dir = run.artifact_uri.removeprefix('file://')

    if config.analysis.get('compare_run_id', None) is not None:
        compare_run = pd.Series(dict(mlflow.get_run(config.compare_run_id).info))
        dataset_artifacts = load_dataset_artifacts(compare_run)
        shap_values, _, _ = get_shap_values(run, dataset_artifacts['df'], dataset_artifacts['splits'])
    else:
        shap_values, _, _ = get_shap_values(run)

    plot_shap_bar(shap_values, max_display=200,
                  save_dir=os.path.join(output_dir, 'shap_bar_plot_overview.png'))

    dependence_save_dir=os.path.join(output_dir, 'shap_dependence_scatter_plots')
    os.makedirs(dependence_save_dir, exist_ok=True)
    plot_dependence_scatter_plot(shap_values, 10, save_dir=dependence_save_dir)

    if config.analysis.get('image_modalities', None) is not None:
        summate_shap_bar(shap_values, config.analysis.image_modalities,
                         save_dir=os.path.join(output_dir, 'shap_bar_image_modalities.png'))
    summate_shap_bar(shap_values, config.analysis.feature_classes,
                     save_dir=os.path.join(output_dir, 'shap_bar_feature_classe.png'))
    if config.multi_class == 'raise':
        # only do this if binary cases
        plot_calibration_curve(run, save_dir=os.path.join(output_dir, 'calibration_curve.png'))
