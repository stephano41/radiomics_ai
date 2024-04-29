import logging
import mlflow
import pandas as pd
from autorad.inference.infer_utils import get_last_run_from_experiment_name, load_dataset_artifacts
from src.analysis.shap import get_shap_values, plot_shap_bar, summate_shap_bar, plot_dependence_scatter_plot
from src.analysis.calibration_curve import plot_calibration_curve
from src.analysis.correlation_plot import plot_correlation_graph
from src.analysis.decision_curve import plot_net_benefit
# from src.evaluation.roc_curve import plot_roc_curve_with_ci
import os
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pickle

logger = logging.getLogger(__name__)


def run_analysis(config):
    # rcParams['font.family'] = 'Arial'
    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams.update({'font.size': 10})

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
        shap_values_file = os.path.join(output_dir, 'shap_values.pkl')
        if os.path.exists(shap_values_file):
            # If the file exists, load shap_values from it
            with open(shap_values_file, 'rb') as f:
                shap_values = pickle.load(f)
        else:
            # If the file doesn't exist, generate shap_values using get_shap_values
            shap_values, _, _ = get_shap_values(run)
            
            # Save shap_values to the output directory
            with open(shap_values_file, 'wb') as f:
                pickle.dump(shap_values, f)

    plot_shap_bar(shap_values, max_display=200,
                  save_dir=os.path.join(output_dir, 'shap_bar_plot_overview.png'))

    plot_dependence_scatter_plot(shap_values, 10, save_dir=output_dir, plots_per_row=3)

    if config.analysis.get('image_modalities', None) is not None:
        summate_shap_bar(shap_values, config.analysis.image_modalities,
                         save_dir=os.path.join(output_dir, 'shap_bar_image_modalities.png'))
    summate_shap_bar(shap_values, config.analysis.feature_classes,
                     save_dir=os.path.join(output_dir, 'shap_bar_feature_classe.png'))
  
    plot_correlation_graph(run, feature_names=shap_values.feature_names, plots_per_row=3, save_dir=os.path.join(output_dir, 'feature_correlation_plot.png'), x_axis_labels=config.labels)

    if 'bootstrap_scores.pkl' in os.listdir(output_dir):
        if config.multi_class == 'raise':
            # only do this if binary cases
            plot_calibration_curve(run, save_dir=os.path.join(output_dir, 'calibration_curve.png'))

        plot_net_benefit(run, save_dir=os.path.join(output_dir, 'decision_curve.png'), estimator_name='model')

        # if config.get('plot_roc_auc') is not None:
        #     plot_roc_curve_with_ci(run, os.path.join(output_dir,'roc_curve.png'))
    else:
        logger.warn('No bootstrap scores found, not running analysis dependent on it')
