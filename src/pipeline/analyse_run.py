import logging
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from autorad.inference.infer_utils import get_last_run_from_experiment_name, load_dataset_artifacts

from src.analysis import get_shap_values, plot_shap_bar, summate_shap_bar, plot_dependence_scatter_plot, \
    plot_correlation_graph, plot_calibration_curve, plot_net_benefit
from src.utils.inference import get_run_info_as_series

logger = logging.getLogger(__name__)


def analyse_run(config):
    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams.update({'font.size': 10})

    if config.get('run_id', None) is not None:
        # convert this to the same format
        run = get_run_info_as_series(config.run_id)
    else:
        logger.info(f'no run specified in config, getting the last run from {config.name} instead')
        run = get_last_run_from_experiment_name(config.name)

    logger.info(f'analysing {run.run_id}')
    output_dir = run.artifact_uri.removeprefix('file://')

    if config.analysis.get('compare_run_id', None) is not None:
        compare_run = get_run_info_as_series(config.compare_run_id)
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

    pd.DataFrame(shap_values.values, columns=shap_values.feature_names).to_csv(
        os.path.join(output_dir, 'shap_values.csv'))

    plot_shap_bar(shap_values, max_display=200,
                  save_dir=os.path.join(output_dir, 'shap_bar_plot_overview.png'))

    plot_dependence_scatter_plot(shap_values, 12, save_dir=output_dir, plots_per_row=3)

    if config.analysis.get('image_modalities', None) is not None:
        summate_shap_bar(shap_values, config.analysis.image_modalities,
                         save_dir=os.path.join(output_dir, 'shap_bar_image_modalities.png'))
    summate_shap_bar(shap_values, config.analysis.feature_classes,
                     save_dir=os.path.join(output_dir, 'shap_bar_feature_classes.png'))

    plot_correlation_graph(run, feature_names=shap_values.feature_names, plots_per_row=3,
                           save_dir=os.path.join(output_dir, 'feature_correlation_plot.png'),
                           x_axis_labels=config.labels)

    if 'bootstrap_scores.pkl' in os.listdir(output_dir):
        if config.multi_class == 'raise':
            # only do this if binary cases
            plot_calibration_curve(run, save_dir=os.path.join(output_dir, 'calibration_curve.png'))

        plot_net_benefit(run, save_dir=os.path.join(output_dir, 'decision_curve.png'), estimator_name='model')
    else:
        logger.warning('No bootstrap scores found, not running analysis dependent on it')
