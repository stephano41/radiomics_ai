import logging
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from autorad.inference.infer_utils import get_last_run_from_experiment_name
from autorad.utils import io
from src.analysis import get_shap_values, plot_shap_bar, summate_shap_bar, plot_dependence_scatter_plot, \
    plot_correlation_graph, plot_calibration_curve, plot_net_benefit

from src.analysis.shap import get_top_shap_feature_names
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

    os.mkdir(os.path.join(output_dir, 'feature_analysis'))
    shap_values_file = os.path.join(output_dir, 'feature_analysis/shap_values_datas.pkl')
    if os.path.exists(shap_values_file):
        # If the file exists, load shap_values from it
        with open(shap_values_file, 'rb') as f:
            shap_values, shap_datas = pickle.load(f)
    else:
        # If the file doesn't exist, generate shap_values using get_shap_values
        shap_values, shap_datas = get_shap_values(run)

        # Save shap_values to the output directory
        with open(shap_values_file, 'wb') as f:
            pickle.dump((shap_values, shap_datas), f)

    plot_shap_bar(shap_values, max_display=12,
                  save_dir=os.path.join(output_dir, 'feature_analysis/shap_bar_plot_overview.png'))

    plot_dependence_scatter_plot(shap_values, shap_datas, n_features=12, save_dir=output_dir, plots_per_row=3)

    if config.analysis.get('image_modalities', None) is not None:
        summate_shap_bar(shap_values, feature_substrings=config.analysis.image_modalities,
                         save_dir=os.path.join(output_dir, 'feature_analysis/shap_bar_image_modalities.png'))
    summate_shap_bar(shap_values, feature_substrings=config.analysis.feature_classes,
                     save_dir=os.path.join(output_dir, 'feature_analysis/shap_bar_feature_classes.png'))

    top_features = get_top_shap_feature_names(shap_values, ascending=False, n_features=12)

    io.save_yaml(top_features, os.path.join(output_dir, 'feature_analysis/selected_features.yml'))

    plot_correlation_graph(run, feature_names=top_features, plots_per_row=3,
                           save_dir=os.path.join(output_dir, 'feature_analysis/feature_correlation_plot.png'),
                           x_axis_labels=config.labels)

    if 'bootstrap_scores.pkl' in os.listdir(output_dir):
        if config.multi_class == 'raise':
            # only do this if binary cases
            plot_calibration_curve(run, save_dir=os.path.join(output_dir, 'calibration_curve.png'))

        plot_net_benefit(run, save_dir=os.path.join(output_dir, 'decision_curve.png'), estimator_name='model')
    else:
        logger.warning('No bootstrap scores found, not running analysis dependent on it')
