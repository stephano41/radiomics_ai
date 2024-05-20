import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc as auc_metric
import logging

logger = logging.getLogger(__name__)


def plot_roc_curve_with_ci(data_dict, save_dir=None):
    """
    Plot ROC curve with visualized confidence interval.

    Parameters:
    data_dict (dict): A list containing dictionaries which have keys 'fpr', 'tpr' and 'thresholds'
    thresholds is used to calculate youden index for optimal threshold

    Returns:
    None
    """
    processed_fpr_tpr = {'fpr': [], 'tpr': [], 'thresholds':[]}
    for d in data_dict:
        processed_fpr_tpr['fpr'].append(d['fpr'])
        processed_fpr_tpr['tpr'].append(d['tpr'])
        processed_fpr_tpr['thresholds'].append(d['thresholds'])

    fpr_list = processed_fpr_tpr['fpr']
    tpr_list = processed_fpr_tpr['tpr']
    threshold_list = processed_fpr_tpr['thresholds']
    auc_values = [auc_metric(fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)]

    # Calculate the mean AUC and standard deviation
    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)

    # Create an array to hold the mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    mean_threshold = np.zeros_like(mean_fpr)

    # Create arrays to store lower and upper bounds of ROC curves
    tprs_upper = []
    tprs_lower = []

    # Calculate ROC curve for each fold and store upper and lower bounds
    for fpr, tpr, thresholds in zip(fpr_list, tpr_list, threshold_list):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        mean_tpr += interp_tpr

        interp_thresholds = np.interp(mean_fpr, fpr, thresholds)
        interp_thresholds[0] = 0.0
        mean_threshold += interp_thresholds

        tprs_upper.append(interp_tpr + 2 * std_auc)
        tprs_lower.append(interp_tpr - 2 * std_auc)

    # Calculate the mean ROC curve and confidence interval bounds
    mean_tpr /= len(fpr_list)
    mean_threshold /= len(fpr_list)
    idx = np.argmax(mean_tpr - mean_fpr)

    logger.info(f"Optimal cutoff point at {mean_threshold[idx]} with sensitivity {mean_tpr[idx]} and specificity {1-mean_fpr[idx]}")

    tprs_upper = np.array(tprs_upper).T
    tprs_lower = np.array(tprs_lower).T

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='b', marker='.', label=f'Mean ROC (AUC = {mean_auc:.2f})')
    plt.scatter(mean_fpr[idx], mean_tpr[idx],marker='o', color='orange', label='Best', zorder=10)
    plt.fill_between(mean_fpr, tprs_lower.mean(axis=1), tprs_upper.mean(axis=1), color='grey', alpha=0.3,
                     label='Confidence Interval')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2, label='Random')
    plt.axis('scaled')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Confidence Interval')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    current_fig = plt.gcf()
    if save_dir is not None:
        current_fig.savefig(save_dir, dpi=1200, bbox_inches='tight')

    return current_fig, {'optimal_threshold':mean_threshold[idx], 'optimal_sensitivity':mean_tpr[idx], 'optimal_specificity': 1-mean_fpr[idx]}
