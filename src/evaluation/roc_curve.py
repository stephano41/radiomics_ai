import numpy as np
from matplotlib import pyplot as plt


def plot_roc_curve_with_ci(data_dict):
    """
    Plot ROC curve with visualized confidence interval.

    Parameters:
    data_dict (dict): A dictionary containing 'auc', 'fpr', and 'tpr' keys, each with a list of values.

    Returns:
    None
    """

    auc_values = data_dict['auc']
    fpr_list = data_dict['fpr']
    tpr_list = data_dict['tpr']

    # Calculate the mean AUC and standard deviation
    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)

    # Create an array to hold the mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    # Create arrays to store lower and upper bounds of ROC curves
    tprs_upper = []
    tprs_lower = []

    # Calculate ROC curve for each fold and store upper and lower bounds
    for fpr, tpr in zip(fpr_list, tpr_list):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        mean_tpr += interp_tpr
        tprs_upper.append(interp_tpr + 2 * std_auc)
        tprs_lower.append(interp_tpr - 2 * std_auc)

    # Calculate the mean ROC curve and confidence interval bounds
    mean_tpr /= len(fpr_list)
    tprs_upper = np.array(tprs_upper).T
    tprs_lower = np.array(tprs_lower).T

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})')
    plt.fill_between(mean_fpr, tprs_lower.mean(axis=1), tprs_upper.mean(axis=1), color='grey', alpha=0.3,
                     label='Confidence Interval')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Confidence Interval')
    plt.legend(loc='lower right')
    plt.grid(True)
    return plt.gcf()
