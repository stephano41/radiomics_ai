import logging

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

log = logging.getLogger(__name__)


def negative_predictive_value(y_true, y_pred, average='raise', labels=None):
    # Check if it's a binary classification case
    if labels is None:
        binary = len(np.unique(y_true)) <= 2
    else:
        binary = len(labels) <= 2

    if binary:
        # Compute confusion matrix for binary classification
        tn, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn)
    else:
        if average == 'raise':
            raise ValueError("Multiclass detected, but averaging method not defined.")
        # Compute confusion matrix for multiclass classification
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Compute true negatives and negatives per class
        true_negatives = np.diag(cm)
        negatives_per_class = np.sum(cm, axis=0) - true_negatives

        if average == 'macro':
            # Calculate NPV per class
            npv_per_class = true_negatives / negatives_per_class

            # Average NPV across classes
            return np.mean(npv_per_class)
        elif average == 'micro':
            # Compute overall true negatives and negatives
            overall_true_negatives = np.sum(true_negatives)
            overall_negatives = np.sum(negatives_per_class)

            # Calculate micro NPV
            return overall_true_negatives / overall_negatives
        else:
            raise ValueError("Invalid average type. Must be either 'macro' or 'micro'.")


def specificity(y_true, y_pred, average='raise', labels=None):
    # Check if it's a binary classification case
    if labels is None:
        binary = len(np.unique(y_true)) <= 2
    else:
        binary = len(labels) <= 2

    if binary:
        # Compute confusion matrix for binary classification
        tn, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    else:
        if average == 'raise':
            raise ValueError("Multiclass detected, but averaging method not defined.")
        # Compute confusion matrix for multiclass classification
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Compute true negatives and positives per class
        true_negatives = np.diag(cm)
        positives_per_class = np.sum(cm, axis=1) - true_negatives

        if average == 'macro':
            # Calculate specificity per class
            specificity_per_class = true_negatives / (true_negatives + positives_per_class)

            # Average specificity across classes
            return np.mean(specificity_per_class)
        elif average == 'micro':
            # Compute overall true negatives and positives
            overall_true_negatives = np.sum(true_negatives)
            overall_positives = np.sum(positives_per_class)

            # Calculate micro specificity
            return overall_true_negatives / (overall_true_negatives + overall_positives)
        else:
            raise ValueError("Invalid average type. Must be either 'macro' or 'micro'.")


def roc_auc(y_true, y_pred, average='macro', multi_class='raise', labels=None):
    if len(y_pred.shape) >= 2:
        if y_pred.shape[1] <= 2:
            y_pred = y_pred[:, 1]

    try:
        auc = roc_auc_score(y_true, y_pred, average=average, multi_class=multi_class, labels=labels)
    except ValueError as e:
        if 'Only one class present' not in str(e):
            raise ValueError(e)
        log.error("Only one class present in y_true. ROC AUC score is not defined in that case")
        auc = np.nan
    return auc
