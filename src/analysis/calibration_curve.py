import pickle
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
import os
from .hosmer_lemeshow import hosmer_lemeshow
import logging

logger = logging.getLogger(__name__)


def plot_calibration_curve(run, save_dir=None):
    if isinstance(run, str):
        run = pd.Series(dict(mlflow.get_run(run).info))

    artifact_uri = run.artifact_uri.removeprefix('file://')

    with open(os.path.join(artifact_uri, 'bootstrap_scores.pkl'), 'rb') as f:
        bootstrap_scores = pickle.load(f)

    y_true_y_preds = [score[2] for score in bootstrap_scores]

    mean_prob_pred = np.linspace(0, 1, 100)
    mean_prob_true = np.zeros_like(mean_prob_pred)

    for y_true, y_preds in y_true_y_preds:
        prob_true, prob_pred = calibration_curve(y_true, y_preds)
        interp_prob_true = np.interp(mean_prob_pred, prob_true, prob_pred)
        interp_prob_true[0] = 0.0
        mean_prob_true += interp_prob_true

    mean_prob_true /= len(y_true_y_preds)

    logger.info(hosmer_lemeshow(mean_prob_pred, mean_prob_true))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", label='Perfectly calibrated')
    ax.plot(mean_prob_pred, mean_prob_true, label='Mean Calibration Curve')
    ax.legend(loc="lower right")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True)

    if save_dir is None:
        plt.show()
    else:
        fig.savefig(save_dir, format='eps')

    return fig, ax

