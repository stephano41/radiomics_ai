import pickle
import numpy as np
from statkit.decision import overall_net_benefit, net_benefit_action, NetBenefitDisplay, net_benefit, net_benefit_oracle
import os
import matplotlib.pyplot as plt
from autorad.inference.infer_utils import get_run_info_as_series



def plot_net_benefit(run, save_dir=None, thresholds=100, estimator_name=None, benefit_type='action', xlim=0.4):
    if isinstance(run, str):
        run = get_run_info_as_series(run)

    artifact_uri = run.artifact_uri.removeprefix('file://')

    with open(os.path.join(artifact_uri, 'bootstrap_scores.pkl'), 'rb') as f:
        bootstrap_scores = pickle.load(f)

    y_true_y_preds = [score[2] for score in bootstrap_scores]

    mean_threshold_prob = np.linspace(0,1,100)
    mean_benefits = np.zeros_like(mean_threshold_prob)
    mean_net_benefit_action = np.zeros_like(mean_threshold_prob)
    mean_net_benefit_noop = np.zeros_like(mean_threshold_prob)

    mean_oracle=[]

    for y_true, y_preds in y_true_y_preds:
        if benefit_type=='action':
            mean_oracle.append(net_benefit_oracle(y_true, action=True))
            thresholds, benefits = net_benefit(y_true, y_preds, thresholds)
            benefit_action = net_benefit_action(y_true, thresholds, action=True)
            benefit_noop = np.zeros_like(benefits)
        elif benefit_type == "noop":
            mean_oracle.append(net_benefit_oracle(y_true, action=False))
            thresholds, benefits = net_benefit(y_true, y_preds, thresholds, action=False)
            benefit_noop = net_benefit_action(y_true, thresholds, action=False)
            benefit_action = np.zeros_like(benefits)

        elif benefit_type == "overall":
            mean_oracle.append(1.0)
            thresholds, benefits = overall_net_benefit(y_true, y_preds, thresholds)
            benefit_action = net_benefit_action(y_true, thresholds, action=True)
            benefit_noop = net_benefit_action(y_true, thresholds, action=False)
        
        interp_benefits = np.interp(mean_threshold_prob, thresholds, benefits)
        mean_benefits += interp_benefits

        interp_net_benefit_action = np.interp(mean_threshold_prob, thresholds, benefit_action)
        mean_net_benefit_action += interp_net_benefit_action

        interp_net_benefit_noop = np.interp(mean_threshold_prob, thresholds, benefit_noop)
        mean_net_benefit_noop += interp_net_benefit_noop

    mean_net_benefit_action /= len(y_true_y_preds)
    mean_net_benefit_noop /= len(y_true_y_preds)
    mean_benefits /= len(y_true_y_preds)
    mean_oracle = np.mean(mean_oracle)

    display = NetBenefitDisplay(threshold_probability=mean_threshold_prob,
                                net_benefit=mean_benefits,
                                net_benefit_action=mean_net_benefit_action,
                                net_benefit_noop=mean_net_benefit_noop,
                                benefit_type=benefit_action,
                                oracle=mean_oracle,
                                estimator_name=estimator_name)
    display.plot()
    display.ax_.set_ylim(-0.05, mean_oracle+0.05)
    display.ax_.set_xlim(0, xlim)
    display.ax_.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    if save_dir is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(save_dir, dpi=1200, bbox_inches='tight')