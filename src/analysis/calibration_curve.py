import pickle
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
import os
from scipy.stats import chi2


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

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
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
        fig.savefig(save_dir, dpi=300)

    return fig, ax


def hosmer_lemeshow(y_prob, y_true):
    # pihat=model.predict()
    pihatcat=pd.cut(y_prob, np.percentile(y_prob,[0,25,50,75,100]),labels=False,include_lowest=True) #here we've chosen only 4 groups


    meanprobs =[0]*4 
    expevents =[0]*4
    obsevents =[0]*4 
    meanprobs2=[0]*4 
    expevents2=[0]*4
    obsevents2=[0]*4 

    for i in range(4):
       meanprobs[i]=np.mean(y_prob[pihatcat==i])
       expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
       obsevents[i]=np.sum(Y[pihatcat==i])
       meanprobs2[i]=np.mean(1-y_prob[pihatcat==i])
       expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
       obsevents2[i]=np.sum(1-y_true[pihatcat==i]) 


    data1={'meanprobs':meanprobs,'meanprobs2':meanprobs2}
    data2={'expevents':expevents,'expevents2':expevents2}
    data3={'obsevents':obsevents,'obsevents2':obsevents2}
    m=pd.DataFrame(data1)
    e=pd.DataFrame(data2)
    o=pd.DataFrame(data3)
    
    # The statistic for the test, which follows, under the null hypothesis,
    # The chi-squared distribution with degrees of freedom equal to amount of groups - 2. Thus 4 - 2 = 2
    tt=sum(sum((np.array(o)-np.array(e))**2/np.array(e))) 
    pvalue=1-chi2.cdf(tt,2)

    return pd.DataFrame([[chi2.cdf(tt,2).round(2), pvalue.round(2)]],columns = ["Chi2", "p - value"])
