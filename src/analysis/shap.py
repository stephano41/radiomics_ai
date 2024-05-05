from __future__ import annotations
import pandas as pd
import shap
import mlflow
from autorad.models import MLClassifier
from autorad.inference import infer_utils

import matplotlib.pyplot as plt
import numpy as np
import string
from src.utils.inference import get_preprocessed_data


def get_shap_values(run: str | pd.Series):
    if isinstance(run, str):
        run = infer_utils.get_run_info_as_series(run)

    artifact_uri = run.artifact_uri
    model = MLClassifier.load_from_mlflow(f"{artifact_uri}/model")
    preprocessor = mlflow.sklearn.load_model(f"{artifact_uri}/preprocessor")

    preprocessed_data = get_preprocessed_data(run)

    auto_preprocessed = preprocessed_data is not None

    if auto_preprocessed:
        feature_dataset = preprocessed_data
    else:
        dataset_artifacts = infer_utils.load_dataset_artifacts(run)

        feature_dataset = infer_utils.load_feature_dataset(feature_df=dataset_artifacts['df'],
                                                        dataset_config=dataset_artifacts['dataset_config'],
                                                        splits=dataset_artifacts['splits']).data
    shap_values=[]
    shap_data=[]
    for X_train, y_train, _, X_val, y_val, _ in feature_dataset.iter_training():
        _X_train = X_train.copy()
        _y_train = y_train.copy()
        
        if not auto_preprocessed:
            _X_train, _y_train = preprocessor._fit_transform(_X_train, _y_train)
        model.fit(_X_train, _y_train)
        explainer = shap.Explainer(model.predict_proba_binary, _X_train)
        shap_value = explainer(_X_train, max_evals=2 * _X_train.shape[1] + 1)
        shap_value_pd = pd.DataFrame(shap_value.values,columns=shap_value.feature_names)
        shap_values.append(shap_value_pd)

        shap_data.append(pd.DataFrame(shap_value.data, columns=shap_value.feature_names))
    # first preprocess the data, check if fit_resample is the last thing

    return shap_values, shap_data


def get_abs_mean_shap_pd(shap_values):
    shap_values_pd_abs_mean=[]
    for shap_value_pd in shap_values:
        shap_values_pd_abs_mean.append(shap_value_pd.abs().mean(axis=0).to_dict())
    
    return pd.DataFrame(shap_values_pd_abs_mean).fillna(0)

def get_top_shap_feature_names(shap_values, n_features=10, ascending=True):
    ab_mean_shap = get_abs_mean_shap_pd(shap_values)
    if ascending:
        list(ab_mean_shap.mean(0).sort_values().keys())[-n_features:]
    else:
        return list(ab_mean_shap.mean(0).sort_values(ascending=False).keys())[:n_features]


def plot_shap_bar(shap_values, max_display=10, save_dir=None):
    plt.close('all')
    combined_shap_values_pd = get_abs_mean_shap_pd(shap_values)

    sorted_shap_values_names = list(combined_shap_values_pd.mean(0).sort_values().keys())[-max_display:]
    # sorted_shap_values_names = list(combined_shap_values_pd.astype(bool).sum(axis=0).sort_values().keys())[-max_display:]

    fig, ax = plt.subplots(figsize=(5, 0.5*(len(sorted_shap_values_names))))

    sorted_shap_values = combined_shap_values_pd[sorted_shap_values_names]
    
    ax.barh(sorted_shap_values.columns, sorted_shap_values.mean(0), color='lightblue', edgecolor="black")
    ax.errorbar(sorted_shap_values.mean(0),sorted_shap_values.columns, xerr=sorted_shap_values.replace(0,np.NaN).std(0), color='r', fmt="o", label="Standard deviation of non-zero values")

    for index, category in enumerate(sorted_shap_values.columns):
        col = sorted_shap_values[category]
        ax.text(col.mean()+0.001, index+0.1,
                f"{col.mean():+.2e} ({len(col[col>0])}/{len(col)})")

    ax.set_xlabel('Mean Absolute SHAP Values')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2))
    ax.set_xlim([0, sorted_shap_values.mean(0).max()+0.04])
    fig.tight_layout()

    if save_dir is None:
        plt.show()
        return fig, ax

    fig.savefig(save_dir, dpi=1200, bbox_inches='tight')
    return fig, ax


def summate_shap_bar(shap_values, feature_substrings, save_dir=None, size=None):
    plt.close('all')

    combined_shap_values_pd = get_abs_mean_shap_pd(shap_values)

    mean_shap_values = combined_shap_values_pd.mean(0)

    result_feature_values = {}
    result_feature_count = {}
    for substring in feature_substrings:
        features = mean_shap_values.filter(regex=substring)
        stripped_substring = substring.translate(str.maketrans('', '', string.punctuation))
        result_feature_values[stripped_substring] = features.sum() if not features.empty else 0
        result_feature_count[stripped_substring] = len(features)

    sorted_feature_values = sorted(result_feature_values.items(), key=lambda x: x[1], reverse=False)

    sorted_categories = [x[0] for x in sorted_feature_values]
    values = [x[1] for x in sorted_feature_values]

    plt.barh(sorted_categories, values, color='lightblue', edgecolor="black")

    for index, category in enumerate(sorted_categories):
        plt.text(result_feature_values[category], index,
                #  f"{result_feature_values[category]:+.2e} (n={result_feature_count[category]})")
                 f"{result_feature_values[category]:+.2e}")


    plt.xlabel('SHAP Value')

    fig = plt.gcf()

    fig.tight_layout()

    if save_dir is None:
        plt.show()
        return fig

    fig.savefig(save_dir, dpi=1200,)
    return fig


def plot_dependence_scatter_plot(shap_values, shap_datas, n_features=10, save_dir=None, plots_per_row=3):
    abs_mean_combined_shap_values_pd = get_abs_mean_shap_pd(shap_values)

    sorted_shap_values_names = list(abs_mean_combined_shap_values_pd.mean(0).sort_values(ascending=False).keys())[:n_features]

    combined_shap_values_pd = pd.concat(shap_values)
    combined_shap_data_pd = pd.concat(shap_datas)
    
    plots_per_row_ = min(plots_per_row, n_features)
    num_rows = (n_features - 1) // plots_per_row_ + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row_, figsize=(5*plots_per_row, 5*num_rows))

    letters = iter(string.ascii_uppercase)

    for idx in range(len(axes.flatten())):
        row = idx // plots_per_row_
        col_idx = idx % plots_per_row_

        if len(axes.shape)==2:
            selected_ax = axes[row, col_idx]
        elif len(axes.shape)==1:
            selected_ax = axes[col_idx]
        
        if idx >= n_features:
            fig.delaxes(selected_ax)
            continue

        col_name = sorted_shap_values_names[idx]

        scatter_data = pd.DataFrame({'x':combined_shap_values_pd[col_name], 'y':combined_shap_data_pd[col_name]}).dropna()
        
        selected_ax.scatter(scatter_data.x.to_numpy(), scatter_data.y.to_numpy(), linewidth=0)

        selected_ax.set_xlabel(col_name, fontsize=10)
        selected_ax.set_ylabel('SHAP value', fontsize=10)

        letter = next(letters)
        selected_ax.text(0.025, 0.975, f"({letter})", transform=selected_ax.transAxes, fontsize=12, va='top', ha='left')
    
    if save_dir is not None:
        fig.savefig(save_dir, dpi=1200, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
    plt.close('all')
