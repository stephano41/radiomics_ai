from __future__ import annotations
import pandas as pd
import shap
import mlflow
from autorad.models import MLClassifier
from autorad.inference import infer_utils

import matplotlib.pyplot as plt
import numpy as np
import string
import seaborn as sns
from src.utils.inference import get_run_info_as_series


def get_shap_values(run: str | pd.Series, existing_feature_df=None, existing_splits=None):
    if isinstance(run, str):
        run = get_run_info_as_series(run)

    artifact_uri = run.artifact_uri
    model = MLClassifier.load_from_mlflow(f"{artifact_uri}/model")
    preprocessor = mlflow.sklearn.load_model(f"{artifact_uri}/preprocessor")

    dataset_artifacts = infer_utils.load_dataset_artifacts(run)

    # allow override
    dataset_df = dataset_artifacts['df'] if existing_feature_df is None else existing_feature_df
    dataset_splits = dataset_artifacts['splits'] if existing_splits is None else existing_splits

    feature_dataset = infer_utils.load_feature_dataset(feature_df=dataset_df,
                                                       dataset_config=dataset_artifacts['dataset_config'],
                                                       splits=dataset_splits)

    # first preprocess the data, check if fit_resample is the last thing
    preprocessed_X, y = preprocessor._fit_transform(feature_dataset.X, feature_dataset.y)

    model.fit(preprocessed_X, y)

    explainer = shap.Explainer(model.predict_proba_binary, preprocessed_X)
    shap_values = explainer(preprocessed_X, max_evals=2 * len(feature_dataset.df.columns) + 1)

    return shap_values, dataset_df, dataset_splits


def plot_shap_bar(shap_values, max_display=10, save_dir=None, size=None):
    plt.close('all')
    shap_values_pd = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)

    ax = sns.barplot(shap_values_pd.abs(), 
                 errorbar=None, 
                 estimator='mean', 
                 color='lightblue', 
                 orient='y', 
                 edgecolor='black',
                 order=list(shap_values_pd.abs().mean(0).sort_values(ascending=False).keys()))

    # Add bar labels
    for i in ax.containers:
        ax.bar_label(i, fmt='%+.2e')

    ax.set_xlabel('Mean Absolute SHAP values')
    fig = plt.gcf()

    if save_dir is None:
        plt.show()
        return fig, ax

    fig.savefig(save_dir, dpi=1200, bbox_inches='tight')
    return fig, ax


def summate_shap_bar(shap_values, feature_substrings, max_display=10, save_dir=None, size=None):
    mean_shap_values = np.mean(np.abs(shap_values.values), 0)
    pd_shap_values = pd.Series(
        {shap_values.feature_names[i]: mean_shap_values[i] for i in range(len(shap_values.feature_names))})

    result_feature_values = {}
    result_feature_count = {}
    for substring in feature_substrings:
        features = pd_shap_values.filter(regex=substring)
        stripped_substring = substring.translate(str.maketrans('', '', string.punctuation))
        result_feature_values[stripped_substring] = features.sum() if not features.empty else 0
        result_feature_count[stripped_substring] = len(features)

    sorted_feature_values = sorted(result_feature_values.items(), key=lambda x: x[1], reverse=False)

    sorted_categories = [x[0] for x in sorted_feature_values]
    values = [x[1] for x in sorted_feature_values]

    plt.close('all')
    plt.barh(sorted_categories, values, color='lightblue', edgecolor="black")

    for index, category in enumerate(sorted_categories):
        plt.text(result_feature_values[category], index,
                 f"{result_feature_values[category]:+.2e} (n={result_feature_count[category]})")

    plt.xlabel('SHAP Value')

    fig = plt.gcf()

    if size is not None:
        fig.set_size_inches(size, forward=True)
    fig.tight_layout()

    if save_dir is None:
        plt.show()
        return fig

    fig.savefig(save_dir, dpi=1200)
    return fig


def plot_dependence_scatter_plot(shap_values, n_features, save_dir=None, plots_per_row=3):
    top_features_indices = np.mean(np.abs(shap_values.values), 0).argsort()
    n_features = min(n_features, len(top_features_indices))

    plots_per_row_ = min(plots_per_row, n_features)
    num_rows = (n_features - 1) // plots_per_row_ + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row_, figsize=(6*plots_per_row, 6*num_rows))

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
    
        shap.plots.scatter(shap_values[:, np.where(top_features_indices==idx)[0][0]], ax=selected_ax, show=False)

        xlabel = selected_ax.get_xlabel()
        ylabel = selected_ax.get_ylabel()

        selected_ax.set_xlabel(xlabel, fontsize=10)
        selected_ax.set_ylabel(ylabel, fontsize=10)

        letter = next(letters)
        selected_ax.text(0.05, 0.95, f"({letter})", transform=selected_ax.transAxes, fontsize=12, va='top', ha='left')
    
    if save_dir is not None:
        fig.savefig(f"{save_dir}/dependence_plot_feature.png", dpi=1200, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
    plt.close('all')
