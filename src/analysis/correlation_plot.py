import yaml
import os
from autorad.inference import infer_utils
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import mlflow

def plot_correlation_graph(run, feature_names=None, x_axis_labels=None, plots_per_row=3, save_dir=None):
    if isinstance(run, str):
        run = pd.Series(dict(mlflow.get_run(run).info))
    artifact_uri = run.artifact_uri.removeprefix('file://')
    
    if feature_names is None:
        with open(os.path.join(artifact_uri, 'selected_features.yaml'), 'r') as file:
            feature_names = yaml.load(file, Loader=yaml.FullLoader)

    dataset_artifacts = infer_utils.load_dataset_artifacts(run)
    feature_dataset = infer_utils.load_feature_dataset(feature_df=dataset_artifacts['df'],
                                            dataset_config=dataset_artifacts['dataset_config'],
                                            splits=dataset_artifacts['splits'])

    selected_dataframe = feature_dataset.df[feature_names + [feature_dataset.target]]

    num_vars = len(selected_dataframe.columns) - 1  # Exclude the dependent variable

    plots_per_row_ = min(plots_per_row, num_vars)

    num_rows = (num_vars - 1) // plots_per_row_ + 1

    # sns.set_theme(style="whitegrid", palette="gray")

    fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row_, figsize=(15, 5*num_rows))

    for i, col in enumerate(selected_dataframe.columns[:-1]):  # Exclude the last column which is the dependent variable
        row = i // plots_per_row_
        col_idx = i % plots_per_row_

        if len(axes.shape)==2:
            selected_ax = axes[row, col_idx]
        elif len(axes.shape)==1:
            selected_ax = axes[col_idx]

        sns.boxplot(x=feature_dataset.target, y=col, data=selected_dataframe, ax=selected_ax, fill=False)
        selected_ax.set_title(f'{col}')

                # Set x-axis labels if provided
        if x_axis_labels is not None:
            selected_ax.set_xticklabels(x_axis_labels)

    plt.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        if save_dir.endswith('.png'):
            fig.savefig(save_dir, dpi=1200)
        else:
            fig.savefig(save_dir)