import yaml
import os
from autorad.inference import infer_utils
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import mlflow

def plot_correlation_graph(run, x_axis_labels=None, plots_per_row=3, save_dir=None):
    if isinstance(run, str):
        run = pd.Series(dict(mlflow.get_run(run).info))
    artifact_uri = run.artifact_uri.removeprefix('file://')
    
    with open(os.path.join(artifact_uri, 'selected_features.yaml'), 'r') as file:
        selected_features = yaml.load(file, Loader=yaml.FullLoader)

    dataset_artifacts = infer_utils.load_dataset_artifacts(run)
    feature_dataset = infer_utils.load_feature_dataset(feature_df=dataset_artifacts['df'],
                                            dataset_config=dataset_artifacts['dataset_config'],
                                            splits=dataset_artifacts['splits'])

    selected_dataframe = feature_dataset.df[selected_features + [feature_dataset.target]]

    num_vars = len(selected_dataframe.columns) - 1  # Exclude the dependent variable
    num_rows = (num_vars - 1) // plots_per_row + 1

    # sns.set_theme(style="whitegrid", palette="gray")

    fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row, figsize=(15, 5*num_rows))

    for i, col in enumerate(selected_dataframe.columns[:-1]):  # Exclude the last column which is the dependent variable
        row = i // plots_per_row
        col_idx = i % plots_per_row

        sns.boxplot(x=feature_dataset.target, y=col, data=selected_dataframe, ax=axes[row, col_idx], fill=False)
        axes[row, col_idx].set_title(f'{col}')

                # Set x-axis labels if provided
        if x_axis_labels is not None:
            axes[row, col_idx].set_xticklabels(x_axis_labels)

    plt.tight_layout()
    if save_dir is None:
        plt.show()
    else:
        if save_dir.endswith('.png'):
            fig.savefig(save_dir, dpi=1200)
        else:
            fig.savefig(save_dir)