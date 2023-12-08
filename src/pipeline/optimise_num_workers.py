import os
from time import time
import multiprocessing as mp
import hydra
from hydra.utils import instantiate
from matplotlib import pyplot as plt, cm
from numba.types import itertools
from omegaconf import OmegaConf
import numpy as np
from src.pipeline.pipeline_components import get_multimodal_feature_dataset


def optimise_num_workers(config):
    if config.preprocessing.autoencoder is None:
        print('no autoencoder in this config')
        return
    output_dir = hydra.utils.HydraConfig.get().run.dir

    feature_dataset = get_multimodal_feature_dataset(**OmegaConf.to_container(config.feature_dataset, resolve=True))
    autoencoder = instantiate(config.preprocessing.autoencoder, _convert_='object')
    dataset_train, _ = autoencoder.get_split_datasets(feature_dataset.X.ID, None)

    num_workers_range = list(range(2, mp.cpu_count(), 2))
    batch_size_range = list(range(5, 35, 5))
    results = []

    best_config = None
    min_timing = float('inf')

    for num_workers, batch_size in itertools.product(num_workers_range, batch_size_range):
        autoencoder.set_params(batch_size=batch_size, iterator_train__num_workers=num_workers)
        train_loader = autoencoder.get_iterator(dataset_train, training=True)

        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        avg_timing = (end-start)/3
        results.append((num_workers, batch_size, avg_timing))
        print("Finish with:{} second, num_workers={}, batch_size={}".format((end - start)/3, num_workers, batch_size))

        if avg_timing < min_timing:
            min_timing = avg_timing
            best_config = (num_workers, batch_size)

    if best_config:
        print("\nBest Configuration:")
        print("Number of Workers: {}".format(best_config[0]))
        print("Batch Size: {}".format(best_config[1]))
        print("Lowest Average Time: {} seconds\n".format(min_timing))

    # Plot the results in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num_workers, batch_size, timings= zip(*results)
    timings = np.array(timings)

    # Normalize timings for color mapping
    norm_timings = (timings - min(timings)) / (max(timings) - min(timings))
    colors = cm.viridis(norm_timings)

    # Bar chart
    ax.bar3d(num_workers, batch_size, 0, 1, 1, timings, shade=True, color=colors)

    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('Average Time (seconds)')

    # Add color bar
    cbar = fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax, pad=0.1)
    cbar.set_label('Normalized Average Time')

    plt.savefig(os.path.join(output_dir, 'optimise_num_workers.png'))
    plt.show()