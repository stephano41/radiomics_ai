import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from src.models.autoencoder import Encoder


def run(idx):
    print(idx)
    images = torch.randn([10, 1, 16, 16, 16]).type(torch.float32)
    encoder = Encoder(module='src.models.autoencoder.DummyVAE',
                                module__in_channels=1,
                                module__latent_dim=64,
                                module__hidden_dims=[8, 16, 32],
                                module__finish_size=2,
                                criterion='src.models.autoencoder.VAELoss',
                                max_epochs=2,
                                dataset='src.dataset.dummy_dataset.DummyDataset',
                                device='cuda'
                                )
    encoder.fit(images)
    encoder.transform(images)

    return idx[0]


def parallel_run(num_processes, num_rounds):
    idxs = torch.randn([num_rounds,2])
    scores = []
    with mp.Pool(processes=num_processes) as pool:
        for score in tqdm(pool.imap_unordered(run, idxs), total=len(idxs)):
            scores.append(score)
    print(scores)


if __name__ == '__main__':
    parallel_run(5, 10)