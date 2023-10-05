import os

import torch
from skorch.callbacks import EarlyStopping, GradientNormClipping

from src.preprocessing import SitkImageProcessor
from src.models.autoencoder import Encoder, VanillaVAE, BetaVAELoss, MSSIM
from src.pipeline.pipeline_components import get_multimodal_feature_dataset, split_feature_dataset
from src.training import EncoderTrainer


def main():
    output_dir = './outputs/meningioma/2023-09-01-01-32-46'
    # setup dataset, extract features, split the data
    feature_dataset = get_multimodal_feature_dataset(data_dir='./data/meningioma_data',
                                                     image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'),
                                                     mask_stem='mask',
                                                     target_column='Grade',
                                                     label_csv_path='./data/meningioma_meta.csv',
                                                     extraction_params='./conf/radiomic_params/meningioma_mr.yaml',
                                                     feature_df_merger={
                                                         '_target_': 'src.pipeline.df_mergers.meningioma_df_merger'},
                                                     n_jobs=6,
                                                     existing_feature_df=os.path.join(output_dir,
                                                                                      'extracted_features.csv'),
                                                     additional_features=['ID']
                                                     )

    feature_dataset = split_feature_dataset(feature_dataset,
                                            existing_split=os.path.join(output_dir, 'splits.yml'))

    sitk_processor = SitkImageProcessor('./outputs', './data/meningioma_data', mask_stem='mask',
                                        image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'), n_jobs=6)

    encoder = Encoder(VanillaVAE,
                      module__in_channels=5,
                      module__latent_dim=128,
                      module__hidden_dims=[32, 64, 128],
                      module__finish_size=2,
                      criterion=MSSIM,
                      std_dim=(0, 2, 3, 4),
                      max_epochs=200,
                      output_format='pandas',
                      callbacks=[EarlyStopping(load_best=True),
                                 GradientNormClipping(1),
                                 # SimpleLoadInitState(f_optimizer='outputs/saved_models/optimizer.pt',
                                 #                     f_params='outputs/saved_models/params.pt')
                                 ],
                      optimizer=torch.optim.AdamW,
                      lr=0.001,
                      # criterion__loss_type='B',
                      # criterion__gamma=10.0,
                      # criterion__max_capacity=1,

                      # criterion__alpha=10.0,
                      # criterion__beta=1.0
                      criterion__in_channels=5,
                      criterion__window_size=4,
                      transform_kwargs=dict(thetaX=(-90, 90),
                                            thetaY=(-90, 90),
                                            thetaZ=(-90, 90),
                                            tx=(-1, 1),
                                            ty=(-1, 1),
                                            tz=(-1, 1),
                                            scale=(1, 1),
                                            n=10),
                      device='cuda'
                      )

    param_grid = {
        'lr': [0.001, 0.0001],
        'criterion__window_size': [2,4,8],
        'criterion__kld_weight': [0.001, 0.01, 0.0001]

    }

    trainer = EncoderTrainer(encoder, param_grid, feature_dataset, sitk_processor)
    trainer.run(wandb_kwargs={'project': 'autoencoder_tuning',
                              'dir': './outputs',
                              # 'mode': 'off'
                              },
                save_model=False)

if __name__ == '__main__':
    main()