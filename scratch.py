import torch
from src.dataset.visualisation import plot_slices
from src.pipeline.pipeline_components import get_multimodal_feature_dataset
from src.models.autoencoder.fmcib_model import FMCIBModel
import pandas as pd
import torchio as tio
from skorch.callbacks import EarlyStopping, GradientNormClipping, ParamMapper
from src.dataset import TransformingDataLoader, SkorchSubjectsDataset
from sklearn.model_selection import ParameterGrid
from src.models.autoencoder import BetaVAELoss, Encoder
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer
from src.metrics import roc_auc
import skorch
import optuna
from datetime import datetime
import os

def freeze_net(net):
    if len(net.history) <=5:
        return skorch.utils.freeze_parameter
    else:
        return skorch.utils.unfreeze_parameter

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
                                                 existing_feature_df='tests/meningioma_feature_dataset.csv',
                                                 additional_features=['ID']
                                                 )

id_list = feature_dataset.X['ID'].to_numpy()

dataset_train_transform = tio.Compose([tio.Resample((1, 1, 1)),
                                                    tio.ToCanonical(),
                                                    tio.Mask(masking_method='mask', outside_value=0),
                                                    tio.CropOrPad(target_shape=(96, 96, 96), mask_name='mask'),
                                                    tio.RescaleIntensity(masking_method='mask'),
                                                    # tio.ZNormalization(masking_method='mask'),
                                                    ])
encoder_kwargs = dict(module=FMCIBModel,
                    module__input_channels=5,
                    module__weights_path='outputs/pretrained_models/model_weights.torch',
                    batch_size=3,
                    # output_format='pandas',
                    max_epochs=200,
                    callbacks=[EarlyStopping(load_best=True),
                               GradientNormClipping(1),
                            #    ParamMapper('trunk.*', schedule=freeze_net)
                               ],
                    optimizer='torch.optim.AdamW',
                    # lr=0.001,
                    iterator_train=TransformingDataLoader,
                    iterator_train__augment_transforms=tio.Compose([tio.RandomGamma(log_gamma=0.1, label_keys=('mask',)),
                                                                    tio.RandomAffine(p=0.5, label_keys=('mask',),
                                                                                     scales=0.1, degrees=0,
                                                                                     translation=0, isotropic=True),
                                                                    tio.RandomFlip(flip_probability=0.5,
                                                                                   label_keys=('mask',), axes=(0, 1, 2))
                                                                    ]),
                    iterator_train__num_workers=15,
                    iterator_train__shuffle=True,
                    iterator_valid__num_workers=4,
                    dataset=SkorchSubjectsDataset,
                    dataset__transform=dataset_train_transform,
                    dataset__data_dir='./data/meningioma_data',
                    dataset__image_stems=('registered_adc', 't2', 'flair', 't1', 't1ce'),
                    dataset__mask_stem='mask',
                    device='cuda',
                    criterion='torch.nn.BCEWithLogitsLoss',
                    optimizer__param_groups=[
                        ('trunk.*', {'lr': 0.0001}),
                        ('latent_var_head.*', {'lr': 0.001}),
                        ('heads.*', {'lr': 0.001}),

                    ]
                    )

# parameter_grid = ParameterGrid({'criterion__loss_type':['B'],'criterion__gamma':[2,4,6,8,16,32],'criterion__max_capacity':[5,25,50,100], 'criterion__Capacity_max_iter':[1e2,1e3,1e4,1e5,1e6]})
# parameter_grid = {'optimizer__param_groups':[[('trunk.*', {'lr': 0.000001}),('latent_var_head.*', {'lr': 0.001}),('heads.*', {'lr': 0.001})],
#                                              [('trunk.*', {'lr': 0.000001}),('latent_var_head.*', {'lr': 0.01}),('heads.*', {'lr': 0.01})],
#                                              [('trunk.*', {'lr': 0.0000001}),('latent_var_head.*', {'lr': 0.01}),('heads.*', {'lr': 0.01})],
#                                              [('trunk.*', {'lr': 0.0000001}),('latent_var_head.*', {'lr': 0.001}),('heads.*', {'lr': 0.001})],
#                                              ]}

# encoder = Encoder(**encoder_kwargs)



# encoder.fit(id_list, feature_dataset.y.to_numpy())
# roc_auc_scorer = make_scorer(roc_auc)
# clf = GridSearchCV(encoder, parameter_grid, scoring=roc_auc_scorer,verbose=3,error_score='raise')
# clf.fit(id_list, feature_dataset.y.to_numpy())
# print(pd.DataFrame(clf.cv_results_))
# pd.DataFrame(clf.cv_results_).to_csv('outputs/foundation_model_test.csv')
# print(f'best params were: {clf.best_params_} which achieved {clf.best_score_}')

def objective(trial):
    lr_pretrained = trial.suggest_float('lr_pretrained', 1e-6,1e-2,log=True)
    lr_untrained = trial.suggest_float('lr_untrained', 1e-5,1e-1,log=True)
    
    parameters = {'optimizer__param_groups': [('trunk.*', {'lr': lr_pretrained}), ('latent_var_head.*', {'lr': lr_untrained}), ('heads.*', {'lr': lr_untrained})]}
    encoder_kwargs.update(parameters)

    encoder = Encoder(**encoder_kwargs)
    # Perform cross-validation with the current parameters
    roc_auc_scorer = make_scorer(roc_auc)
    scores = cross_val_score(encoder, id_list, feature_dataset.y.to_numpy(), cv=5, scoring=roc_auc_scorer)
    
    # Return the mean of cross-validation scores as the objective value
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best score: {study.best_value}")
print(f"Best params: {study.best_params}")

output_dir = f"outputs/dl_tests/{datetime.now().strftime('%Y%m%d%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

study.trials_dataframe().to_csv(os.path.join(output_dir, "study_df.csv"))