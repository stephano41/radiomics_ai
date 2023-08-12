from sklearn.base import BaseEstimator, TransformerMixin
from skorch import NeuralNetRegressor
from skorch.callbacks import PassthroughScoring, PrintLog, EarlyStopping
from skorch.dataset import ValidSplit
from .autoencoder import VAELoss


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, module, **encoder_kwargs):
        if len(encoder_kwargs)<=0:
            encoder_kwargs={
                "max_epochs": 10,
                "callbacks": [
                    ('early_stop', EarlyStopping(
                        monitor='valid_loss',
                        patience=5
                    ))
                ],
                "train_split": ValidSplit(5),
                'criterion': VAELoss
            }

        self.model = NeuralNetRegressor(module, **encoder_kwargs)

    def fit(self, X, y=None):
        self.model.fit(X, X)
        return self

    def transform(self, X, y=None):
        return self.model.predict(X)