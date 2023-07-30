from autorad.models import MLClassifier as OrigMLClassifier
import numpy as np


class MLClassifier(OrigMLClassifier):
    def __getattr__(self, item):
        if item.startswith("__"):  # this allows for deepcopy
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(item)
            )
        return getattr(self.model, item)
