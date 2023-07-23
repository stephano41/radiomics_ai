import numpy as np
from sklearn.utils import resample


class BootstrapGenerator(object):
    """
    Parameters
    ----------

    n_splits : int (default=200)
        Number of bootstrap iterations.
        Must be larger than 1.

    random_seed : int (default=None)
        If int, random_seed is the seed used by
        the random number generator.


    Returns
    -------
    train_idx : ndarray
        The training set indices for that split.

    test_idx : ndarray
        The testing set indices for that split.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/BootstrapOutOfBag/

    """

    def __init__(self, n_splits=200, random_seed=None, stratify=False):
        self.random_seed = random_seed

        self.stratify=stratify

        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("Number of splits must be greater than 1.")
        self.n_splits = n_splits

    def split(self, X, y, groups=None):
        """

        y : array-like or None (default: None)
            The target variable.

        groups : array-like or None (default: None)
            Argument is not used and only included as parameter
            for compatibility, similar to `KFold` in scikit-learn.

        """
        sample_idx = np.arange(X.shape[0])
        set_idx = set(sample_idx)

        if self.stratify:
            stratify_arg = y
        else:
            stratify_arg = None

        for _ in range(self.n_splits):
            train_idx = resample(sample_idx, replace=True, stratify=stratify_arg, random_state=self.random_seed)
            test_idx = np.array(list(set_idx - set(train_idx)))

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility with scikit-learn.

        y : object
            Always ignored, exists for compatibility with scikit-learn.

        groups : object
            Always ignored, exists for compatibility with scikit-learn.

        Returns
        -------

        n_splits : int
            Returns the number of splitting iterations in the cross-validator.

        """
        return self.n_splits
