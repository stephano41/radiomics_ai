from contextlib import suppress

from skorch import NeuralNet
from skorch.callbacks import Callback


class SimpleLoadInitState(Callback):
    """Loads the model, optimizer, and history from a checkpoint into a
    :class:`.NeuralNet` when training begins.

    On the first run, the :class:`.Checkpoint` saves the model, optimizer, and
    history when the validation loss is minimized. During the first run,
    there are no files on disk, thus :class:`.LoadInitState` will
    not load anything. When running the example a second time,
    :class:`LoadInitState` will load the best model from the first run and
    continue training from there.

    Parameters
    ----------
    checkpoint: :class:`.Checkpoint`
      Checkpoint to get filenames from.

    use_safetensors : bool (default=False)
      Whether to use the ``safetensors`` library to load the state. By default,
      PyTorch is used, which in turn uses :mod:`pickle` under the hood. When the
      state was saved using ``safetensors``, (e.g. by enabling it with the
      :class:`.Checkpoint`), you should set this to ``True``.

    """

    def __init__(self,
                 f_params=None,
                 f_optimizer=None,
                 f_criterion=None,
                 f_history=None):

        self.f_params = f_params
        self.f_optimizer = f_optimizer
        self.f_criterion = f_criterion
        self.f_history = f_history

    def initialize(self):
        self.did_load_ = False
        return self

    def on_train_begin(self, net: NeuralNet,
                       X=None, y=None, **kwargs):
        if not self.did_load_:
            self.did_load_ = True
            net.load_params(f_params=self.f_params, f_history=self.f_history, f_optimizer=self.f_optimizer,
                            f_criterion=self.f_criterion)

