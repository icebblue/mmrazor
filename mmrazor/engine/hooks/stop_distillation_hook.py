# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmrazor.registry import HOOKS


@HOOKS.register_module()
class StopDistillHook(Hook):
    """Stop distilling at a certain time.

    Args:
        stop_epoch (int): Stop distillation at this epoch.
    """

    priority = 'LOW'

    def __init__(self, stop_epoch: int) -> None:
        self.stop_epoch = stop_epoch

    def before_train_epoch(self, runner) -> None:
        """Stop distillation."""
        if runner.epoch >= self.stop_epoch:
            model = runner.model
            # TODO: refactor after mmengine using model wrapper
            if is_model_wrapper(model):
                model = model.module
            assert hasattr(model, 'distillation_stopped')

            runner.logger.info('Distillation has been stopped!')
            model.distillation_stopped = True

@HOOKS.register_module()
class Swin2I3dHook(Hook):

    def __init__(self, interval=1):
        self.interval = interval

    def before_train_iter(self, runner, batch_idx, data_batch=None, video_idx =None):
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
        """
        print(f"Batch index: {batch_idx}")
        print(data_batch.keys())
        print(data_batch)
        print("-------------------------------")