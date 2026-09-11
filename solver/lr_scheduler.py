from __future__ import print_function, absolute_import
import torch

AVAI_SCH = ['single_step', 'multi_step', 'cosine']
AVAI_WARMUP = ['linear', 'exp']


class WarmupScheduler:
    def __init__(self, scheduler, warmup_epochs=0, warmup_type='linear', base_lr=0.001):
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self, epoch=None):
        if self.current_epoch < self.warmup_epochs:
            if self.warmup_type == 'linear':
                # lr = base_lr * (step / warmup_steps)
                warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
                for param_group in self.scheduler.optimizer.param_groups:
                    param_group['lr'] = self.base_lr * warmup_factor
            elif self.warmup_type == 'exp':
                min_lr = 1e-6
                if self.current_epoch == 0:
                    for param_group in self.scheduler.optimizer.param_groups:
                        param_group['lr'] = min_lr
                else:
                    exp_factor = (self.current_epoch + 1) / self.warmup_epochs
                    lr = min_lr * (self.base_lr / min_lr) ** exp_factor
                    for param_group in self.scheduler.optimizer.param_groups:
                        param_group['lr'] = lr
            self.current_epoch += 1
        else:
            if epoch is not None:
                self.scheduler.step(epoch - self.warmup_epochs)
            else:
                self.scheduler.step()
            self.current_epoch += 1

    def state_dict(self):
        return {
            'scheduler': self.scheduler.state_dict(),
            'warmup_epochs': self.warmup_epochs,
            'warmup_type': self.warmup_type,
            'base_lr': self.base_lr,
            'current_epoch': self.current_epoch
        }

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_type = state_dict['warmup_type']
        self.base_lr = state_dict['base_lr']
        self.current_epoch = state_dict['current_epoch']

    def get_last_lr(self):
        """返回当前学习率列表"""
        return [group['lr'] for group in self.scheduler.optimizer.param_groups]


def build_lr_scheduler(
    optimizer, lr_scheduler='single_step', stepsize=1, gamma=0.1, max_epoch=1,
    warmup_epochs=0, warmup_type='linear'
):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is single_step.
        stepsize (int or list, optional): step size to decay learning rate. When ``lr_scheduler``
            is "single_step", ``stepsize`` should be an integer. When ``lr_scheduler`` is
            "multi_step", ``stepsize`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.
        warmup_epochs (int, optional): number of warmup epochs. Default is 0 (no warmup).
        warmup_type (str, optional): warmup type. Default is linear.

    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError(
            'Unsupported scheduler: {}. Must be one of {}'.format(
                lr_scheduler, AVAI_SCH
            )
        )

    if lr_scheduler == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

    # Get base learning rate from optimizer
    base_lr = optimizer.param_groups[0]['lr']

    # Wrap with WarmupScheduler if warmup is enabled
    if warmup_epochs > 0:
        if warmup_type not in AVAI_WARMUP:
            raise ValueError(
                'Unsupported warmup type: {}. Must be one of {}'.format(
                    warmup_type, AVAI_WARMUP
                )
            )
        scheduler = WarmupScheduler(
            scheduler,
            warmup_epochs=warmup_epochs,
            warmup_type=warmup_type,
            base_lr=base_lr
        )

    return scheduler
