from torch.optim.lr_scheduler import _LRScheduler

class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (0.95 ** self.last_epoch) for base_lr in self.base_lrs]