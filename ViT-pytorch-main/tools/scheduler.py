import logging
import math
import argparse
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup based on steps and then constant based on epochs.
    """
    def __init__(self, optimizer, warmup_steps, total_epochs, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_epochs = total_epochs
        self.current_step = 0
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, _):
        if self.current_step < self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1.0, self.warmup_steps))
        else:
            lr_scale = 1.0
        self.current_step += 1
        return lr_scale

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup based on steps and then linear decay based on epochs.
    """
    def __init__(self, optimizer, warmup_steps, total_epochs, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_epochs = total_epochs
        self.current_step = 0
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, _):
        if self.current_step < self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            epoch = (self.current_step - self.warmup_steps) // self.warmup_steps
            lr_scale = max(0.0, float(self.total_epochs - epoch) / float(max(1.0, self.total_epochs)))
        self.current_step += 1
        return lr_scale

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup based on steps and then cosine decay based on epochs.
    """
    def __init__(self, optimizer, warmup_steps, total_epochs, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_epochs = total_epochs
        self.cycles = cycles
        self.current_step = 0
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        
    def lr_lambda(self, _):
        if self.current_step < self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1.0, self.warmup_steps))
        else:
            epoch = (self.current_step - self.warmup_steps) // self.warmup_steps
            progress = float(epoch) / float(max(1, self.total_epochs))
            lr_scale = max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
        self.current_step += 1
        return lr_scale
        

