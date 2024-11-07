import logging

from .lr_schedulers import *


__scheduler = {
    'step': StepScheduler,
    'cosine': CosineScheduler,
}


def create(lr_scheduler, num_sample, **kwargs):
    if lr_scheduler not in __scheduler.keys():
        logging.info('')
        logging.error('This lr_scheduler is not implemented: {}!'.format(lr_scheduler))
        raise ValueError()
    return __scheduler[lr_scheduler](num_sample, **kwargs)
