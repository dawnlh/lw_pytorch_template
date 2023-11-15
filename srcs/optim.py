import torch
import inspect
# ============================
# build optimizer & lr_scheduler
# ============================


OPTIMS, SCHEDS = {}, {}
for module_name in dir(torch.optim):
    if module_name.startswith('__'):
        continue
    _optim = getattr(torch.optim, module_name)
    if inspect.isclass(_optim) and issubclass(_optim,torch.optim.Optimizer):
        OPTIMS.update({module_name:_optim})

for module_name in dir(torch.optim.lr_scheduler):
    if module_name.startswith('__'):
        continue
    _sched = getattr(torch.optim.lr_scheduler, module_name)
    if inspect.isclass(_sched) and issubclass(_sched, torch.optim.lr_scheduler._LRScheduler):
        SCHEDS.update({module_name: _sched})


def build_optim(name, *args, **kwargs):
    """
    build optimizer
    """
    if name in OPTIMS:
        return OPTIMS.get(name)(*args, **kwargs)
    else:
        raise NotImplementedError


def build_sched(name, *args, **kwargs):
    """
    build scheduler
    """
    if name in SCHEDS:
        return SCHEDS.get(name)(*args, **kwargs)
    else:
        raise NotImplementedError