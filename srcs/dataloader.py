import torch

from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

# ===========================
# build dataloader
# ===========================


def build_dataloader(status, dataset, val=0, batch_size=1, num_workers=0,is_distributed=False):
    if status == 'train':
        return build_train_dataloader(dataset, val, batch_size, num_workers,is_distributed)
    elif status in ['test', 'realexp', 'simuexp']:
        return build_test_dataloader(dataset, batch_size, num_workers)
    else:
        raise NotImplementedError(
            f"status {status} not implemented")


def build_train_dataloader(train_dataset, val=0, batch_size=1, num_workers=0,is_distributed=False):
    """
    Build dataloader for training and validation
    """

    if isinstance(val, (int, float)):
        assert 0 <= val < 1, "val_split should be within 0 to 1"
        num_total = len(train_dataset)
        num_valid = int(num_total * val)
        num_train = num_total - num_valid
        train_dataset, val_dataset = random_split(
            train_dataset, [num_train, num_valid])


    if not val_dataset:
        # val_dataset is empty
        val_dataloader = None

    if is_distributed:
        # multi GPU
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      sampler=train_sampler,
                                      num_workers=num_workers)
        if val_dataset:
            val_sampler = DistributedSampler(val_dataset, shuffle=True)
            val_dataloader = DataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        sampler=val_sampler,
                                        num_workers=num_workers)
    else:
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
        if val_dataset:
            val_dataloader = DataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers)
    return train_dataloader, val_dataloader


def build_test_dataloader(test_dataset, batch_size=1, num_workers=0):
    """
    Build dataloader for testing and exp
    """

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    return test_dataloader
