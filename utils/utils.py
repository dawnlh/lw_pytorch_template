import os
import os.path as osp
import platform
from shutil import copyfile
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch.distributed as dist
import torch


# =================
# basic functions
# =================
def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')

def read_wav_np(path):
    sr, wav = read(path)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return sr, wav

def file_traverse(dir, ext=None):
    """
    traverse all the files and get their paths
    Args:
        dir (str): root dir path
        ext (list[str], optional): included file extensions. Defaults to None, meaning inculding all files.
    """

    data_paths = []
    skip_num = 0
    file_num = 0

    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            img_path = osp.join(dirpath, filename)
            if ext and img_path.split('.')[-1] not in ext:
                print('Skip a file: %s' % (img_path))
                skip_num += 1
            else:
                data_paths.append(img_path)
                file_num += 1
    return sorted(data_paths), file_num, skip_num


def get_file_path(data_dir, ext=None):
    """
    Get file paths for given directory or directories

    Args:
        data_dir (str): root dir path
        ext (list[str], optional): included file extensions. Defaults to None, meaning inculding all files.
    """

    if isinstance(data_dir, str):
        # single dataset
        data_paths, file_num, skip_num = file_traverse(data_dir, ext)
    elif isinstance(data_dir, list):
        # multiple datasets
        data_paths, file_num, skip_num = [], 0, 0
        for data_dir_n in sorted(data_dir):
            data_paths_n, file_num_n, skip_num_n = file_traverse(
                data_dir_n, ext)
            data_paths.extend(data_paths_n)
            file_num += file_num_n
            skip_num += skip_num_n
    else:
        raise ValueError('data dir should be a str or a list of str')

    return sorted(data_paths), file_num, skip_num


def is_master():
    # check whether it is the master node
    return not dist.is_initialized() or dist.get_rank() == 0

def ddp_init(rank=0, num_gpus=1):
    if(platform.system() == 'Windows'):
        backend = 'gloo'
    elif(platform.system() == 'Linux'):
        backend = 'nccl'
    else:
        raise RuntimeError('Unknown Platform (Windows and Linux are supported')
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:34567',
        world_size=num_gpus,
        rank=rank)

def collect(scalar):
    """
    util function for DDP.
    syncronize a python scalar or pytorch scalar tensor between GPU processes.
    """
    # move data to current device
    if not isinstance(scalar, torch.Tensor):
        scalar = torch.tensor(scalar)
    scalar = scalar.to(dist.get_rank())

    # average value between devices
    dist.reduce(scalar, 0, dist.ReduceOp.SUM)
    return scalar.item() / dist.get_world_size()


def save_checkpoint(epoch, state, ckp_dir, logger=None, save_latest_k=-1, milestone_ckp=[]):
    # Saving checkpoints

    filename = osp.join(ckp_dir,
                    f'ckp-epoch{epoch}.pth')
    torch.save(state, filename)
    if logger:
        logger.info(
        f"Save ckpt: {filename}")
    if milestone_ckp and epoch in milestone_ckp:
        landmark_path = osp.join(
            ckp_dir, f'model_epoch{epoch}.pth')
        copyfile(filename, landmark_path)
        logger.info(
            f"Save milestone checkpoint at epoch {epoch}!")
    if save_latest_k > 0:
        latest_path = osp.join(
            ckp_dir, 'model_latest.pth')
        copyfile(filename, latest_path)
        outdated_path = osp.join(ckp_dir,
                            f'ckp-epoch{epoch-save_latest_k}.pth')
        try:
            os.remove(outdated_path)
        except FileNotFoundError:
            # this happens when current model is loaded from checkpoint
            # or target file is already removed somehow
            pass
