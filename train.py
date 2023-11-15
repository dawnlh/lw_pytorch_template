import os.path as osp
import os, time, yaml
import numpy as np
import argparse
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from srcs.trainer import train

# fix random seeds for reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # arg & cfg
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.resume is not None:
        cfg.train.resume = args.resume
        
    # dir
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg.work_dir = osp.join('work_dir', cfg.exp_name, 'train', datetime)
    cfg.ckp_dir = osp.join(cfg.work_dir, 'ckpt')
    cfg.log_dir = osp.join(cfg.work_dir, 'log')
    os.makedirs(cfg.ckp_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # gpu
    if not cfg.gpus or cfg.gpus == -1:
        cfg.gpus = list(range(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        map(str, cfg.gpus))  # set visible gpu ids
    cfg.num_gpus = len(cfg.gpus)

    # save config
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    with open(osp.join(cfg.work_dir, 'config.yaml'),'w') as f:
        f.write(config_yaml)

    # train
    if cfg.num_gpus > 1:
        mp.spawn(train, nprocs=cfg.num_gpus,args=(cfg,))
    else:
        train(0, cfg)
