import os, time
import os.path as osp
import torch
import argparse
from scipy.io.wavfile import write
from omegaconf import OmegaConf
import numpy as np
from srcs.tester import test

# fix random seeds for reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint', type=str, help="path of checkpoint pt file for evaluation")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.checkpoint is not None:
        cfg.test.checkpoint = args.checkpoint

    # dir
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg.work_dir = osp.join('work_dir', cfg.exp_name, 'test', datetime)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # gpu
    if not cfg.gpus or cfg.gpus == -1:
        cfg.gpus = list(range(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        map(str, cfg.gpus))  # set visible gpu ids
    cfg.num_gpus = len(cfg.gpus)

    # save config
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    with open(osp.join(cfg.work_dir, 'config.yaml'), 'w') as f:
        f.write(config_yaml)

    # test
    test(cfg)
