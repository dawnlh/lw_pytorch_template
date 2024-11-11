import os
import torch
import time
from tqdm import tqdm
from utils.eval import gpu_inference_time, calc_model_complexity
from utils.image import tensor2uint, imsave_n, imsave
from utils.logger import Logger
from srcs.model import build_model
from srcs.dataloader import build_dataloader
from srcs.dataset import build_dataset

def test(cfg):
    ## model, optim, lr_sched
    model = build_model(**cfg.model)

    ## data
    test_dataset = build_dataset(cfg.test_dataset)
    test_dataloader = build_dataloader(status=cfg.test.status, dataset=test_dataset, batch_size=cfg.test_dataloader.batch_size, num_workers=cfg.test_dataloader.num_workers)

    ## metrics
    # import here to avoid CUDA_VISIBLE_DEVICE setting failure
    from srcs.metrics import build_metrics
    metrics = build_metrics(**cfg.metrics)

    ## logger & writer
    logger = Logger(name='train',log_path=os.path.join(cfg.work_dir, 'log.txt'))

    ## load checkpoint
    logger.info(f"Load pre_train model from: \n\t{cfg.test.checkpoint}")
    checkpoint = torch.load(cfg.test.checkpoint)
    model.load_state_dict(checkpoint['model'])
    
    ## gpu
    if cfg.num_gpus>1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(cfg.num_gpus)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    ## model info
    calc_model_complexity(model, input_res=(3, 256, 256), logger=logger)
    gpu_inference_time(model, input_shape=(1, 3, 256, 256), logger=logger, device=device)

    # init
    model.eval()
    if cfg.test.save_img:
        os.makedirs(cfg.work_dir+'/images')

    ## test loop
    model.eval()
    total_metrics = {}
    time_start = time.time()
    with torch.no_grad():
        for batch_idx, (img_noise, img_target) in enumerate(tqdm(test_dataloader)):

            # data to device
            img_noise, img_target = img_noise.to(device), img_target.to(device)

            # forward
            output = model(img_noise)

            # save image
            if cfg.test.save_img:
                for k, (in_img, out_img, gt_img) in enumerate(zip(img_noise, output, img_target)):
                    in_img = tensor2uint(in_img)
                    out_img = tensor2uint(out_img)
                    gt_img = tensor2uint(gt_img)
                    imgs = [in_img, out_img, gt_img]
                    imsave_n(
                        imgs, f'{cfg.work_dir}/images/test{batch_idx+1:03d}_{k+1:03d}.png')
                    # imsave(in_img, f'{cfg.work_dir}/images/test{i+1:03d}_{k+1:03d}_in.png')

            if cfg.test.status != 'realexp':
                cur_batch_size = img_noise.shape[0]
                calc_metrics = metrics(output, img_target)
                for k, v in calc_metrics.items():
                    total_metrics.update(
                        {k: total_metrics.get(k, 0) + v * cur_batch_size})

        # time cost
        time_end = time.time()
        time_cost = time_end-time_start
        n_samples = len(test_dataloader.sampler)

        # metrics average
        test_metrics = {k: v / n_samples for k, v in total_metrics.items()}
        metrics_str = ' '.join(
            [f'{k}: {v:6.4f}' for k, v in test_metrics.items()])

        logger.info(
            '='*80 + f'\n time/sample {time_cost/n_samples:6.4f} ' + metrics_str + '\n' + '='*80)
