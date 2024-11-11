import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from srcs.model import build_model
from srcs.optim import build_optim, build_sched
from srcs.dataloader import build_dataloader
from srcs.dataset import build_dataset
from srcs.loss import build_loss
from utils.logger import TensorboardWriter, Logger
from utils.utils import ddp_init, collect, save_checkpoint



def train(rank, cfg):
    ## gpus
    if cfg.num_gpus > 1:
        torch.cuda.set_device(rank)
        ddp_init(rank=rank, num_gpus=cfg.num_gpus)
    device = torch.device('cuda:{:d}'.format(rank))

    ## model, optim, lr_sched
    model = build_model(**cfg.model).to(device)

    optim = build_optim(**cfg.optim, params=model.parameters())
    lr_scheduler = build_sched(**cfg.lr_scheduler, optimizer=optim)

    ## data
    train_dataset = build_dataset(cfg.train_dataset)
    train_dataloader, val_dataloader = build_dataloader(status='train', dataset=train_dataset, val=cfg.train_dataloader.val,
                                        batch_size=cfg.train_dataloader.batch_size, num_workers=cfg.train_dataloader.num_workers, is_distributed=(cfg.num_gpus > 1))
    ## loss & metrics
    criterion = build_loss(**cfg.loss)
    # import here to avoid CUDA_VISIBLE_DEVICE setting failure
    from srcs.metrics import build_metrics 
    metrics = build_metrics(**cfg.metrics)


    ## logger & writer
    logger = Logger(name='train',log_path=os.path.join(cfg.work_dir, 'log.txt'))
    if rank == 0:
        writer = TensorboardWriter(cfg.log_dir, True)
    else:
        writer = TensorboardWriter(cfg.log_dir, False)
    
    if rank==0:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        param_num = f'Trainable parameters: {sum([p.numel() for p in trainable_params])/1e6:.2f} M\n'
        dash_line = '=' * 80 + '\n'
        logger.info('Model info:\n'+ dash_line + 
                    str(model)+'\n'+ param_num +'\n'+                            dash_line)
    ## resume ckpt
    init_epoch = -1
    step = 0
    train_start_time = time.time()
    if cfg.train.resume is not None:
        if rank == 0:
            logger.info("----- Resuming From Checkpoint: %s -----" % cfg.train.resume)
        checkpoint = torch.load(cfg.train.resume)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']
    else:
        if rank == 0:
            logger.info("----- Starting New Training -----")

    if cfg.num_gpus > 1:
        model = DDP(model, device_ids=[rank]).to(device)


    ## train
    for epoch in range(init_epoch+1, cfg.train.num_epochs):
        epoch_start_time = time.time()      
        
        # train epoch
        for batch_idx, (img_noise, img_target) in enumerate(train_dataloader):
            step += 1
            
            # data to device
            img_noise, img_target = img_noise.to(device), img_target.to(device)
            
            # forward
            output = model(img_noise)

            # loss calc
            loss = criterion(output, img_target)

            # backward
            optim.zero_grad()
            loss.backward()

            # clip gradient and update
            optim.step()
        
            # metrics           
            if step % cfg.log.summary_interval == 0:
                iter_metrics = {}
                calc_metrics = metrics(torch.clip(output,0,1), img_target)
                calc_metrics.update({'loss': loss})

                for k, v in calc_metrics.items():
                    vv = collect(v) if cfg.num_gpus > 1 else v
                    iter_metrics.update({k: vv})
                # logger & writer
                metric_str = writer.writer_update(
                    step, '[train]', iter_metrics, {})
                
                if rank==0:
                    logger.info(
                    f'Train epoch: {epoch}/{cfg.train.num_epochs} [{batch_idx}/{len(train_dataloader)}] lr: {optim.param_groups[0]["lr"]:.4e}  | {metric_str}')
        
        # learning rate update
        if lr_scheduler is not None:
            lr_scheduler.step()

        # after epoch
        if rank == 0:
            # validate
            if epoch % cfg.log.validation_interval == 0:
                with torch.no_grad():
                    valid(model, val_dataloader, criterion, device, metrics,
                        epoch, logger, writer)
            # save checkpoint
            state = {
                'model': (model.module if cfg.num_gpus > 1 else model).state_dict(),
                'optim': optim.state_dict(),
                'step': step,
                'epoch': epoch
                }
            save_checkpoint(epoch, state, cfg.ckp_dir, logger=logger,
                            save_latest_k=cfg.log.save_latest_k, milestone_ckp=cfg.log.milestone_ckp)
            # log
            cur_time = time.time()
            logger.info(
                f"Epoch time: {(cur_time-epoch_start_time)/60:.2f} min, Total time: {(cur_time-train_start_time)/3600:.2f} h\n{'-'*80}\n")
    
def valid(model, val_dataloader, criterion, device, metrics, epoch, logger=None, writer=None):
    # validate
    model.eval()
    iter_metrics = {}
    with torch.no_grad():
        # valid iter
        for batch_idx, (img_noise, img_target) in enumerate(val_dataloader):
            img_noise, img_target = img_noise.to(
                device), img_target.to(device)

            # forward
            output = model(img_noise)
            # loss calc
            loss = criterion(output, img_target)
            # metrics
            calc_metrics = metrics(torch.clip(output,0,1), img_target)
            calc_metrics.update({'loss': loss})

            # average metric between processes
            for k, v in calc_metrics.items():
                iter_metrics.update({k: iter_metrics.get(k, 0)+v})

        # average iter metrics
        iter_metrics = {k: v/(batch_idx+1)
                        for k, v in iter_metrics.items()}
        # iter images
        image_tensors = {
            'input': img_noise[0:4, ...], 'img_target': img_target[0:4, ...], 'output': output[0:4, ...]}

        # logger & writer
        metric_str = writer.writer_update(
            epoch, '[valid]', iter_metrics, image_tensors)
        logger.info('-'*80 + f'\nValidate |  {metric_str}\n' + '-'*80)
