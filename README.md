# A lightweight PyTorch template for deep learning projects.


This is a code template library for deep learning project development, which has the following features:

- clear code structure (single file for each pipeline element like model, loss, etc.)
- yaml based configuration system
- multiple GPU training (DDP)
- log & tensorboard


## Train

```bash
python train.py -c config/default_config.yaml 

Param:
-r: checkpoint path to resume
```

## Test

```bash
python test.py -c config/default_config.yaml -p xxx/model.pth
```