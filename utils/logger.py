import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from datetime import datetime
import logging, logging.config

def Logger(name=None, log_path='./runtime.log'):
    config_dict = {
        "version": 1,
        "formatters": {
            "simple": {
            "format": "%(message)s"
            },
            "detailed": {
            "format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
            }
        },
        "handlers": {
            "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
            },
            "file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "filename": log_path
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        },
        "disable_existing_loggers": False
    }
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)
    return logger 

class TensorboardWriter():
    def __init__(self, log_dir, enabled=True):
        self.writer = SummaryWriter(log_dir) if enabled else None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

        self.step = 0

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding', 'add_hparams'
        }
        self.timer = datetime.now()

    def set_step(self, step, speed_chk=None):  # phases = 'train'|'valid'|None
        self.step = step
        # measure the calculation speed by call this function between 2 steps (steps_per_sec)
        if speed_chk and step != 0:
            duration = datetime.now() - self.timer
            self.add_scalar(f'steps_per_sec/{speed_chk}',
                            1 / duration.total_seconds())
        self.timer = datetime.now()


    def writer_update(self, step, phase, metrics, image_tensors=None):
        # hook after iter
        self.set_step(step, speed_chk=f'{phase}')

        metric_str = ''
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.add_scalar(f'{phase}/{k}', v)
                metric_str += f'{k}: {v:8.5f} '

        if image_tensors:
            for k, v in image_tensors.items():
                self.add_image(
                    f'{phase}/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))

        return metric_str  # metric string for logger

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(var1, var2, *args, **kwargs):
                if add_data is not None:
                    add_data(var1, var2, self.step, *args, **kwargs)
            return wrapper
        else:
            attr = getattr(self.writer, name, None)
            if not attr:
                raise AttributeError('unimplemented attribute')
            return attr
