import torch
import pyiqa  # pip install pyiqa
import torch.nn as nn

# ===========================
# build metric
# ===========================

def build_metrics(met_name, calc_mean):
    return IQA_Metric(metric_names=met_name, calc_mean=calc_mean)

# ===========================
# metrics
# ===========================

class IQA_Metric(nn.Module):
    """image quality assessment metric calculation using [pyiqa package](https://github.com/chaofengc/IQA-PyTorch)
    Note: 
        - use `print(pyiqa.list_models())` to list all available metrics
        - the inputs and outputs are in 'torch tensor' format
    """

    def __init__(self, metric_names: str, calc_mean: bool = True):
        super(IQA_Metric, self).__init__()
        self.__names__ = metric_names
        self.metrics = {}
        for met_name in metric_names:
            self.metrics.update({met_name: pyiqa.create_metric(metric_name=met_name)})
        self.calc_mean = calc_mean

    def forward(self, output, target):
        with torch.no_grad():
            metric_scores = {}
            for met_name, met in self.metrics.items():
                metric_scores.update({met_name: met(output, target)})
        if self.calc_mean:
            metric_scores_mean = {}
            for met_name, met_score in metric_scores.items():
                metric_scores_mean.update({met_name: torch.mean(met_score)})
            return metric_scores_mean
        else:
            return metric_scores
