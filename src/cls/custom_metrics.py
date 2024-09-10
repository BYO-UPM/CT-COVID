import torch
import numpy as np
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.aggregation import BaseAggregator
from bayesian_torch.utils.avuc_loss import eval_avu, accuracy_vs_uncertainty

class AvU(Metric):
    higher_is_better = True

    def __init__(self, uncertainty_threshold, **kwargs):
        super().__init__(**kwargs)
        self.uncertainty_threshold = uncertainty_threshold
        self.add_state("n_ac", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_au", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_ic", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_iu", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target, uncertainty):
        preds, target, uncertainty = self._input_format(preds, target, uncertainty)
        assert preds.shape == target.shape and preds.shape == uncertainty.shape 
        self.n_ac+= torch.logical_and(torch.eq(preds, target), torch.lt(uncertainty, self.uncertainty_threshold)).sum()
        self.n_au+= torch.logical_and(torch.eq(preds, target), torch.ge(uncertainty, self.uncertainty_threshold)).sum()
        self.n_ic+= torch.logical_and(torch.ne(preds, target), torch.lt(uncertainty, self.uncertainty_threshold)).sum()
        self.n_iu+= torch.logical_and(torch.ne(preds, target), torch.ge(uncertainty, self.uncertainty_threshold)).sum()

    def compute(self):
        return (self.n_ac + self.n_iu) / (self.n_ac + self.n_au + self.n_ic + self.n_iu)
    
class CorrectUncertaintiesConcatenator(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("value", default=[], dist_reduce_fx="cat")

    def update(self, preds_t, target_t, uncertainty_t) -> None:
        self.value.append(uncertainty_t[preds_t == target_t])

    def compute(self):
        """Compute the aggregated value."""
        if isinstance(self.value, list) and self.value:
            return dim_zero_cat(self.value)
        return self.value
    
class IncorrectUncertaintiesConcatenator(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("value", default=[], dist_reduce_fx="cat")
    
    def update(self, preds_t, target_t, uncertainty_t) -> None:
        self.value.append(uncertainty_t[preds_t != target_t])
    
    def compute(self):
        """Compute the aggregated value."""
        if isinstance(self.value, list) and self.value:
            return dim_zero_cat(self.value)
        return self.value
    
