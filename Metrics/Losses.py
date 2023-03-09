import torch
from torch.nn import Module
import numpy as np
import pandas as pd


class QuantileLoss(Module):
    '''source: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629'''

    def __init__(self, quantiles, device):
        super().__init__()
        self.quantiles = quantiles
        self.device = device

    def forward(self, preds, target):
        """preds: tensor of shape (batch, num_horizons, num_quantiles)
        target: tensor of shape (batch, num_horizons"""
        if self.device == "gpu":
            preds = preds.to("cuda")
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        preds = preds.to("cuda")
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        losses = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1), dim=1)
        total_loss = torch.mean(losses)
        return total_loss, losses


def get_upper_bond(true_value,upper_bonds ):
    in_range = (true_value <= upper_bonds)
    if isinstance(in_range, torch.Tensor):
        return in_range.float().mean()
    else:
        return in_range.mean()
    
    
def get_calibration(true_value, preds, quantiles_name=None, device = None):
    if isinstance(preds, torch.Tensor) and device == "gpu":
        preds = preds.to("cuda")
        true_value = true_value.to("cuda")
    calibration_by_quantile = {}
    if quantiles_name is None:
        quantiles_name = [float(c) for c in preds.columns]
    for c in quantiles_name:
        if isinstance(preds,pd.DataFrame) :

            upper_bonds = preds[str(c)]
            calibration_by_quantile[c]= get_upper_bond(true_value,upper_bonds)
        else :
            upper_bonds = preds[:,quantiles_name.index(c)]
            calibration_by_quantile[f'qt_{c}']= get_upper_bond(true_value, upper_bonds)
            
    return calibration_by_quantile

def get_sharpness_score_by_alpha(true_value, lower_bonds, upper_bonds, alpha,calibration_penalty=True, weighted=True):
        true_value_cpu = true_value.cpu().detach()
        
        sharpness = (upper_bonds - lower_bonds) if not weighted else (upper_bonds-lower_bonds)/true_value_cpu
        indicator_lower_bond = (true_value_cpu <lower_bonds).float()
        indicator_upper_bond = (true_value_cpu >upper_bonds).float()
        lower_bond_pen = (lower_bonds-true_value_cpu) if not weighted else  (lower_bonds-true_value_cpu)/true_value_cpu
        upper_bond_pen = (true_value_cpu-upper_bonds) if not weighted else  (true_value_cpu-upper_bonds)/true_value_cpu
        penalty_terms = 2/alpha *(lower_bond_pen*indicator_lower_bond  + upper_bond_pen*indicator_upper_bond)
        sharpness_score = sharpness + penalty_terms if calibration_penalty else sharpness
        return sharpness_score.abs().mean()


def get_weighted_interval_score(sharpness_values, weights, true_value, median ):
    '''Based on https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618''' 
    true_value_cpu = true_value.cpu().detach()
    weighted_sharp_median = 0.5 * (true_value_cpu-median).abs().mean()
    sharpness_score_weighted = (sum([sharp * weight for sharp, weight in zip(sharpness_values, weights)])+weighted_sharp_median)/(len(weights)+1/2)
    return sharpness_score_weighted
    
def get_sharpnesses(true_value, preds, quantiles_name=None, device = None, calibration_penalty = True,  weighted=True):
    if isinstance(preds, torch.Tensor) and device == 'gpu' :
        preds = preds.to("cuda")
        true_value = true_value.to("cuda")
    sharpness_by_quantile = {}
    if quantiles_name is None:
        quantiles_name = [float(c) for c in preds.columns]
    couples = [(c, 1-c) for c in quantiles_name[:len(quantiles_name)//2] if (c in quantiles_name and (1-c) in quantiles_name)]
    alphas = [c_low*2 for c_low, _ in couples]
    weights = np.array(alphas)/2
    for  ind_alpha, (c_low, c_up) in enumerate(couples):
        alpha = alphas[ind_alpha]
        if  isinstance(preds,pd.DataFrame) :
            pass
        else :
            lower_bonds = preds[:,:,quantiles_name.index(c_low)]
            upper_bonds = preds[:,:,quantiles_name.index(c_up)]
            median = preds[:,:, quantiles_name.index(0.5)]
            sharpness_score = get_sharpness_score_by_alpha(true_value, lower_bonds, upper_bonds,alpha=alpha ,calibration_penalty=calibration_penalty, weighted=weighted )
            sharpness_by_quantile[f'qt_{c_low}']= sharpness_score
        sharpness_score_weighted = get_weighted_interval_score(sharpness_by_quantile.values(), weights, true_value, median  )
    return sharpness_by_quantile, sharpness_score_weighted

