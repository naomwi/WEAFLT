import numpy as np
import torch
import torch.nn as nn

def metric(pred, true):
    MAE = np.mean(np.abs(pred - true))
    MSE = np.mean((pred - true) ** 2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs((pred - true) / true))
    
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    R2 = 1 - (ss_res / (ss_tot + 1e-7))
    
    return MAE, MAPE, RMSE, R2


class EventWeightedLoss(nn.Module):
    def __init__(self, event_weight=5.0):
        super(EventWeightedLoss, self).__init__()
        self.event_weight = event_weight
        self.mse = nn.MSELoss(reduction='none') 

    def forward(self, pred, target, event_flag):
        
        squared_errors = self.mse(pred, target)
        weights = 1.0 + (event_flag * (self.event_weight - 1.0))
        
        weighted_errors = squared_errors * weights
        return torch.mean(weighted_errors)
    
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss