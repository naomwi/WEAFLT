import torch 
import torch.nn as nn

class EventWeightedMSE(nn.Module):
    """Event-Weighted Loss (used for ALL models in fair comparison)"""
    def __init__(self, alpha=3.0):
        super(EventWeightedMSE, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, is_event):
        loss = self.mse(pred, target)
        weights = 1.0 + (is_event * (self.alpha - 1.0))
        return (loss * weights).mean()
    
class EarlyStopping:
    """Early Stopping Class (Standard Implementation)"""
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)