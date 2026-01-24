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
        weights = torch.ones_like(loss)
        weights[is_event == 1] = self.alpha
        return (loss * weights).mean()
    
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience; self.counter = 0; self.best_score = None; self.early_stop = False
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None: self.best_score = score; torch.save(model.state_dict(), path)
        elif score < self.best_score: self.counter += 1; self.early_stop = (self.counter >= self.patience)
        else: self.best_score = score; torch.save(model.state_dict(), path); self.counter = 0
