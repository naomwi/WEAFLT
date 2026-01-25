import os
import numpy as np
import pandas as pd
import time
import torch
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from src.Utils.parameter import CONFIG,RUNTIME_LOG,STABILITY_LOG
from src.Utils.support_class import EarlyStopping
from src.Utils.path import OUTPUT_DIR
# --- config ---

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                epochs, device, model_name="Model",force_train = False):

    
    
    safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "")
    save_path = os.path.join(OUTPUT_DIR/'model', f"best_model_{safe_model_name}.pth")

    print(f" Checking model: {safe_model_name}...")

    if os.path.exists(save_path) and not force_train:
        print(f"Found existing model at '{save_path}'")
        print(f"Skipping training & Loading weights...")
        
        model.load_state_dict(torch.load(save_path, map_location=device))
        return model  
    
    print(f"Start training for {epochs} epochs...")
    
    stopper = EarlyStopping(patience=CONFIG['patience'])
    best_val_loss = float('inf')
    epoch_times = []

    for epoch in range(epochs):
        epoch_s = time.time()
        
        # --- TRAIN ---
        model.train()
        t_loss = 0
        for x, y, evt, _ in train_loader: 
            x, y, evt = x.to(device), y.to(device), evt.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y, evt)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            
        # --- VALIDATE ---
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y, evt, _ in val_loader:
                x, y, evt = x.to(device), y.to(device), evt.to(device)
                pred = model(x)
                v_loss += criterion(pred, y, evt).item()
        
        t_loss /= len(train_loader)
        v_loss /= len(val_loader)
        
        # Cập nhật Best Loss để hiển thị
        if v_loss < best_val_loss: 
            best_val_loss = v_loss

        # Gọi Early Stopping (Tự động lưu file nếu v_loss giảm)
        stopper(v_loss, model, save_path)
        
        # Tracking thời gian
        e_time = time.time() - epoch_s
        epoch_times.append(e_time)
        
        # Dừng sớm
        if stopper.early_stop: 
            print(f" Early Stop at Epoch {epoch+1}")
            break
        
        # In Log
        if (epoch+1) % 10 == 0 or epoch == 0: 
            print(f"      Ep {epoch+1}/{epochs}: Train={t_loss:.4f} Val={v_loss:.4f} Time={e_time:.2f}s")

    # Load lại weight tốt nhất sau khi train xong
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Training Finished. Best Val Loss: {best_val_loss:.4f}")
    else:
        print("Warning: No model saved (Loss never improved?)")

    # Log Runtime (Optional)
    if 'RUNTIME_LOG' in globals():
        globals()['RUNTIME_LOG'].append({
            "Stage": "Training",
            "Model": model_name,
            "Time_s": np.mean(epoch_times) if epoch_times else 0,
            "Unit": "s/epoch"
        })
        
    return model

def evaluate_model(model, loader, device, split_info):
    """
    Evaluate model với Inverse Log Transform
    is_log_target: True nếu biến mục tiêu đã bị log1p trước đó
    """
    model.eval()
    preds, actuals, events = [], [], []
    
    y_mean = split_info['scaler']['y_mean'].to(device).view(1, 1, -1)
    y_std = split_info['scaler']['y_std'].to(device).view(1, 1, -1)

    target_names = split_info.get('target_names',[])
    with torch.no_grad():
        for bx, by, bevent, _ in loader:
            bx, by, bevent = bx.to(device), by.to(device), bevent.to(device)
            
            out = model(bx) # Shape: [Batch, 24, 6]

            out_real = out * y_std + y_mean
            by_real = by * y_std + y_mean

            preds.append(out_real.cpu().numpy())
            actuals.append(by_real.cpu().numpy())
            if bevent.dim() == 3:
                events.append(bevent.cpu().numpy())
            else:
                events.append(bevent.unsqueeze(-1).cpu().numpy())
                    
            
    if len(preds) == 0: return {}
    preds_total = np.concatenate(preds, axis=0)
    actuals_total = np.concatenate(actuals, axis=0)
    events_total = np.concatenate(events, axis=0)

    for i, col_name in enumerate(target_names):
        if "_log" in col_name:
            preds_total[:, :, i] = np.expm1(preds_total[:, :, i])
            actuals_total[:, :, i] = np.expm1(actuals_total[:, :, i])
            preds_total[:, :, i] = np.maximum(preds_total[:, :, i], 0)

    preds = np.concatenate(preds_total).flatten()
    actuals = np.concatenate(actuals_total).flatten()
    events = np.concatenate(events_total).flatten()

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((actuals - preds) / (actuals + 1e-6))
        mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
        mape = np.mean(mape) * 100

    event_indices = np.where(events == 1)[0]
    
    if len(event_indices) > 0:
        sf_mae = mean_absolute_error(actuals[event_indices], preds[event_indices])
    else:
        sf_mae = 0.0

    return {
        "RMSE": rmse, 
        "MAE": mae, 
        "R2": r2, 
        "MAPE": mape,
        "SF_MAE": sf_mae, 
        "N_Events": len(event_indices)
    }