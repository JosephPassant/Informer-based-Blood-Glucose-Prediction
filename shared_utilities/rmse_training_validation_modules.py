import os
import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utilities import *
from metrics import *


def validate(val_loader, model):
    """
    Runs validation and computes RMSE & MAE at each time step.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    model.eval()
    total_rmse, total_mae = 0, 0
    all_outputs, all_targets = [], []

    print("Running validation...")
    with torch.no_grad():
        for batch_x, batch_dec, batch_y in val_loader:
            batch_x, batch_dec, batch_y = batch_x.to(device), batch_dec.to(device), batch_y.to(device)

            # Get predictions from the model

            outputs = model(batch_x, batch_dec)

            # Ensure target shape matches predictions
            batch_y = batch_y.unsqueeze(-1) if batch_y.ndim == 2 else batch_y  

            # Compute per-batch RMSE & MAE
            batch_rmse = rmse(outputs, batch_y)
            batch_mae = mae(outputs, batch_y)

            total_rmse += batch_rmse
            total_mae += batch_mae

            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    # Convert lists to numpy arrays
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute RMSE & MAE per time step
    rmse_per_timestep = np.sqrt(np.mean((all_outputs - all_targets) ** 2, axis=0))
    mae_per_timestep = np.mean(np.abs(all_outputs - all_targets), axis=0)

    # Compute overall RMSE & MAE
    avg_rmse = total_rmse / max(len(val_loader), 1)
    avg_mae = total_mae / max(len(val_loader), 1)

    print(f"Validation RMSE: {avg_rmse:.6f}")
    print(f"Validation MAE: {avg_mae:.6f}")

    print("\nPer-Time Step RMSE & MAE:")
    for i, (rmse_t, mae_t) in enumerate(zip(rmse_per_timestep.flatten(), mae_per_timestep.flatten())):
        print(f"Time Step {i+1}: RMSE={rmse_t:.6f}, MAE={mae_t:.6f}")

    return avg_rmse, avg_mae, rmse_per_timestep, mae_per_timestep

def train(train_iter, model, optimizer, lr_scheduler, config, train_loader, val_loader, save_dir):
    """
    Trains the model using batches from the training set.
    Implements early stopping based on validation loss.
    """
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    best_loss = float("inf")
    best_model_state_dict = None
    validation_loss_table = []

    for epoch in range(config.train_epochs):
        print(f"Epoch {epoch+1}/{config.train_epochs}")

        epoch_start_time = time.time()  

        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":6.6f")  # Higher precision
        progress = ProgressMeter(len(train_loader), [batch_time, losses], prefix=f"Epoch: [{epoch+1}]")

        model.train()
        epoch_loss = 0
        end = time.time()

        for batch_idx in range(config.iters_per_epoch):
            batch_x, batch_dec, batch_y = next(train_iter)
            batch_x, batch_dec, batch_y = batch_x.to(device), batch_dec.to(device), batch_y.to(device)

            optimizer.zero_grad()
            
            # print(f"batch_x shape: {batch_x.shape}")
            # print(f"batch_dec shape: {batch_dec.shape}")
            outputs = model(batch_x, batch_dec)

            # Ensure batch_y has the same shape as outputs
            batch_y = batch_y.unsqueeze(-1) if batch_y.ndim == 2 else batch_y  
            loss = rmse(outputs, batch_y)

            # Backward pass & optimization step
            loss.backward()
            optimizer.step()

            # Update batch tracking
            losses.update(loss.item(), batch_x.size(0))
            epoch_loss += loss.item()
            batch_time.update(time.time() - end)
            end = time.time()

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                progress.display(batch_idx)

        lr_scheduler.step()
        # Ensure the last update to progress meter is displayed
        progress.display(len(train_loader) - 1)

        avg_epoch_loss = epoch_loss / config.iters_per_epoch
        val_rmse, val_mae, rmse_per_timestep, mae_per_timestep = validate(val_loader, model)

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1} | Epoch Train Loss: {avg_epoch_loss:.6f} | Validation RMSE: {val_rmse:.6f} | Validation MAE: {val_mae:.6f}")

        validation_loss_table.append({
            "Epoch": epoch+1,
            "ValRMSE": val_rmse,
            "ValMAE": val_mae,
            "Time": epoch_time
        })

        if val_rmse < best_loss:
            best_loss = val_rmse
            best_model_state_dict = model.state_dict() 
            print(f"New best model found at epoch {epoch+1} with RMSE: {val_rmse:.6f}")


    if best_model_state_dict:
        best_model_path = os.path.join(save_dir, f"best_validation_model_rmse_{best_loss:.4f}.pth")
        torch.save(best_model_state_dict, best_model_path)
        print(f"Best model saved at {best_model_path} with RMSE: {best_loss:.6f}")
    else:
        best_model_path = None

    df_val_loss = pd.DataFrame(validation_loss_table)

    if df_val_loss is None or df_val_loss.empty:
        print("ERROR: validation_metrics_df is empty. Skipping save.")
    else:
        df_val_loss.to_csv(os.path.join(save_dir, "validation_metrics.csv"), index=False)
        print("Saved validation_metrics.csv")

    return  best_model_path    