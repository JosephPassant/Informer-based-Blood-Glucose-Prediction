import os
import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utilities import *
from metrics import *
from dual_weighted_loss_function import *


def validate(val_loader, model, include_cg_ega=True, mean=152.91051040286524, std=70.27050122812615):
    """
    Runs validation and computes RMSE, MAE, and CG-EGA metrics.
    Modified to use float32 for MPS compatibility.
    """

        # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model.eval()
    total_rmse, total_mae = 0, 0
    all_outputs, all_targets = [], []
    
    # Initialize CG-EGA tracking counters
    hypo_stats = {"AP": 0, "BE": 0, "EP": 0, "count": 0}
    eu_stats = {"AP": 0, "BE": 0, "EP": 0, "count": 0}
    hyper_stats = {"AP": 0, "BE": 0, "EP": 0, "count": 0}
    
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
            
            # Calculate CG-EGA metrics
            if include_cg_ega:
                # Move to CPU first, then detach
                outputs_cpu = outputs.cpu().detach()
                batch_y_cpu = batch_y.cpu().detach()
                
                pred_glucose = outputs_cpu * std + mean
                target_glucose = batch_y_cpu * std + mean
                
                # Process each sample in batch
                for i in range(len(outputs)):
                    # Convert to numpy with explicit float32 type
                    y_pred = pred_glucose[i].numpy().astype(np.float32).flatten()
                    y_true = target_glucose[i].numpy().astype(np.float32).flatten()
                    
                    # Calculate derivatives with explicit float32
                    dy_pred = np.zeros_like(y_pred, dtype=np.float32)
                    dy_true = np.zeros_like(y_true, dtype=np.float32)
                    
                    if len(y_pred) > 1:
                        dy_pred[1:] = (y_pred[1:] - y_pred[:-1]) / np.float32(5.0)
                        dy_true[1:] = (y_true[1:] - y_true[:-1]) / np.float32(5.0)
                    
                    # Calculate CG-EGA
                    try:
                        cg_ega = CG_EGA_Loss(y_true, dy_true, y_pred, dy_pred, freq=5)
                        results, counts = cg_ega.simplified()
                        
                        # Update counters
                        for region in ["hypo", "eu", "hyper"]:
                            if counts[f"count_{region}"] > 0:
                                for category in ["AP", "BE", "EP"]:
                                    locals()[f"{region}_stats"][category] += results[f"{category}_{region}"]
                                locals()[f"{region}_stats"]["count"] += counts[f"count_{region}"]
                    except Exception as e:
                        print(f"CG-EGA calculation failed: {str(e)}. Skipping this sample.")

    # Convert lists to numpy arrays
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute RMSE & MAE per time step
    rmse_per_timestep = np.sqrt(np.mean((all_outputs - all_targets) ** 2, axis=0))
    mae_per_timestep = np.mean(np.abs(all_outputs - all_targets), axis=0)

    # Compute overall RMSE & MAE
    avg_rmse = total_rmse / max(len(val_loader), 1)
    avg_mae = total_mae / max(len(val_loader), 1)

    # Print validation metrics
    print(f"Validation RMSE: {avg_rmse:.6f}")
    print(f"Validation MAE: {avg_mae:.6f}")

    # Calculate and print CG-EGA metrics
    total_samples = hypo_stats["count"] + eu_stats["count"] + hyper_stats["count"]
    total_AP = hypo_stats["AP"] + eu_stats["AP"] + hyper_stats["AP"]
    total_BE = hypo_stats["BE"] + eu_stats["BE"] + hyper_stats["BE"]
    total_EP = hypo_stats["EP"] + eu_stats["EP"] + hyper_stats["EP"]
    
    # Calculate percentages (avoid division by zero)
    ap_percent = (total_AP / total_samples * 100) if total_samples > 0 else 0
    be_percent = (total_BE / total_samples * 100) if total_samples > 0 else 0
    ep_percent = (total_EP / total_samples * 100) if total_samples > 0 else 0
    
    # Print per-region CG-EGA stats
    if include_cg_ega:
        print("\nCG-EGA Results:")
        for region, stats in [("Hypo", hypo_stats), ("Eu", eu_stats), ("Hyper", hyper_stats)]:
            if stats["count"] > 0:
                ap_pct = stats["AP"] / stats["count"] * 100
                be_pct = stats["BE"] / stats["count"] * 100
                ep_pct = stats["EP"] / stats["count"] * 100
                print(f"  {region}: AP={ap_pct:.1f}%, BE={be_pct:.1f}%, EP={ep_pct:.1f}% (n={stats['count']})")
        
        print(f"Overall: AP={ap_percent:.1f}%, BE={be_percent:.1f}%, EP={ep_percent:.1f}% (n={total_samples})")

    # Print per-time step RMSE & MAE
    print("\nPer-Time Step RMSE & MAE:")
    for i, (rmse_t, mae_t) in enumerate(zip(rmse_per_timestep.flatten(), mae_per_timestep.flatten())):
        print(f"Time Step {i+1}: RMSE={rmse_t:.6f}, MAE={mae_t:.6f}")

    # Create a dictionary with all CG-EGA statistics
    cg_ega_stats = {
        "hypo": hypo_stats,
        "eu": eu_stats,
        "hyper": hyper_stats,
        "total_samples": total_samples,
        "ap_percent": ap_percent,
        "be_percent": be_percent, 
        "ep_percent": ep_percent
    }
    
    return ep_percent, avg_rmse, avg_mae, rmse_per_timestep, mae_per_timestep, cg_ega_stats



def train(train_iter, model, optimizer, lr_scheduler, config, ap_weight, be_weight, ep_weight, hypo_multiplier, train_loader, val_loader, save_dir):
    """
    Trains the model using batches from the training set.
    Implements model selection based on lowest percentage of erroneous predictions.
    """

        # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    best_ep_percent = float("inf")  # Lower is better for erroneous prediction percentage
    best_rmse = float("inf")
    best_model_state_dict = None
    validation_metrics_table = []

    # Create loss function once
    loss_fn = CGEGALoss(AP_weight=ap_weight, BE_weight=be_weight, EP_weight=ep_weight, hypo_multiplier=hypo_multiplier)
    print(f"Using CG-EGA Loss with AP={ap_weight}, BE={be_weight}, EP={ep_weight}, Hypo={hypo_multiplier}")
    for epoch in range(config.train_epochs):
        print(f"Epoch {epoch+1}/{config.train_epochs}")

        epoch_start_time = time.time()  

        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":6.6f")
        progress = ProgressMeter(len(train_loader), [batch_time, losses], prefix=f"Epoch: [{epoch+1}]")

        model.train()
        epoch_loss = 0
        end = time.time()

        for batch_idx in range(config.iters_per_epoch):
            batch_x, batch_dec, batch_y = next(train_iter)
            batch_x, batch_dec, batch_y = batch_x.to(device), batch_dec.to(device), batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_x, batch_dec)

            # Ensure batch_y has the same shape as outputs
            batch_y = batch_y.unsqueeze(-1) if batch_y.ndim == 2 else batch_y  
            loss = loss_fn(outputs, batch_y)

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
        progress.display(len(train_loader) - 1)

        avg_epoch_loss = epoch_loss / config.iters_per_epoch
        
        # Run validation with CG-EGA metrics
        ep_percent, val_rmse, val_mae, rmse_per_timestep, mae_per_timestep, cg_ega_stats = validate(val_loader, model)

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1} | Train Loss: {avg_epoch_loss:.6f} | EP%: {ep_percent:.2f}% | RMSE: {val_rmse:.6f}")

        # Record validation metrics
        validation_metrics_table.append({
            "Epoch": epoch+1,
            "ValRMSE": val_rmse.item() if isinstance(val_rmse, torch.Tensor) else val_rmse,
            "ValMAE": val_mae.item() if isinstance(val_mae, torch.Tensor) else val_mae,
            "EP_Percent": ep_percent,
            "AP_Percent": cg_ega_stats["ap_percent"],
            "BE_Percent": cg_ega_stats["be_percent"],
            "Time": epoch_time
        })

        # Model selection based on erroneous prediction percentage
        if ep_percent < best_ep_percent:
            best_ep_percent = ep_percent
            best_model_state_dict = model.state_dict() 
            print(f"New best model found at epoch {epoch+1} with EP%: {ep_percent:.2f}% (RMSE: {val_rmse:.6f})")


    if best_model_state_dict:
        best_model_path = os.path.join(save_dir, f"best_validation_model_rmse_{best_ep_percent:.2f}.pth")
        torch.save(best_model_state_dict, best_model_path)
        print(f"Best model saved at {best_model_path} with RMSE: {best_ep_percent:.2f}")
    else:
        best_model_path = None

    df_val_loss = pd.DataFrame(validation_metrics_table)

    if df_val_loss is None or df_val_loss.empty:
        print("ERROR: validation_metrics_df is empty. Skipping save.")
    else:
        df_val_loss.to_csv(os.path.join(save_dir, "validation_metrics.csv"), index=False)
        print("Saved validation_metrics.csv")

    return  best_model_path     