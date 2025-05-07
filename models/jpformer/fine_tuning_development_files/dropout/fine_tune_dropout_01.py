import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import transformers
import shutil
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch import optim, nn
from sklearn.metrics import mean_squared_error
from transformers import get_cosine_schedule_with_warmup
from datetime import datetime


current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../../../"))


sys.path.append(os.path.join(PROJECT_ROOT, "shared_utilities"))
from utilities import *
from dual_weighted_loss_function import *
from dual_weighted_loss_training_validation_modules import *

from evaluation_for_no_feature_enhancement import *

sys.path.append(os.path.join(PROJECT_ROOT, "models/jpformer"))
from jpformer import JPFormer

# set random seed for torch and numpy and all other

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



"""
=================================================
CONFIGURATION LOADING FUNCTIONS
=================================================
"""

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r") as file:
        return json.load(file)

class ConfigObject:
    """
    Convert a dictionary to an object with attributes.
    """
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

"""
=================================================
MODEL AND ENVIRONMENT SETUP
=================================================
"""

def setup_device(config):
    """
    Set up and return the appropriate computation device based on availability.
    
    Args:
        config: Configuration object with device preferences
        
    Returns:
        torch.device: The selected computation device
    """
    if torch.cuda.is_available() and config.use_gpu:
        device = torch.device(f"cuda:{config.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon (M1/M2)
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def load_model(config, device, pretrained_weights_path=None):
    """
    Initialize and load a model with optional pretrained weights.
    
    Args:
        config: Configuration object with model parameters
        device: Computation device
        pretrained_weights_path: Optional path to pretrained model weights
        
    Returns:
        model: Initialized model
    """
    model = JPFormer(
        enc_in=config.enc_in,
        dec_in=config.dec_in,
        c_out=config.c_out,
        seq_len=config.seq_len,
        label_len=config.label_len,
        out_len=config.pred_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_layers=config.d_layers,
        d_ff=config.d_ff,
        factor=config.factor,
        dropout=config.dropout,
        embed=config.embed,
        activation=config.activation,
        output_attention=config.output_attention,
        mix=config.mix,
        device=device
    ).float().to(device)

    # Debugging: Check model name
    model_name = model.__class__.__name__
    print(f"Initialized model: {model_name}")

    # Load pre-trained weights if provided
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        try:
            model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
            print(f"Successfully loaded pretrained weights from: {pretrained_weights_path}")
        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            print("Continuing with random initialization...")
    else:
        print(f"Pretrained weights file not found or not provided. Using random initialization.")

    return model, model_name

"""
=================================================
DATA LOADING FUNCTIONS FOR PATIENT-SPECIFIC DATA
=================================================
"""

def get_patient_data_paths(ptid, project_root):
    """
    Generate data paths for a specific patient.
    
    Args:
        ptid: Patient ID
        project_root: Root project directory
        
    Returns:
        dict: Dictionary containing paths for training, validation, and testing data
    """

    paths = {
        "train": {
            "encoder": os.path.join(project_root, f"data/processed_data/ohio/training/over_and_under_sampling/ohio_training_{ptid}/EncoderSlices"),
            "decoder": os.path.join(project_root, f"data/processed_data/ohio/training/over_and_under_sampling/ohio_training_{ptid}/DecoderSlices"),
            "target": os.path.join(project_root, f"data/processed_data/ohio/training/over_and_under_sampling/ohio_training_{ptid}/TargetSlices")
        },
        "val": {
            "encoder": os.path.join(project_root, f"data/processed_data/ohio/validation/ohio_validation_{ptid}/EncoderSlices"),
            "decoder": os.path.join(project_root, f"data/processed_data/ohio/validation/ohio_validation_{ptid}/DecoderSlices"),
            "target": os.path.join(project_root, f"data/processed_data/ohio/validation/ohio_validation_{ptid}/TargetSlices")
        },
        "test": {
            "encoder": os.path.join(project_root, f"data/processed_data/ohio/testing/ohio_test_{ptid}/EncoderSlices"),
            "decoder": os.path.join(project_root, f"data/processed_data/ohio/testing/ohio_test_{ptid}/DecoderSlices"),
            "target": os.path.join(project_root, f"data/processed_data/ohio/testing/ohio_test_{ptid}/TargetSlices")
        }
    }
    # Verify existence of directories
    for split, split_paths in paths.items():
        for path_type, path in split_paths.items():
            if not os.path.exists(path):
                print(f"Warning: {split} {path_type} path does not exist: {path}")
    
    return paths

def load_patient_datasets(patient_paths, batch_size, num_workers):
    """
    Load datasets and create data loaders for a specific patient.
    
    Args:
        patient_paths: Dictionary of patient-specific data paths
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        dict: Dictionary containing data loaders and iterators
    """
    # Create datasets
    train_dataset = BloodGlucoseDataset(
        patient_paths["train"]["encoder"], 
        patient_paths["train"]["decoder"], 
        patient_paths["train"]["target"]
    )
    
    val_dataset = BloodGlucoseDataset(
        patient_paths["val"]["encoder"], 
        patient_paths["val"]["decoder"], 
        patient_paths["val"]["target"]
    )
    
    test_dataset = BloodGlucoseDataset(
        patient_paths["test"]["encoder"], 
        patient_paths["test"]["decoder"], 
        patient_paths["test"]["target"]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=num_workers
    )
    
    # Create infinite iterator for training
    train_iter = ForeverDataIterator(train_loader)
    
    data_loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_iter": train_iter
    }
    
    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return data_loaders

"""
=================================================
VALIDATION FUNCTION
=================================================
"""

def validate(val_loader, model, device, include_cg_ega=True, mean=153.1055921182904, std=70.26895577737373):
    """
    Runs validation and computes RMSE, MAE, and CG-EGA metrics.
    Modified to use float32 for MPS compatibility.
    
    Args:
        val_loader: DataLoader for validation data
        model: Model to validate
        device: Computation device
        include_cg_ega: Whether to include CG-EGA metrics
        mean: Mean value for denormalization
        std: Standard deviation for denormalization
        
    Returns:
        tuple: EP percentage, RMSE, MAE, timestep metrics, and CG-EGA statistics
    """
    model.eval()
    total_rmse, total_mae = 0, 0
    all_outputs, all_targets = [], []
    
    # Initialize CG-EGA tracking counters
    hypo_stats = {"AP": 0, "BE": 0, "EP": 0, "count": 0}
    eu_stats = {"AP": 0, "BE": 0, "EP": 0, "count": 0}
    hyper_stats = {"AP": 0, "BE": 0, "EP": 0, "count": 0}
    
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

"""
=================================================
TRAINING FUNCTION
=================================================
Dynamic iterations per epoch: The training function now calculates 
iterations based on each patient's dataset size rather than using 
a fixed value from the config. This ensures:
1. Each patient's entire dataset is processed exactly once per epoch
2. Training adapts to varying dataset sizes between patients
3. Learning rate scheduling is properly scaled to dataset size
"""

def train_patient_model(
    train_iter, model, optimizer, lr_scheduler, 
    config, device, patient_id,
    data_loaders, ap_weight, be_weight, ep_weight, hypo_multiplier,
    fine_tuning_dir):  # Add fine_tuning_dir as parameter
    """
    Trains the model for a specific patient using batches from the training set.
    Tests the model every second epoch to track progress.
    
    Args:
        train_iter: Iterator for training data
        model: Model to train
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        config: Configuration object
        device: Computation device
        patient_id: Patient ID
        data_loaders: Dictionary of data loaders
        ap_weight: Weight for Accurate Prediction in CG-EGA Loss
        be_weight: Weight for Benign Error in CG-EGA Loss
        ep_weight: Weight for Erroneous Prediction in CG-EGA Loss
        hypo_multiplier: Multiplier for hypoglycemia in CG-EGA Loss
        fine_tuning_dir: Directory to save fine-tuning results
        
    Returns:
        tuple: Best model path, validation metrics DataFrame, test metrics DataFrame
    """
    best_ep_percent = float("inf")  # Lower is better for erroneous prediction percentage
    best_rmse = float("inf")
    best_model_state_dict = None
    validation_metrics_table = []
    test_metrics_table = []

    # Create loss function once
    loss_fn = CGEGALoss(AP_weight=ap_weight, BE_weight=be_weight, EP_weight=ep_weight, hypo_multiplier=hypo_multiplier)
    print(f"Using CG-EGA Loss with AP={ap_weight}, BE={be_weight}, EP={ep_weight}, Hypo={hypo_multiplier}")
    
    # Calculate dynamic iterations per epoch based on dataset size
    dataset_size = len(data_loaders["train_loader"].dataset)
    iterations_per_epoch = max(1, dataset_size // config.batch_size)
    
    print(f"Dynamic iterations per epoch: {iterations_per_epoch} (dataset size: {dataset_size}, batch size: {config.batch_size})")
    
    for epoch in range(config.train_epochs):
        print(f"Epoch {epoch+1}/{config.train_epochs}")

        epoch_start_time = time.time()  

        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":6.6f")
        progress = ProgressMeter(iterations_per_epoch, [batch_time, losses], prefix=f"Epoch: [{epoch+1}]")

        model.train()
        epoch_loss = 0
        end = time.time()

        for batch_idx in range(iterations_per_epoch):
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
        progress.display(iterations_per_epoch - 1)

        avg_epoch_loss = epoch_loss / iterations_per_epoch
        
        # Run validation with CG-EGA metrics
        ep_percent, val_rmse, val_mae, rmse_per_timestep, mae_per_timestep, cg_ega_stats = validate(
            data_loaders["val_loader"], model, device
        )

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1} | Train Loss: {avg_epoch_loss:.6f} | EP%: {ep_percent:.2f}% | RMSE: {val_rmse:.6f}")

        # Record validation metrics
        validation_metrics_table.append({
            "Patient_ID": patient_id,
            "Epoch": epoch+1,
            "ValRMSE": val_rmse.item() if isinstance(val_rmse, torch.Tensor) else val_rmse,
            "ValMAE": val_mae.item() if isinstance(val_mae, torch.Tensor) else val_mae,
            "EP_Percent": ep_percent,
            "AP_Percent": cg_ega_stats["ap_percent"],
            "BE_Percent": cg_ega_stats["be_percent"],
            "Hypo_EP_Percent": cg_ega_stats["hypo"]["EP"] / max(cg_ega_stats["hypo"]["count"], 1) * 100,
            "Time": epoch_time
        })

        # Model selection based on erroneous prediction percentage
        if ep_percent < best_ep_percent:
            best_ep_percent = ep_percent
            best_rmse = val_rmse
            best_model_state_dict = model.state_dict() 
            print(f"New best model found at epoch {epoch+1} with EP%: {ep_percent:.2f}% (RMSE: {val_rmse:.6f})")

        # Run testing every second epoch or on the last epoch
        if (epoch+1) % 2 == 0 or epoch+1 == config.train_epochs:
            print(f"\n-----------------Testing at epoch {epoch+1}-----------------")
            tester = ModelTester(model, data_loaders["test_loader"], device)
            test_rmse, test_mae, stats_tables, _ = tester.test()
            
            # Extract test CG-EGA metrics
            test_ep_percent = 0
            test_ap_percent = 0
            test_be_percent = 0
            test_hypo_ep_percent = 0
            
            try:
                # Extract metrics from stats_tables
                overall_table = stats_tables["overall"]
                overall_stats = overall_table.iloc[-1]  # Get overall stats row
                test_ap_percent = float(overall_stats["AP%"])
                test_be_percent = float(overall_stats["BE%"])
                test_ep_percent = float(overall_stats["EP%"])
                
                # Extract hypoglycemia stats
                for i, row in overall_table.iterrows():
                    if row["Region"] == "hypo":
                        test_hypo_ep_percent = float(row["EP%"])
                        break
            except Exception as e:
                print(f"Error extracting test CG-EGA metrics: {str(e)}")
            
            # Record test metrics
            test_metrics_table.append({
                "Patient_ID": patient_id,
                "Epoch": epoch+1,
                "TestRMSE": test_rmse,
                "TestMAE": test_mae,
                "Test_EP_Percent": test_ep_percent,
                "Test_AP_Percent": test_ap_percent,
                "Test_BE_Percent": test_be_percent,
                "Test_Hypo_EP_Percent": test_hypo_ep_percent
            })
            
            print(f"Test RMSE: {test_rmse:.6f}")
            print(f"Test MAE: {test_mae:.6f}")
            print(f"Test EP%: {test_ep_percent:.2f}%")
            print("---------------------------------------------------------\n")

    # Save best model
    best_model_path = None
    if best_model_state_dict:
        save_dir = os.path.join(PROJECT_ROOT, "models", "jpformer","fine_tuning_development_files", "dropout", "01", f"patient_{patient_id}")
        create_dir(save_dir)

        best_model_path = os.path.join(save_dir, f"patient_{patient_id}_EP_{best_ep_percent:.1f}_RMSE_{best_rmse:.4f}.pth")
        torch.save(best_model_state_dict, best_model_path)
        print(f"Best model saved at {best_model_path} with EP%: {best_ep_percent:.2f}% (RMSE: {best_rmse:.6f})")

    df_val_metrics = pd.DataFrame(validation_metrics_table)
    df_test_metrics = pd.DataFrame(test_metrics_table)

    return best_model_path, df_val_metrics, df_test_metrics

"""
=================================================
UTILITY FUNCTIONS
=================================================
"""

def create_dir(directory):
    """
    Safely creates a directory if it does not exist.
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def setup_optimizer_scheduler(model, config, num_batches):
    """
    Set up optimizer and learning rate scheduler.
    
    Args:
        model: Model for which to create optimizer
        config: Configuration object
        num_batches: Number of batches per epoch (dynamically calculated)
        
    Returns:
        tuple: Optimizer and learning rate scheduler
    """
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    
    # Use dynamically calculated number of batches instead of fixed config value
    total_steps = num_batches * config.train_epochs
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    print(f"Learning rate scheduler set up with {total_steps} total steps ({num_batches} batches × {config.train_epochs} epochs)")
    
    return optimizer, lr_scheduler

"""
=================================================
MAIN EXECUTION FUNCTION
=================================================
"""

def main():
    # Load configuration
    config_path = os.path.join(PROJECT_ROOT, "models/jpformer/fine_tuning_development_files/dropout/dropout_01_config.json")
    config_dict = load_config(config_path)
    config = ConfigObject(config_dict)
    
    # Setup device
    device = setup_device(config)
    
    # Patient IDs
    patient_ids = [540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596]
    
    # Pretrained weights path
    pretrained_weights_path = os.path.join(PROJECT_ROOT, "models/jpformer/final_model_training_files/jpformer_dual_weighted_rmse_loss_func_high_dim_4_enc_lyrs_high_dropout_0.5696_MAE_0.3965.pth")
    
    # Loss function weights

    ap_weight = 1
    be_weight = 2
    ep_weight = 6
    hypo_multiplier = 1.5
    
    # Create results directory
    results_dir = os.path.join(PROJECT_ROOT, "models", "jpformer","fine_tuning_development_files", "dropout", "01")
    create_dir(results_dir)
    
    # Initialize combined results dataframes
    all_validation_results = []
    all_test_results = []

    
    # Process each patient
    for patient_id in patient_ids:
        print(f"\n{'='*30} PATIENT {patient_id} {'='*30}\n")
        
        # Get patient-specific data paths
        patient_paths = get_patient_data_paths(patient_id, PROJECT_ROOT)
        
        # Load patient datasets and create data loaders
        data_loaders = load_patient_datasets(patient_paths, config.batch_size, config.num_workers)
        
        # Initialize model with pretrained weights
        model, model_name = load_model(config, device, pretrained_weights_path)
        
        # Calculate dynamic iterations per epoch
        dataset_size = len(data_loaders["train_loader"].dataset)
        iterations_per_epoch = max(1, dataset_size // config.batch_size)
        print(f"Patient {patient_id} dataset size: {dataset_size}, batch size: {config.batch_size}")
        print(f"Dynamic iterations per epoch: {iterations_per_epoch}")
        
        # Setup optimizer and scheduler with dynamic iterations
        optimizer, lr_scheduler = setup_optimizer_scheduler(model, config, iterations_per_epoch)
        
        # Create patient-specific directories
        patient_dir = os.path.join(results_dir, f"patient_{patient_id}")
        create_dir(patient_dir)
        
        # Create base model and fine-tuning evaluation directories

        fine_tuning_dir = os.path.join(patient_dir, "fine_tuning_eval")

        create_dir(fine_tuning_dir)
        
        # Run initial validation on the pretrained model
        print("-------------Initial Validation of Pretrained Model for Patient-------------")
        ep_percent, val_rmse, val_mae, _, _, cg_ega_stats = validate(data_loaders["val_loader"], model, device)
        print(f"Pretrained model for Patient {patient_id} | EP%: {ep_percent:.2f}% | RMSE: {val_rmse:.6f} | MAE: {val_mae:.6f}")
        
        
        # Train the model for this patient
        print(f"-------------Training for Patient {patient_id}-------------")
        best_model_path, val_metrics_df, test_metrics_df = train_patient_model(
            data_loaders["train_iter"], model, optimizer, lr_scheduler, 
            config, device, patient_id, data_loaders,
            ap_weight, be_weight, ep_weight, hypo_multiplier,
            fine_tuning_dir  # Pass fine_tuning_dir to the function
        )
        
        # Extend combined results
        all_validation_results.append(val_metrics_df)
        all_test_results.append(test_metrics_df)
        
        # Save patient-specific results
        patient_dir = os.path.join(results_dir, f"patient_{patient_id}")
        create_dir(patient_dir)
        
        # Adapt file saving paths to use fine_tuning_eval directory
        val_metrics_df.to_csv(os.path.join(fine_tuning_dir, f"patient_{patient_id}_validation_metrics.csv"), index=False)
        test_metrics_df.to_csv(os.path.join(fine_tuning_dir, f"patient_{patient_id}_test_metrics.csv"), index=False)
        
        # Save final test results for the best model
        print("\n-------------Final Testing with Best Model-------------")
        if best_model_path:
            # Load best model
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path} for final testing")
            
            # Final test
            final_tester = ModelTester(model, data_loaders["test_loader"], device)
            final_test_rmse, final_test_mae, final_stats_tables, detailed_df = final_tester.test()
            
            print(f"Final Test RMSE: {final_test_rmse:.6f}")
            print(f"Final Test MAE: {final_test_mae:.6f}")
            
            # Save detailed test results to fine_tuning_eval directory
            if detailed_df is not None and not detailed_df.empty:
                detailed_df.to_csv(os.path.join(fine_tuning_dir, f"patient_{patient_id}_detailed_test_results.csv"), index=False)
                
            # Save CG-EGA statistics to fine_tuning_eval directory
            overall_cg_ega_df = final_stats_tables["overall"]
            overall_cg_ega_df.to_csv(os.path.join(fine_tuning_dir, f"patient_{patient_id}_overall_cg_ega.csv"))
            
            # Save timepoint statistics to fine_tuning_eval directory
            for timepoint, df in final_stats_tables["timepoints"].items():
                df.to_csv(os.path.join(fine_tuning_dir, f"patient_{patient_id}_cg_ega_{timepoint}.csv"))   

if __name__ == "__main__":
    main()