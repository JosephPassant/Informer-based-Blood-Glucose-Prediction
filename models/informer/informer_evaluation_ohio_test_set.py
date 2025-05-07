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
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))
print(f"Project root directory: {PROJECT_ROOT}")

sys.path.append(os.path.join(PROJECT_ROOT, "shared_utilities"))
from utilities import *
from dual_weighted_loss_function import *
from dual_weighted_loss_training_validation_modules import *
from evaluation_for_no_feature_enhancement import *

sys.path.append(os.path.join(PROJECT_ROOT, "models/informer"))
from Informer import *

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
    model = Informer(
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
            "encoder": os.path.join(project_root, f"data/processed_data/ohio/training/no_undersampling/ohio_training_{ptid}/EncoderSlices"),
            "decoder": os.path.join(project_root, f"data/processed_data/ohio/training/no_undersampling/ohio_training_{ptid}/DecoderSlices"),
            "target": os.path.join(project_root, f"data/processed_data/ohio/training/no_undersampling/ohio_training_{ptid}/TargetSlices")
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


def main():
        # Load configuration
    config_path = os.path.join(PROJECT_ROOT, "models/shared_config_files/fine_tuning_config.json")
    config_dict = load_config(config_path)
    config = ConfigObject(config_dict)
    
    # Setup device
    device = setup_device(config)
    
    # Patient IDs
    patient_ids = [540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596]
   
    # Pretrained weights path
    pretrained_weights_path = os.path.join(PROJECT_ROOT, "models/informer/final_model_training_files/jpformer_final_model_0.5810_MAE_0.4020.pth")
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
        patient_dir = os.path.join(PROJECT_ROOT, "models", "informer", "ohiot1dm_ptid_evaluation", f"patient_{patient_id}")
        create_dir(patient_dir)
        
        # Create base model and fine-tuning evaluation directories
        base_model_dir = os.path.join(patient_dir, "base_model_eval")
        fine_tuning_dir = os.path.join(patient_dir, "fine_tuning_eval")
        create_dir(base_model_dir)
        create_dir(fine_tuning_dir)

                # Initial testing and save base model results
        print("-------------Initial Testing of Pretrained Model for Patient-------------")
        initial_tester = ModelTester(model, data_loaders["test_loader"], device)
        initial_test_rmse, initial_test_mae, initial_stats_tables, initial_detailed_df = initial_tester.test()
        print(f"Pretrained model for Patient {patient_id} | Test RMSE: {initial_test_rmse:.6f} | Test MAE: {initial_test_mae:.6f}")
        
        # Save initial (base model) evaluation results
        if initial_detailed_df is not None and not initial_detailed_df.empty:
            initial_detailed_df.to_csv(os.path.join(base_model_dir, f"patient_{patient_id}_base_model_detailed_test.csv"), index=False)
        
        # Save initial CG-EGA statistics
        initial_overall_cg_ega_df = initial_stats_tables["overall"]
        initial_overall_cg_ega_df.to_csv(os.path.join(base_model_dir, f"patient_{patient_id}_base_model_overall_cg_ega.csv"))
        
        # Save initial timepoint statistics
        for timepoint, df in initial_stats_tables["timepoints"].items():
            df.to_csv(os.path.join(base_model_dir, f"patient_{patient_id}_base_model_cg_ega_{timepoint}.csv"))

if __name__ == "__main__":
    main()
