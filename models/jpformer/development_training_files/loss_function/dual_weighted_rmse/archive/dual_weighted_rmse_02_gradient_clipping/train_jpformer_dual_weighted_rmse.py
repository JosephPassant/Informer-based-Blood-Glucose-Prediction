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
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../../../../"))

sys.path.append(os.path.join(PROJECT_ROOT, "shared_utilities"))
from utilities import *
from dual_weighted_loss_function import *
from dual_weighted_loss_training_validation_modules import *

sys.path.append(os.path.join(PROJECT_ROOT, "evaluation_files"))
from evaluation_for_no_feature_enhancement import *

sys.path.append(os.path.join(PROJECT_ROOT, "models/jpformer"))
from jpformer import JPFormer



# Set configuration file path
config_path = os.path.join(PROJECT_ROOT, "models/shared_config_files/base_training_config_without_feature_enhancement.json")

# Load configuration file
config_dict = load_config(config_path)
config = ConfigObject(config_dict)

# Setup device
if torch.cuda.is_available() and config.use_gpu:
    device = torch.device(f"cuda:{config.gpu}")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon (M1/M2)
    print("Using MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Model setup
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

model_name = model.__class__.__name__
print(f"Initialized model: {model_name}")


# Data setup
TRAIN_ENCODER_DIR = os.path.join(PROJECT_ROOT, "data/processed_data/replace_bg/baseline_no_feature_enhancement_211_undersample/training/encoder_slices")
TRAIN_DECODER_DIR = os.path.join(PROJECT_ROOT, "data/processed_data/replace_bg/baseline_no_feature_enhancement_211_undersample/training/decoder_slices")
TRAIN_TARGET_DIR = os.path.join(PROJECT_ROOT, "data/processed_data/replace_bg/baseline_no_feature_enhancement_211_undersample/training/target_slices")

VAL_ENCODER_DIR = os.path.join(PROJECT_ROOT, "data/processed_data/replace_bg/baseline_no_feature_enhancement_211_undersample/validation/encoder_slices")
VAL_DECODER_DIR = os.path.join(PROJECT_ROOT, "data/processed_data/replace_bg/baseline_no_feature_enhancement_211_undersample/validation/decoder_slices")
VAL_TARGET_DIR = os.path.join(PROJECT_ROOT, "data/processed_data/replace_bg/baseline_no_feature_enhancement_211_undersample/validation/target_slices")


TEST_ENCODER_DIR = os.path.join(PROJECT_ROOT, "data/processed_data/replace_bg/baseline_no_feature_enhancement_211_undersample/testing/encoder_slices")
TEST_DECODER_DIR = os.path.join(PROJECT_ROOT, "data/processed_data/replace_bg/baseline_no_feature_enhancement_211_undersample/testing/decoder_slices")
TEST_TARGET_DIR = os.path.join(PROJECT_ROOT, "data/processed_data/replace_bg/baseline_no_feature_enhancement_211_undersample/testing/target_slices")


# Training Data
train_dataset = BloodGlucoseDataset(TRAIN_ENCODER_DIR, TRAIN_DECODER_DIR, TRAIN_TARGET_DIR)
val_dataset = BloodGlucoseDataset(VAL_ENCODER_DIR, VAL_DECODER_DIR, VAL_TARGET_DIR)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=config.num_workers)
train_iter = ForeverDataIterator(train_loader)

# Test Data
test_dataset = BloodGlucoseDataset(TEST_ENCODER_DIR, TEST_DECODER_DIR, TEST_TARGET_DIR)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=config.num_workers)

# Wrap training DataLoader for continuous iteration
train_iter = ForeverDataIterator(train_loader)

# Optimizer & Scheduler
optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

total_steps = len(train_loader) * config.train_epochs
# use basic cosine annealing scheduler
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)


if __name__ == "__main__":

    model_type = "jpformer_dual_weighted_rmse_loss_func_gc_02"

    ap_weight = 1.0
    be_weight = 2.0
    ep_weight = 6
    hypo_multiplier = 1.5



    print("-------------Training-------------")
    
    save_dir = os.path.join(PROJECT_ROOT, "models", "jpformer", "development_training_files", "loss_function", "dual_weighted_rmse_02_gradient_clipping")
    os.makedirs(save_dir, exist_ok=True)



    best_model_path = train(train_iter, model, optimizer, lr_scheduler, config,ap_weight=ap_weight,be_weight=be_weight,ep_weight=ep_weight,hypo_multiplier=hypo_multiplier, train_loader=train_loader, val_loader=val_loader, save_dir=save_dir)
                            

    print("\n-------------Testing-------------")
    
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path} for final testing.")

    tester = ModelTester(model, test_loader, device)

    test_rmse, test_mae, stats_tables, detailed_df = tester.test()

    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_description = f"{model_type}_{test_rmse:.4f}_MAE_{test_mae:.4f}"


    if detailed_df is None or detailed_df.empty:
        print("ERROR: actual_predicted_dy_df is empty. Skipping save.")
    else:
        detailed_df.to_csv(os.path.join(save_dir, "detailed_results_table.csv"), index=False)
        print("Saved detailed_results_table.csv")
    
    # Save CG-EGA statistics
    overall_cg_ega_df = stats_tables["overall"]
    overall_cg_ega_df.to_csv(os.path.join(save_dir, "overall_cg_ega.csv"))
    print("Saved overall CG-EGA statistics")

    # Save timepoint statistics
    for timepoint, df in stats_tables["timepoints"].items():
        df.to_csv(os.path.join(save_dir, f"cg_ega_stats_{timepoint}.csv"))
        print(f"Saved {timepoint} CG-EGA statistics")
    
    # Save trained model
    model_filename = f"{model_type}_{test_rmse:.4f}_MAE_{test_mae:.4f}.pth"
    model_path = os.path.join(save_dir, model_filename)

    if model.state_dict():
        torch.save(model.state_dict(), model_path)
        print(f"Saved model at {model_path}")
    else:
        print("ERROR: model.state_dict() is empty. Model not saved.")

    print(f"Model and results saved to {save_dir}/")

