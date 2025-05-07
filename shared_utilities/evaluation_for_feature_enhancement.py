import pandas as pd
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../"))

sys.path.append(os.path.join(PROJECT_ROOT, "shared_utilities"))
from metrics import *
from utilities import *
from cg_ega_loss_metric import P_EGA_Loss, R_EGA_Loss, CG_EGA_Loss
# Set configuration file path
config_path = os.path.join(PROJECT_ROOT, 'evaluation_files', 'evaluation_config_with_feature_enhancement.json')

# Load configuration file
config_dict = load_config(config_path)
config = ConfigObject(config_dict)



def get_cg_ega_mappings():
    """
    Returns the mappings used to convert P-EGA and R-EGA classifications to CG-EGA classes (AP, BE, EP)
    for different glycemic regions.
    
    Returns:
        dict: Dictionary containing mappings for hypoglycemia, euglycemia, and hyperglycemia
    """
    # Hypoglycemia mapping
    hypo_mapping = {
        # AP: Accurate Predictions
        ("A", "A"): "AP",
        ("A", "B"): "AP",

        # BE: Benign Errors
        ("A", "uC"): "BE",
        ("A", "lC"): "BE",
        ("A", "lD"): "BE",
        ("A", "lE"): "BE",

        # EP: Erroneous Predictions
        ("A", "uD"): "EP",
        ("A", "uE"): "EP",
        ("D", "*"): "EP",
        ("E", "*"): "EP"
    }

    # Euglycemia mapping
    eu_mapping = {
        # AP: Accurate Predictions
        ("A", "A"): "AP",
        ("A", "B"): "AP",
        ("B", "A"): "AP",
        ("B", "B"): "AP",

        # BE: Benign Errors
        ("A", "uC"): "BE",
        ("A", "lC"): "BE",
        ("A", "uD"): "BE",
        ("A", "lD"): "BE",
        ("B", "uC"): "BE",
        ("B", "lC"): "BE",
        ("B", "uD"): "BE",
        ("B", "lD"): "BE",

        # EP: Erroneous Predictions
        ("A", "uE"): "EP",
        ("A", "lE"): "EP",
        ("B", "uE"): "EP",
        ("B", "lE"): "EP",
        ("C", "*"): "EP"
    }

    # Hyperglycemia mapping
    hyper_mapping = {
        # AP: Accurate Predictions
        ("A", "A"): "AP",
        ("A", "B"): "AP",
        ("B", "A"): "AP",
        ("B", "B"): "AP",

        # BE: Benign Errors
        ("A", "uC"): "BE",
        ("A", "lC"): "BE",
        ("A", "uD"): "BE",
        ("B", "uC"): "BE",
        ("B", "lC"): "BE",
        ("B", "uD"): "BE",

        # EP: Erroneous Predictions
        ("A", "lD"): "EP",
        ("A", "lE"): "EP",
        ("A", "uE"): "EP",
        ("B", "lD"): "EP",
        ("B", "lE"): "EP",
        ("B", "uE"): "EP",
        ("C", "*"): "EP",
        ("D", "*"): "EP",
        ("E", "*"): "EP"
    }
    
    return {
        "hypo": hypo_mapping,
        "eu": eu_mapping,
        "hyper": hyper_mapping
    }


def map_cg_ega(p_label, r_label, glucose_region):
    """
    Maps the P-EGA and R-EGA classification to CG-EGA class (AP, BE, EP) based on glucose region.

    Args:
        p_label: String, P-EGA classification (A, B, C, D, E)
        r_label: String, R-EGA classification (A, B, uC, lC, uD, lD, uE, lE)
        glucose_region: String, one of ("hypo", "eu", "hyper")
    
    Returns:
        String, CG-EGA classification (AP, BE, EP)
    """
    mappings = get_cg_ega_mappings()
    
    if glucose_region == "hypo":
        mapping = mappings["hypo"]
    elif glucose_region == "eu":
        mapping = mappings["eu"]
    else:  # hyper
        mapping = mappings["hyper"]

    # Look up classification, accounting for wildcard conditions ("*")
    if (p_label, r_label) in mapping:
        return mapping[(p_label, r_label)]
    elif (p_label, "*") in mapping:
        return mapping[(p_label, "*")]
    elif ("*", r_label) in mapping:
        return mapping[("*", r_label)]
    else:
        return "EP"  # Default to EP if no match


class ModelTester:
    """
    Class for evaluating the trained model on test data.
    
    Methods:
        test(): Runs evaluation on test data and computes RMSE, MAE, and CG-EGA metrics.
    """

    def __init__(self, model, test_loader, device):
        """
        Initialize the ModelTester.

        Args:
            model (nn.Module): The trained model.
            test_loader (DataLoader): DataLoader for the test dataset.
            device (torch.device): The device (CPU/GPU) to use.
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.rmse = rmse
        self.mae = mae

    def test(self):
        """
        Run the test evaluation and compute RMSE, MAE, and CG-EGA metrics.
        Creates a detailed DataFrame with one prediction per row for fine-grained analysis.
        
        Returns:
            tuple: (average_rmse, average_mae, stats_tables, detailed_df)
            - average_rmse: Average RMSE across all batches
            - average_mae: Average MAE across all batches
            - stats_tables: Dictionary with overall and timepoint statistics tables
            - detailed_df: DataFrame with one row per prediction point and all classifications
        """
        self.model.eval()
        total_rmse, total_mae = 0, 0
        num_batches = len(self.test_loader)
        all_results = []
        sequence_ids = []  # Track which sequence each point belongs to
        
        with torch.no_grad():
            for batch_idx, (batch_x, batch_dec, batch_y) in enumerate(self.test_loader):
                batch_x, batch_dec, batch_y = (
                    batch_x.float().to(self.device),
                    batch_dec.float().to(self.device),
                    batch_y.float().to(self.device),
                )

                # Get model predictions
                outputs = self.model(batch_x, batch_dec)

                # Ensure target shape matches model output
                if batch_y.ndim == 2:
                    batch_y = batch_y.unsqueeze(-1)

                # Compute RMSE and MAE
                batch_rmse = self.rmse(outputs, batch_y)
                batch_mae = self.mae(outputs, batch_y)

                total_rmse += batch_rmse
                total_mae += batch_mae

                # Store results with sequence ID
                for i in range(len(batch_y)):
                    seq_id = batch_idx * batch_x.shape[0] + i
                    true_values = batch_y[i].cpu().numpy().flatten()
                    pred_values = outputs[i].cpu().numpy().flatten()
                    
                    # Calculate derivatives using np.diff with zero insertion (consistent with original implementation)
                    true_derivatives = np.zeros_like(true_values)
                    pred_derivatives = np.zeros_like(pred_values)
                    if len(true_values) > 1:
                        true_derivatives[1:] = np.diff(true_values) / 5
                        pred_derivatives[1:] = np.diff(pred_values) / 5
                    
                    # Store each point in the sequence with its metadata
                    for t in range(len(true_values)):
                        all_results.append({
                            "sequence_id": seq_id,
                            "timepoint": t,
                            "true": true_values[t],
                            "predicted": pred_values[t],
                            "dy_true": true_derivatives[t],
                            "dy_pred": pred_derivatives[t]
                        })

        # Create initial DataFrame
        detailed_df = pd.DataFrame(all_results)
        
        # Denormalize the values (consistent with original implementation)
        detailed_df["true_glucose"] = detailed_df["true"] * config.std + config.mean
        detailed_df["predicted_glucose"] = detailed_df["predicted"] * config.std + config.mean
        detailed_df["dy_true_glucose"] = detailed_df["dy_true"] * config.std
        detailed_df["dy_pred_glucose"] = detailed_df["dy_pred"] * config.std
        
        
        # Prepare lists to store classifications
        p_ega_classifications = []
        r_ega_classifications = []
        cg_ega_classifications = []
        
        # Process each sequence to calculate P-EGA, R-EGA, and CG-EGA
        for seq_id in detailed_df["sequence_id"].unique():
            seq_data = detailed_df[detailed_df["sequence_id"] == seq_id]
            
            # Get sorted data for this sequence
            seq_data = seq_data.sort_values("timepoint")
            true_glucose = seq_data["true_glucose"].values
            predicted_glucose = seq_data["predicted_glucose"].values
            dy_true_glucose = seq_data["dy_true_glucose"].values
            dy_pred_glucose = seq_data["dy_pred_glucose"].values
            
            # Calculate CG-EGA for the sequence
            cg_ega = CG_EGA_Loss(
                true_glucose, 
                dy_true_glucose, 
                predicted_glucose, 
                dy_pred_glucose,
                freq=5
            )
            
            # Calculate P-EGA (point error grid)
            p_ega_results = P_EGA_Loss(
                true_glucose, 
                dy_true_glucose, 
                predicted_glucose
            ).full()
            
            # Calculate R-EGA (rate error grid)
            r_ega_results = R_EGA_Loss(
                dy_true_glucose, 
                dy_pred_glucose
            ).full()
            
            # Convert one-hot encoded results to class labels
            p_ega_labels = np.array(["A", "B", "C", "D", "E"])[np.argmax(p_ega_results, axis=1)]
            r_ega_labels = np.array(["A", "B", "uC", "lC", "uD", "lD", "uE", "lE"])[np.argmax(r_ega_results, axis=1)]
            
            # Determine glycemic region for each point
            regions = np.where(
                true_glucose <= 70, "hypo", 
                np.where(true_glucose <= 180, "eu", "hyper")
            )
            
            # Get CG-EGA classifications (AP, BE, EP)
            sample_classifications = np.array([
                cg_ega.map_cg_ega(p_idx, r_idx, region)
                for p_idx, r_idx, region in zip(
                    np.argmax(p_ega_results, axis=1),
                    np.argmax(r_ega_results, axis=1),
                    regions
                )
            ])
            
            # Append to our lists in the original sequence order
            for idx in seq_data.index:
                pos = seq_data.loc[idx, "timepoint"]
                p_ega_classifications.append(p_ega_labels[pos])
                r_ega_classifications.append(r_ega_labels[pos])
                cg_ega_classifications.append(sample_classifications[pos])
        
        # Add classifications to DataFrame
        detailed_df["P_EGA_Class"] = p_ega_classifications
        detailed_df["R_EGA_Class"] = r_ega_classifications
        detailed_df["CG_EGA_Class"] = cg_ega_classifications
        
        # Add glycemic region column
        detailed_df["glycemic_region"] = np.where(
            detailed_df["true_glucose"] <= 70, "hypo", 
            np.where(detailed_df["true_glucose"] <= 180, "eu", "hyper")
        )
        
        # Generate statistics for specific timepoints (30, 60, 90, 120 minutes)
        timepoints = {
            "30min": 5,   # 30 minutes = index 5
            "60min": 11,  # 60 minutes = index 11
            "90min": 17,  # 90 minutes = index 17
            "120min": 23  # 120 minutes = index 23
        }
        
        # Initialize statistics tables
        timepoint_stats = {}
        region_stats = {
            "hypo": {"AP": 0, "BE": 0, "EP": 0, "count": 0},
            "eu": {"AP": 0, "BE": 0, "EP": 0, "count": 0},
            "hyper": {"AP": 0, "BE": 0, "EP": 0, "count": 0},
            "overall": {"AP": 0, "BE": 0, "EP": 0, "count": 0}
        }
        
        # Initialize timepoint statistics
        for tp_name in timepoints:
            timepoint_stats[tp_name] = {
                "hypo": {"AP": 0, "BE": 0, "EP": 0, "count": 0},
                "eu": {"AP": 0, "BE": 0, "EP": 0, "count": 0},
                "hyper": {"AP": 0, "BE": 0, "EP": 0, "count": 0},
                "overall": {"AP": 0, "BE": 0, "EP": 0, "count": 0}
            }
        
        # Group data for statistics
        for region in ["hypo", "eu", "hyper"]:
            region_data = detailed_df[detailed_df["glycemic_region"] == region]
            region_stats[region]["AP"] = sum(region_data["CG_EGA_Class"] == "AP")
            region_stats[region]["BE"] = sum(region_data["CG_EGA_Class"] == "BE")
            region_stats[region]["EP"] = sum(region_data["CG_EGA_Class"] == "EP")
            region_stats[region]["count"] = len(region_data)
            
            region_stats["overall"]["AP"] += region_stats[region]["AP"]
            region_stats["overall"]["BE"] += region_stats[region]["BE"]
            region_stats["overall"]["EP"] += region_stats[region]["EP"]
            region_stats["overall"]["count"] += region_stats[region]["count"]
            
            # Calculate statistics for specific timepoints
            for tp_name, tp_idx in timepoints.items():
                tp_data = region_data[region_data["timepoint"] == tp_idx]
                timepoint_stats[tp_name][region]["AP"] = sum(tp_data["CG_EGA_Class"] == "AP") 
                timepoint_stats[tp_name][region]["BE"] = sum(tp_data["CG_EGA_Class"] == "BE")
                timepoint_stats[tp_name][region]["EP"] = sum(tp_data["CG_EGA_Class"] == "EP")
                timepoint_stats[tp_name][region]["count"] = len(tp_data)
                
                timepoint_stats[tp_name]["overall"]["AP"] += timepoint_stats[tp_name][region]["AP"]
                timepoint_stats[tp_name]["overall"]["BE"] += timepoint_stats[tp_name][region]["BE"]
                timepoint_stats[tp_name]["overall"]["EP"] += timepoint_stats[tp_name][region]["EP"]
                timepoint_stats[tp_name]["overall"]["count"] += timepoint_stats[tp_name][region]["count"]
        
        # Create percentage tables
        # Create tables with counts
        def create_count_table(stats_dict):
            table = {}
            for region, counts in stats_dict.items():
                table[region] = {
                    "AP": counts["AP"],
                    "BE": counts["BE"],
                    "EP": counts["EP"],
                    "Count": counts["count"]
                }
            return pd.DataFrame(table).T
        
        # Create the region and timepoint tables
        region_table = create_count_table(region_stats)
        timepoint_tables = {}
        for tp_name in timepoints:
            timepoint_tables[tp_name] = create_count_table(timepoint_stats[tp_name])
        
        # Print summary
        print(f"\nTest Average RMSE: {total_rmse/num_batches:.4f}")
        print(f"Test Average MAE: {total_mae/num_batches:.4f}")
        
        print("\nOverall CG-EGA Statistics by Glycemic Region:")
        print(region_table)
        
        for tp_name in timepoints:
            print(f"\nCG-EGA Statistics at {tp_name}:")
            print(timepoint_tables[tp_name])
        
        # Return results in order expected by the main script
        stats_tables = {
            "overall": region_table,
            "timepoints": timepoint_tables
        }
        
        average_rmse = total_rmse / num_batches
        average_mae = total_mae / num_batches
        
        return average_rmse, average_mae, stats_tables, detailed_df