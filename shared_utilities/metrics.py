import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F

from datetime import datetime, timedelta



"""
    This file contains AP/BE/EP filters for every glycemia regions. It is used by the CG-EGA object.
"""

filter_AP_hypo = [
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

filter_BE_hypo = [
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
]

filter_EP_hypo = [
    [0, 1, 1],
    [0, 1, 1],
    [0, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, 1, 1],
]

filter_AP_eu = [
    [1, 1, 0],
    [1, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

filter_BE_eu = [
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
]

filter_EP_eu = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [1, 1, 1],
    [1, 1, 1],
]

filter_AP_hyper = [
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

filter_BE_hyper = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

filter_EP_hyper = [
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
]


def rmse(predictions, targets):
    # Ensure predictions and targets have the same shape
    if predictions.shape != targets.shape:
        print(f"[WARNING] Adjusting Shapes: predictions={predictions.shape}, targets={targets.shape}")
        targets = targets.squeeze(-1) if targets.shape[-1] == 1 else targets
        predictions = predictions.unsqueeze(-1) if predictions.ndim == 2 else predictions

    assert predictions.shape == targets.shape, f"[RMSE Function] Shape Mismatch: {predictions.shape} vs {targets.shape}"
    
    return torch.sqrt(F.mse_loss(predictions, targets))


def mae(predictions, targets):
    """
    Compute MAE (Mean Absolute Error).

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth values.

    Returns:
        float: MAE value.
    """
    return torch.mean(torch.abs(predictions - targets))


class CG_EGA():
    """
    The Continuous Glucose-Error Grid Analysis (CG-EGA) gives a measure of the clinical acceptability of the glucose predictions. It analyzes both the
    prediction accuracy (through the P-EGA) and the predicted variation accuracy (R-EGA).

    The implementation has been made following "Evaluating the accuracy of continuous glucose-monitoring sensors:
    continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et al., 2004.
    """
    def __init__(self, y_true, dy_true, y_pred, dy_pred, freq):
        """
        Instantiate the CG-EGA object with explicit rate-of-change inputs.
        
        :param y_true: numpy array of actual glucose values
        :param dy_true: numpy array of actual glucose rate of change
        :param y_pred: numpy array of predicted glucose values
        :param dy_pred: numpy array of predicted glucose rate of change
        :param freq: prediction frequency in minutes (e.g., 5)
        """
        self.y_true = y_true
        self.dy_true = dy_true
        self.y_pred = y_pred
        self.dy_pred = dy_pred
        self.freq = freq

        # Ensure input arrays are properly structured
        assert len(y_true) == len(y_pred), f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        assert len(dy_true) == len(dy_pred), f"Length mismatch: dy_true ({len(dy_true)}) != dy_pred ({len(dy_pred)})"


        # Instantiate P-EGA and R-EGA with provided values
        self.p_ega = P_EGA(self.y_true, self.dy_true, self.y_pred).full()
        self.r_ega = R_EGA(dy_true, dy_pred).full()

        assert self.p_ega.shape[0] == len(y_true), f"P-EGA output shape mismatch: {self.p_ega.shape}"
        assert self.r_ega.shape[0] == len(y_true), f"R-EGA output shape mismatch: {self.r_ega.shape}"

    def full(self):
        """
        Compute CG-EGA by combining P-EGA and R-EGA.

        :return: CG-EGA matrices for hypoglycemia, euglycemia, and hyperglycemia.
        """

        # **Define Glycemic Regions**
        hypoglycemia = (self.y_true <= 70).reshape(-1, 1)
        euglycemia = ((self.y_true > 70) & (self.y_true <= 180)).reshape(-1, 1)
        hyperglycemia = (self.y_true > 180).reshape(-1, 1)

        # **Apply Region Filters (Ensures Correct Dimensions for np.dot)**
        P_hypo = np.concatenate([self.p_ega[:, 0:1], self.p_ega[:, 3:4], self.p_ega[:, 4:5]], axis=1) * hypoglycemia
        P_eu = np.concatenate([self.p_ega[:, 0:1], self.p_ega[:, 1:2], self.p_ega[:, 2:3]], axis=1) * euglycemia
        P_hyper = self.p_ega * hyperglycemia

        R_hypo = self.r_ega * hypoglycemia
        R_eu = self.r_ega * euglycemia
        R_hyper = self.r_ega * hyperglycemia

        # **Ensure Integer Conversion for Correct Matrix Multiplication**
        P_hypo, P_eu, P_hyper = P_hypo.astype(int), P_eu.astype(int), P_hyper.astype(int)
        R_hypo, R_eu, R_hyper = R_hypo.astype(int), R_eu.astype(int), R_hyper.astype(int)

        # **Compute CG-EGA Matrices via Dot Product**
        CG_EGA_hypo = np.dot(R_hypo.T, P_hypo)  # (8×3)
        CG_EGA_eu = np.dot(R_eu.T, P_eu)  # (8×3)
        CG_EGA_hyper = np.dot(R_hyper.T, P_hyper)  # (8×5)

        # **Corrected Assertions to Match Expected CG-EGA Grid Dimensions**
        assert CG_EGA_hypo.shape == (8, 3), f"CG-EGA Hypo shape incorrect: {CG_EGA_hypo.shape}"
        assert CG_EGA_eu.shape == (8, 3), f"CG-EGA Eu shape incorrect: {CG_EGA_eu.shape}"
        assert CG_EGA_hyper.shape == (8, 5), f"CG-EGA Hyper shape incorrect: {CG_EGA_hyper.shape}"

        return CG_EGA_hypo, CG_EGA_eu, CG_EGA_hyper

    def simplified(self, count=False):
        """
        Simplifies the full CG-EGA into Accurate Prediction (AP), Benign Prediction (BE), and Erroneous Prediction (EP)
        rates for every glycemia regions.

        :param count: if False, the results, for every region, will be expressed as a ratio

        :return: AP rate in hypoglycemia, BE rate in hypoglycemia, EP rate in hypoglycemia,
                    AP rate in euglycemia, BE rate in euglycemia, EP rate in euglycemia,
                     AP rate in hyperglycemia, BE rate in hyperglycemia, EP rate in hyperglycemia
        """

        CG_EGA_hypo, CG_EGA_eu, CG_EGA_hyper = self.full()

        # Define error classifications as per Kovatchev et al.
        AP_hypo = np.sum(CG_EGA_hypo * filter_AP_hypo)
        BE_hypo = np.sum(CG_EGA_hypo * filter_BE_hypo)
        EP_hypo = np.sum(CG_EGA_hypo * filter_EP_hypo)

        AP_eu = np.sum(CG_EGA_eu * filter_AP_eu)
        BE_eu = np.sum(CG_EGA_eu * filter_BE_eu)
        EP_eu = np.sum(CG_EGA_eu * filter_EP_eu)

        AP_hyper = np.sum(CG_EGA_hyper * filter_AP_hyper)
        BE_hyper = np.sum(CG_EGA_hyper * filter_BE_hyper)
        EP_hyper = np.sum(CG_EGA_hyper * filter_EP_hyper)

        # Compute proportions (handling division by zero)
        total_hypo = AP_hypo + BE_hypo + EP_hypo
        total_eu = AP_eu + BE_eu + EP_eu
        total_hyper = AP_hyper + BE_hyper + EP_hyper
      
        # udpated to return counts for CHI-SQUARE TEST
        AP_hypo, BE_hypo, EP_hypo = (AP_hypo , BE_hypo, EP_hypo) if total_hypo > 0 else (np.nan, np.nan, np.nan)
        AP_eu, BE_eu, EP_eu = (AP_eu, BE_eu, EP_eu) if total_eu > 0 else (np.nan, np.nan, np.nan)
        AP_hyper, BE_hyper, EP_hyper = (AP_hyper, BE_hyper, EP_hyper) if total_hyper > 0 else (np.nan, np.nan, np.nan)

        return {
            "AP_hypo": AP_hypo, "BE_hypo": BE_hypo, "EP_hypo": EP_hypo,
            "AP_eu": AP_eu, "BE_eu": BE_eu, "EP_eu": EP_eu,
            "AP_hyper": AP_hyper, "BE_hyper": BE_hyper, "EP_hyper": EP_hyper
        }, {"count_hypo": total_hypo, "count_eu": total_eu, "count_hyper": total_hyper}

    def reduced(self):
        """
            Reduces the simplified CG-EGA by not dividing the results into the glycemia regions
            :return: overall AP rate, overall BE rate, overall EP rate
        """

        AP_hypo, BE_hypo, EP_hypo, AP_eu, BE_eu, EP_eu, AP_hyper, BE_hyper, EP_hyper = self.simplified(count=True)
        sum = (AP_hypo + BE_hypo + EP_hypo + AP_eu + BE_eu + EP_eu + AP_hyper + BE_hyper + EP_hyper)
        return (AP_hypo + AP_eu + AP_hyper) / sum, (BE_hypo + BE_eu + BE_hyper) / sum, (
                EP_hypo + EP_eu + EP_hyper) / sum


class P_EGA():
    """
    The Point-Error Grid Analysis (P-EGA) estimates the clinical acceptability of glucose predictions
    based on their point accuracy. It follows the Continuous Glucose-Error Grid Analysis (CG-EGA) method 
    as described by Kovatchev et al., 2004.

    This implementation enforces strict one-hot encoding, ensuring that each prediction is classified 
    into exactly one category (A, B, C, D, or E).
    """

    def __init__(self, y_true, dy_true, y_pred):
        """
        :param y_true: numpy array of actual glucose values
        :param dy_true: numpy array of actual glucose rate of change
        :param y_pred: numpy array of predicted glucose values
        """
        self.y_true = y_true
        self.dy_true = dy_true
        self.y_pred = y_pred

        # Ensure input consistency
        assert len(y_true) == len(y_pred) == len(dy_true), "Mismatch in input array lengths."

    def full(self):
        """
        Compute P-EGA classifications based on Kovatchev et al. (2004), ensuring one-hot encoding.

        :return: numpy array (num_samples, 5), where each row contains a one-hot encoded classification.
        """
        num_samples = len(self.y_true)

        # Compute dynamic error boundary modification for A, B, and D classifications
        mod = np.zeros_like(self.dy_true)

        # Expand upper limits for falling rates (-1 to -2 mg/dL/min)
        mod[(self.dy_true >= -2) & (self.dy_true <= -1)] = 10
        # Expand lower limits for rising rates (1 to 2 mg/dL/min)
        mod[(self.dy_true >= 1) & (self.dy_true <= 2)] = 10
        # Expand upper limits for rapid falls (< -2 mg/dL/min)
        mod[self.dy_true < -2] = 20
        # Expand lower limits for rapid rises (> 2 mg/dL/min)
        mod[self.dy_true > 2] = 20

        # Initialize classification matrix (one-hot encoded)
        classifications = np.zeros((num_samples, 5), dtype=int)  # Columns represent A, B, C, D, E

        # **Classification Conditions**

        ## Accurate Prediction (A)
        A_mask = (
            (self.y_pred <= 70 + mod) & (self.y_true <= 70)
        ) | (
            (self.y_pred <= self.y_true * 6 / 5 + mod) & 
            (self.y_pred >= self.y_true * 4 / 5 - mod)
        )

        ## Erroneous Prediction (E) - No mod applied except for implicit shifts
        E_mask = (
            (self.y_true > 180 + mod) & (self.y_pred < 70 - mod)  # Shift EP upper boundary upwards
        ) | (
            (self.y_pred > 180 + mod) & (self.y_true <= 70 - mod)  # Shift EP lower boundary downwards
        )

        ## Dangerous Prediction (D)
        D_mask = (
            (self.y_pred > 70 + mod) & 
            (self.y_pred > self.y_true * 6 / 5 + mod) & 
            (self.y_true <= 70) & 
            (self.y_pred <= 180 + mod)
        ) | (
            (self.y_true > 240) & 
            (self.y_pred < 180 - mod) & 
            (self.y_pred >= 70 - mod)
        )

        ## Clinically Benign Error (C) - Implicitly shift boundary based on B expansions
        C_mask = (
            (self.y_true > 70 + mod) &  # Shift CP upper boundary upwards
            (self.y_pred > self.y_true * 22 / 17 + (180 - 70 * 22 / 17) + mod)  
        ) | (
            (self.y_true <= 180 - mod) &  # Shift CP lower boundary downwards
            (self.y_pred < self.y_true * 7 / 5 - 182 - mod)  
        )

        # Assign categories (ensuring one-hot encoding)
        classifications[A_mask, 0] = 1  # A
        classifications[E_mask, 4] = 1  # E
        classifications[D_mask, 3] = 1  # D
        classifications[C_mask, 2] = 1  # C

        ## Clinically Acceptable but Outside Ideal (B) - Assigned last to prevent overlaps
        B_mask = ~A_mask & ~C_mask & ~D_mask & ~E_mask
        classifications[B_mask, 1] = 1  # B

        # Ensure one-hot encoding
        assert np.all(np.sum(classifications, axis=1) == 1), "One-hot encoding violated in P-EGA."

        return classifications

    def mean(self):
        """
        Compute the mean occurrence rate of each classification category.

        :return: NumPy array with the mean frequency of each classification.
        """
        return np.mean(self.full(), axis=0)

    def a_plus_b(self):
        """
        Compute the proportion of predictions classified as clinically acceptable (A or B).

        :return: Fraction of total predictions classified as A or B.
        """
        full = self.full()
        a_plus_b = full[:, 0] + full[:, 1]  # Sum of A and B classifications
        return np.sum(a_plus_b) / len(a_plus_b)


import numpy as np

class R_EGA():
    """
    The Rate-Error Grid Analysis (R-EGA) estimates the clinical acceptability of glucose predictions 
    based on their rate-of-change accuracy.
    """

    def __init__(self, dy_true, dy_pred):
        """
        :param dy_true: numpy array of actual glucose rate of change
        :param dy_pred: numpy array of predicted glucose rate of change
        """
        self.dy_true = dy_true
        self.dy_pred = dy_pred

        # Ensure input consistency
        assert len(dy_true) == len(dy_pred), "Mismatch in dy_true and dy_pred lengths."

    def full(self):
        """
        Compute R-EGA classifications ensuring one-hot encoding.

        :return: numpy array with 8 columns (A, B, uC, lC, uD, lD, uE, lE).
        """
        num_samples = len(self.dy_true)

        # Initialize classification matrix (one-hot encoded)
        classifications = np.zeros((num_samples, 8), dtype=int)  # Columns: A, B, uC, lC, uD, lD, uE, lE

        adaptive_threshold = np.where(np.abs(self.dy_true) >= 4, 2, 1)

        # **Accurate Prediction (A)**
        A_mask = np.abs(self.dy_pred - self.dy_true) <= adaptive_threshold

        # **Extreme Errors (E) - Assigned first to avoid overlap**
        uE_mask = (self.dy_pred > 1) & (self.dy_true < -1)
        lE_mask = (self.dy_pred < -1) & (self.dy_true > 1)

        # **Clinically Dangerous Predictions (D) - Only if not classified as E**
        uD_mask = ~uE_mask & (self.dy_pred >= -1) & (self.dy_pred <= 1) & (self.dy_true > self.dy_pred + 2)
        lD_mask = ~lE_mask & (self.dy_pred >= -1) & (self.dy_pred <= 1) & (self.dy_true < self.dy_pred - 2)

        # **Clinically Acceptable, Benign Errors (C)**
        uC_mask = (self.dy_true >= -1) & (self.dy_true <= 1) & (self.dy_pred > self.dy_true + 2)
        lC_mask = (self.dy_true >= -1) & (self.dy_true <= 1) & (self.dy_pred < self.dy_true - 2)

        # **Benign Prediction (B) - Assigned last to ensure exclusivity**
        B_mask = ~(A_mask | uC_mask | lC_mask | uD_mask | lD_mask | uE_mask | lE_mask)

        # Assign categories (ensuring one-hot encoding)
        classifications[:, 0] = A_mask.astype(int)  # A
        classifications[:, 1] = B_mask.astype(int)  # B
        classifications[:, 2] = uC_mask.astype(int)  # uC
        classifications[:, 3] = lC_mask.astype(int)  # lC
        classifications[:, 4] = uD_mask.astype(int)  # uD
        classifications[:, 5] = lD_mask.astype(int)  # lD
        classifications[:, 6] = uE_mask.astype(int)  # uE
        classifications[:, 7] = lE_mask.astype(int)  # lE

        # **One-hot Encoding Assertion**
        assert np.all(np.sum(classifications, axis=1) == 1), "One-hot encoding violated in R-EGA!"

        return classifications

    def mean(self):
        """
        Compute the mean occurrence rate of each classification category.

        :return: NumPy array with the mean frequency of each classification.
        """
        return np.mean(self.full(), axis=0)

    def a_plus_b(self):
        """
        Compute the proportion of predictions classified as clinically acceptable (A or B).

        :return: Fraction of total predictions classified as A or B.
        """
        full = self.full()
        a_plus_b = full[:, 0] + full[:, 1]  # Sum of A and B classifications
        return np.sum(a_plus_b) / len(a_plus_b)




def weighted_rmse(pred, target, hypoglycemia_threshold=-1.18, hyperglycemia_threshold = 0.39,  hypo_weight_factor=6.0, hyper_weight_factor=2.0):
    """
    Compute a weighted RMSE where errors in the hypoglycemic range (<70 mg/dL) are weighted more.
    
    Args:
        pred (Tensor): Model predictions.
        target (Tensor): Ground truth values.
        hypoglycemia_threshold (float): Glucose threshold below which loss is weighted more.
        weight_factor (float): Multiplier for hypoglycemia error weighting.

    Returns:
        float: Weighted RMSE.
    """

    error = pred - target
    weight = torch.ones_like(target)  # Default weight is 1


    # Apply higher weight where target is in hypoglycemia range
    weight[target < hypoglycemia_threshold] = hypo_weight_factor

    # Apply higher weight where target is in hyperglycemia range
    weight[target > hyperglycemia_threshold] = hyper_weight_factor

    # Compute weighted RMSE
    weighted_mse = (weight * (error ** 2)).mean()
    return torch.sqrt(weighted_mse)