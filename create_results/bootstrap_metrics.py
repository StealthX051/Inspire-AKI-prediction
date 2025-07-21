# bootstrap_metrics.py
#
# Contains the core function for calculating bootstrap 95% confidence intervals
# for key performance and calibration metrics using parallel processing.

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             fbeta_score, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor
import os

# Helper function for a single bootstrap iteration.
# Must be a top-level function to be "picklable" by multiprocessing.
def _bootstrap_single_iteration(args):
    """Performs a single bootstrap resample and metric calculation."""
    y_true, p_calib, tau_star, seed = args
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    
    # Generate bootstrap indices
    idx = rng.integers(0, n_samples, n_samples)
    y_boot, p_boot = y_true[idx], p_calib[idx]

    # Initialize results dictionary for this iteration
    results = {}
    
    if len(np.unique(y_boot)) < 2:
        return None # Return None if a class is missing

    # Calculate metrics
    results["auroc"] = roc_auc_score(y_boot, p_boot)
    results["auprc"] = average_precision_score(y_boot, p_boot)
    results["f2"] = fbeta_score(y_boot, p_boot >= tau_star, beta=2)
    results["brier"] = brier_score_loss(y_boot, p_boot)

    try:
        lr_calib = LogisticRegression(penalty="none", solver="lbfgs")
        lr_calib.fit(p_boot.reshape(-1, 1), y_boot)
        results["calib_int"] = lr_calib.intercept_[0]
        results["calib_slp"] = lr_calib.coef_[0, 0]
    except Exception:
        results["calib_int"] = np.nan
        results["calib_slp"] = np.nan
        
    return results

def bootstrap_metrics(y_true, p_calib, tau_star, B=1000, seed=0):
    """
    Calculates bootstrap 95% confidence intervals in parallel.
    """
    # Prepare a dictionary to store the results of each bootstrap iteration
    bootstrap_samples = {
        "auroc": [], "auprc": [], "f2": [], "brier": [],
        "calib_int": [], "calib_slp": []
    }
    
    # Create a master RNG to generate seeds for each worker process
    master_rng = np.random.default_rng(seed)
    worker_seeds = master_rng.integers(0, 2**32 - 1, B)
    
    # Prepare arguments for each worker
    args_list = [(y_true, p_calib, tau_star, s) for s in worker_seeds]

    # Use ProcessPoolExecutor to run iterations in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # map() applies the function to each item in args_list and returns an iterator of results
        results_iterator = executor.map(_bootstrap_single_iteration, args_list)

    # Collect results from the parallel execution
    for result in results_iterator:
        if result is not None:
            for key, value in result.items():
                bootstrap_samples[key].append(value)

    def get_ci(metric_values):
        """Helper function to calculate mean and 95% CI from a list of values."""
        valid_values = np.asarray(metric_values)
        valid_values = valid_values[~np.isnan(valid_values)]
        if len(valid_values) == 0:
            return (np.nan, np.nan, np.nan)
        point_estimate = valid_values.mean()
        lower_bound = np.percentile(valid_values, 2.5)
        upper_bound = np.percentile(valid_values, 97.5)
        return (point_estimate, lower_bound, upper_bound)

    # Calculate the confidence interval for each metric
    final_results = {key: get_ci(values) for key, values in bootstrap_samples.items()}
    return final_results
