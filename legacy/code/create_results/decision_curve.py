# decision_curve.py
#
# Contains functions for performing Decision-Curve Analysis (DCA), including
# calculating net benefit and generating the DCA plot with bootstrap CIs
# using parallel processing.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor

def calculate_net_benefit_for_thresholds(y_true, p_calib, pt_grid):
    """
    Calculates net benefit across a grid of thresholds. This function is now
    public (no leading underscore) so it can be imported by other scripts.
    """
    benefits = []
    for pt in pt_grid:
        if pt >= 1.0:
            pt = 0.99999
        y_pred = p_calib >= pt
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        benefits.append((tp / n) - (fp / n) * (pt / (1 - pt)))
    return np.array(benefits)

def _dca_bootstrap_single_iteration(args):
    """Performs a single bootstrap resample for DCA (internal use only)."""
    y_true, p_calib, pt_grid, seed = args
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    
    idx = rng.integers(0, n_samples, n_samples)
    y_boot, p_boot = y_true[idx], p_calib[idx]
    
    if len(np.unique(y_boot)) < 2:
        return np.full(len(pt_grid), np.nan)
        
    # Call the public helper function
    return calculate_net_benefit_for_thresholds(y_boot, p_boot, pt_grid)

def generate_dca_plot(y_true, p_calib, tau_star, data_source_name, model_name, save_dir, B=1000, seed=0, title=None):
    """
    Generates and saves a Decision-Curve Analysis plot in parallel.
    
    Args:
        title (str, optional): A custom title for the plot. If None, a default is generated.
    """
    pt_grid = np.arange(0.01, 0.305, 0.005)
    n_samples = len(y_true)

    # 1. Calculate point estimates
    nb_model = calculate_net_benefit_for_thresholds(y_true, p_calib, pt_grid)
    prevalence = y_true.mean()
    nb_all = prevalence - (1 - prevalence) * (pt_grid / (1 - pt_grid))
    nb_none = np.zeros_like(pt_grid)

    # 2. Perform bootstrapping in parallel
    master_rng = np.random.default_rng(seed)
    worker_seeds = master_rng.integers(0, 2**32 - 1, B)
    args_list = [(y_true, p_calib, pt_grid, s) for s in worker_seeds]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results_iterator = executor.map(_dca_bootstrap_single_iteration, args_list)
        boot_nb_samples = np.array(list(results_iterator))

    # Calculate CIs
    ci_lower, ci_upper = np.nanpercentile(boot_nb_samples, [2.5, 97.5], axis=0)

    # 3. Create and save the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(pt_grid * 100, nb_model, color='blue', lw=2, label=f'{model_name} Model')
    plt.fill_between(pt_grid * 100, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.plot(pt_grid * 100, nb_all, color='black', lw=1.5, ls='--', label='Treat-All Strategy')
    plt.plot(pt_grid * 100, nb_none, color='grey', lw=1.5, ls='-', label='Treat-None Strategy')
    plt.axvline(x=tau_star * 100, color='red', ls=':', lw=2, label=f'F₂-Optimal Threshold (τ* = {tau_star:.2f})')
    
    plot_title = title if title else f'Decision-Curve Analysis: {model_name} ({data_source_name.upper()})'
    plt.title(plot_title, fontsize=16)
    
    plt.xlabel('Risk Threshold Probability (%)', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.xlim(1, 30)
    plt.ylim(min(nb_all.min(), nb_model.min(), -0.05), max(nb_all.max(), nb_model.max(), 0.2))
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    filename_plot = f'dca_curve_{data_source_name}_{safe_model_name}.png'
    save_path_plot = os.path.join(save_dir, filename_plot)
    plt.savefig(save_path_plot)
    plt.close()
    print(f"  -> Saved DCA plot to: {save_path_plot}")

    # 4. Create and save the data table
    dca_df = pd.DataFrame({
        'threshold_prob': pt_grid,
        'net_benefit_model': nb_model,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper,
        'net_benefit_treat_all': nb_all
    })
    filename_csv = f'dca_table_{data_source_name}_{safe_model_name}.csv'
    save_path_csv = os.path.join(save_dir, filename_csv)
    dca_df.to_csv(save_path_csv, index=False)
    print(f"  -> Saved DCA data to: {save_path_csv}")