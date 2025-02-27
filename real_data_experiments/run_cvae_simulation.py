#!/usr/bin/env python
import argparse
import configparser
import os
import time
import json

import torch
from torch import nn
from utils import CVAE, N_COMPONENTS  # Ensure utils.py is in your PYTHONPATH

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
sns.set_theme()

# -----------------------------------------------------------------------------
# 0. Load configuration for data directory
# -----------------------------------------------------------------------------
config = configparser.ConfigParser()
config.read('../config.ini')
DATA_DIR = config['PATHS']['DATA_DIR']  # Example: '/path/to/data'

# -----------------------------------------------------------------------------
# 1. Load the pre-trained CVAE model and define m_func
# -----------------------------------------------------------------------------
latent_dim = 32  # Small latent dimension to regularize learning

# Load the trained CVAE model and move it to GPU.
cvae_model = CVAE(input_dim=N_COMPONENTS, latent_dim=latent_dim).cuda()
cvae_model.load_state_dict(torch.load('cvae_model.pt'))
cvae_model.eval()

def m_func(y_input):
    """
    Uses the trained CVAE model to generate samples given a wealth level y_input.
    """
    cvae_model.eval()
    # Convert input to a torch tensor and add a singleton dimension.
    y_input = torch.tensor(y_input, dtype=torch.float32).cuda().unsqueeze(1)
    with torch.no_grad():
        # Sample random latent vectors
        z = torch.randn((len(y_input), latent_dim)).cuda()
        # Concatenate latent vectors with y_input and decode
        dec_input = torch.cat([z, y_input], dim=1)
        generated_X = cvae_model.decoder(dec_input).cpu().numpy()
    return generated_X

# -----------------------------------------------------------------------------
# 2. Data simulation and population generation functions
# -----------------------------------------------------------------------------
def generate_population(m_func, n_samples=10000, tau=0.2, base_alpha=1, base_beta=10, sigma_Y=0.05, sigma_M=0.5, seed=42):
    """
    Generates a synthetic population for causal inference simulations.
    """
    np.random.seed(seed)
    # Generate confounder X ~ N(0.5, 0.1)
    X = np.random.normal(0.5, 0.1, n_samples)
    # Generate treatment T ~ Bernoulli(p(X)) where p(X) is given by a logistic function
    p_T_given_X = 0.5 * 1 / (1 + np.exp(-X))
    T = np.random.binomial(1, p_T_given_X)
    
    # Modify the shape parameters based on the confounder (e.g., scale them)
    alpha = base_alpha + (X * 2)  # Modify alpha based on the confounder
    beta = base_beta + (X * 2)    # Modify beta based on the confounder

    # Generate Beta samples based on modified alpha and beta
    Y = np.array([np.random.beta(a, b) for a, b in zip(alpha, beta)]) + np.random.normal(0, sigma_Y, n_samples)
    Y = 0.8 * (Y - Y.min()) / (Y.max() - Y.min())
    Y += tau * T
    
    # Generate proxy M using m_func and add noise
    M = m_func(Y)
    M += np.random.normal(0, sigma_M, M.shape)
    population = pd.DataFrame({
        'X': X,
        'T': T,
        'Y': Y,
        'p_T_given_X': p_T_given_X,
        'M': list(M)
    })
    return population

# -----------------------------------------------------------------------------
# 3. Regression model, training, and calibration functions
# -----------------------------------------------------------------------------
# (A stub for plot_history so that train_model runs even if plotting is not needed.)
def plot_history(train_history, val_history, model_name):
    pass

class RegressionModel(nn.Module):
    def __init__(self, size='M'):
        super(RegressionModel, self).__init__()
        if size == 'S':
            self.fc = nn.Sequential(
                nn.Linear(N_COMPONENTS, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        elif size == 'M':
            self.fc = nn.Sequential(
                nn.Linear(N_COMPONENTS, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

    def forward(self, x):
        return self.fc(x).squeeze()

def mse_loss(pred, target):
    mse = nn.MSELoss()(pred, target)
    return mse, {'MSE': mse.item()}

def train_model(X_train, y_train, loss_fn, lr=1e-3, l2=1e-5, base_model=None, model_size='M', model_name=None, verbose=1, seed=42):
    """
    Train a regression model.
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    torch.manual_seed(seed)
    main_model = base_model if base_model else RegressionModel(model_size).cuda()
    main_optimizer = torch.optim.Adam(main_model.parameters(), lr=lr, weight_decay=l2)
    train_history = []
    val_history = []
    best_model = main_model.state_dict()
    best_loss = np.inf
    best_epoch = 0
    max_epochs = 50000
    for epoch in tqdm(range(max_epochs), disable=verbose < 1):
        main_optimizer.zero_grad()
        y_pred = main_model(X_train)
        loss, log_object = loss_fn(y_pred, y_train)
        loss.backward()
        main_optimizer.step()
        train_history.append(log_object)
        with torch.no_grad():
            y_pred_val = main_model(X_val)
            loss_val, log_object_val = loss_fn(y_pred_val, y_val)
        val_history.append(log_object_val)
        if loss_val < best_loss:
            best_model = main_model.state_dict()
            best_loss = loss_val.item()
            best_epoch = epoch
        # Early stopping if no improvement for 200 epochs
        if epoch - best_epoch > 200:
            break
    if best_epoch == epoch:
        print('Terminated before early stopping.')
    if verbose > 1:
        plot_history(train_history, val_history, model_name)
    main_model.load_state_dict(best_model)
    return main_model

class CDF_model:
    """
    Fits a regression model and calibrates its residuals to produce a conditional prediction distribution.
    """
    def __init__(self, size='M'):
        self.model = RegressionModel(size).cuda()

    def fit(self, M_train, y_train, seed=42, verbose=-1):
        M_train = torch.tensor(M_train, dtype=torch.float32).cuda()
        y_train = torch.tensor(y_train, dtype=torch.float32).cuda()
        self.model = train_model(M_train, y_train, mse_loss, lr=1e-3, l2=1e-3, base_model=self.model, verbose=verbose, seed=seed)

    def calibrate(self, M_cal, y_cal):
        with torch.no_grad():
            y_pred = self.model(torch.tensor(M_cal, dtype=torch.float32).cuda()).cpu().numpy()
        # Fit a simple linear calibration model
        slope_model = LinearRegression().fit(y_cal.reshape(-1, 1), y_pred)
        self.m = slope_model.intercept_
        self.k = slope_model.coef_[0]
        # Correct predictions and store sorted residuals
        y_pred_corrected = (y_pred - self.m) / self.k
        self.residuals = np.sort(y_cal - y_pred_corrected)

    def predict(self, M_test):
        with torch.no_grad():
            y_pred = self.model(torch.tensor(M_test, dtype=torch.float32).cuda()).cpu().numpy()
        y_pred = (y_pred - self.m) / self.k
        # Expand predictions to have one column per residual (to form a CDF)
        y_pred = np.repeat(y_pred[:, np.newaxis], len(self.residuals), axis=1)
        # Add the sorted residuals to obtain a distribution of predictions
        y_pred += self.residuals
        return y_pred

# -----------------------------------------------------------------------------
# 4. ATE estimation helper functions
# -----------------------------------------------------------------------------
def get_ate(t, y_preds, p_t):
    """
    Calculate the Average Treatment Effect (ATE) using Inverse Probability of Treatment Weighting (IPTW).
    """
    weights_treated = t / p_t
    weights_control = (1 - t) / (1 - p_t)
    weighted_outcome_treated = weights_treated * y_preds
    weighted_outcome_control = weights_control * y_preds
    return np.mean(weighted_outcome_treated - weighted_outcome_control)

def fit_and_calibrate_model(M_train, Y_train, seed=42):
    """
    Fits and calibrates the CDF_model on training data.
    """
    M_train, M_cal, Y_train, Y_cal = train_test_split(M_train, Y_train, test_size=0.25, random_state=seed)
    model = CDF_model()
    model.fit(M_train, Y_train, seed=seed)
    model.calibrate(M_cal, Y_cal)
    return model

def estimate_ate_with_model(model, M_trial, t_trial, p_T_trial, n_cdf_rounds=1000, seed=42):
    """
    Estimate the ATE using draws from the predictive distribution.
    """
    y_pred_dists = model.predict(M_trial)
    n_samples, n_quantiles = y_pred_dists.shape
    ate_estimates = []
    np.random.seed(seed)
    for i in range(n_cdf_rounds):
        indices = np.random.randint(0, n_quantiles, n_samples)
        draw_y_preds = y_pred_dists[np.arange(n_samples), indices]
        ate_estimate = get_ate(t_trial, draw_y_preds, p_T_trial)
        ate_estimates.append(ate_estimate)
    return ate_estimates

def calculate_errors(ate_estimates):
    """
    Calculate modeling and sampling errors from the bootstrap ATE estimates.
    """
    # Modeling error: variance within each bootstrap sample
    row_variances = ate_estimates.var(axis=1)
    modeling_error = np.sqrt(row_variances.mean())
    # Sampling error: variance across bootstrap sample means
    row_means = ate_estimates.mean(axis=1)
    sampling_error = np.sqrt(row_means.var(ddof=1))
    return modeling_error, sampling_error

# -----------------------------------------------------------------------------
# 5. Main simulation procedure
# -----------------------------------------------------------------------------
def main(RANDOM_STATE, sigma_M):

    # Simulation parameters
    tau = 0.2
    n_train_samples = 60000
    n_trial_samples = 10000

    print("Generating training population with seed =", RANDOM_STATE)
    train_pop = generate_population(m_func, tau=tau, n_samples=n_train_samples, sigma_M=sigma_M, seed=RANDOM_STATE)
    print("Generating trial population with seed =", RANDOM_STATE + 1)
    trial_pop = generate_population(m_func, tau=tau, n_samples=n_trial_samples, sigma_M=sigma_M, seed=RANDOM_STATE + 1)

    # Prepare training and trial datasets
    D_train = (np.vstack(train_pop['M'].values), train_pop['Y'].values)
    D_trial = (np.vstack(trial_pop['M'].values), trial_pop['T'].values, trial_pop['p_T_given_X'].values)
    
    target_tau = get_ate(trial_pop['T'].values, trial_pop['Y'].values, trial_pop['p_T_given_X'].values)
    print("Estimated true ATE (using true labels) of trial data:", target_tau)
    
    n_boots = 1000  # Number of bootstrap samples
    ate_estimates_all = []
    bootstrap_trial = True

    # Phase 1: Fit and calibrate the model on training data.
    model = fit_and_calibrate_model(*D_train, seed=RANDOM_STATE)
    
    # Evaluate R^2 score on trial set
    with torch.no_grad():
        M_trial_tensor = torch.tensor(D_trial[0], dtype=torch.float32).cuda()
        y_pred_trial = model.model(M_trial_tensor).cpu().numpy()
        y_pred_trial = (y_pred_trial - model.m) / model.k

    y_true_trial = trial_pop['Y'].values
    r2_score = 1 - np.sum((y_true_trial - y_pred_trial) ** 2) / np.sum((y_true_trial - np.mean(y_true_trial)) ** 2)
    print("R^2 score on trial set:", r2_score)

    # Phase 2: Run bootstrap procedure to obtain a distribution of ATE estimates.
    for i in tqdm(range(n_boots)):
        if bootstrap_trial:
            M_trial, t_trial, p_T_trial = resample(*D_trial, random_state=i, n_samples=n_trial_samples)
        boot_ate_ests = estimate_ate_with_model(model, M_trial, t_trial, p_T_trial, seed=i)
        ate_estimates_all.append(boot_ate_ests)
    
    ate_estimates_all = np.asarray(ate_estimates_all)
    modeling_error, sampling_error = calculate_errors(ate_estimates_all)
    global_mean = ate_estimates_all.mean()
    z = norm.ppf(0.95)
    
    # Confidence intervals
    ci_sampling_only = (global_mean - z * sampling_error, global_mean + z * sampling_error)
    total_error = np.sqrt(sampling_error**2 + modeling_error**2)
    ci_with_both_errors = (global_mean - z * total_error, global_mean + z * total_error)
    
    # Print results
    print('Mean estimated ATE:', global_mean)
    print('Sampling error:', sampling_error)
    print('Modeling error:', modeling_error)
    print(f'90% CI (sampling error only): ({ci_sampling_only[0]:.3f} - {ci_sampling_only[1]:.3f})')
    print(f'90% CI (both errors): ({ci_with_both_errors[0]:.3f} - {ci_with_both_errors[1]:.3f})')
    
    # -----------------------------------------------------------------------------
    # 6. Save results to a unique run directory under DATA_DIR
    # -----------------------------------------------------------------------------
    # Create a unique run id based on the random state, sigma_M, and current timestamp.
    run_id = f"run_{RANDOM_STATE}_sigmaM_{sigma_M}_{int(time.time())}"
    run_dir = os.path.join(DATA_DIR, 'simulation_runs', run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save ATE estimates
    output_path_ate = os.path.join(run_dir, "ate_estimates.npy")
    np.save(output_path_ate, ate_estimates_all)
    print("Saved ATE estimates to", output_path_ate)
    
    # Save additional results
    results = {
        'r2_score': r2_score,
        'sigma_M': sigma_M,
        'k': model.k,
        'm': model.m,
        'modeling_error': modeling_error,
        'sampling_error': sampling_error,
        'global_mean': global_mean
    }
    output_path_results = os.path.join(run_dir, "results.json")
    with open(output_path_results, 'w') as f:
        json.dump(results, f)
    print("Saved additional results to", output_path_results)

# -----------------------------------------------------------------------------
# 7. Parse command-line arguments and run main()
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CVAE-based causal inference simulation and estimate ATE."
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility and parallel runs (default: 42)"
    )
    parser.add_argument(
        "--sigma_M", type=float, default=0.5,
        help="Standard deviation of the noise added to the proxy M (default: 0.5)"
    )
    args = parser.parse_args()
    main(args.random_state, args.sigma_M)
