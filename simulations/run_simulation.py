#!/usr/bin/env python
import argparse
import configparser
import os
import time
import json
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 0. Load configuration for data directory
# -----------------------------------------------------------------------------
config = configparser.ConfigParser()
config.read('../config.ini')
DATA_DIR = config['PATHS']['DATA_DIR']

# -----------------------------------------------------------------------------
# 1. Define img_proxy_func
# -----------------------------------------------------------------------------
img_proxy_func = lambda Y: np.vstack([2 * Y, 3 * Y, -4 * Y]).T

# -----------------------------------------------------------------------------
# 2. Data simulation and population generation functions
# -----------------------------------------------------------------------------
def generate_population(img_proxy_func, n_samples=10000, tau=0.2, alpha=1.0, sigma_Y=1.0, sigma_X=5.0, seed=42):
    """
    Generates a synthetic population for causal inference simulations.

    Parameters:
    img_proxy_func (function): A function that takes Y as input and returns an image proxy X.
    n_samples (int, optional): Number of samples to generate. Default is 10000.
    tau (float, optional): True Average Treatment Effect (ATE) of treatment A. Default is 0.2.
    alpha (float, optional): Effect of confounder C on outcome Y. Default is 1.0.
    sigma_Y (float, optional): Standard deviation of noise in outcome Y. Default is 1.0.
    sigma_X (float, optional): Standard deviation of noise in proxy X. Default is 0.5.
    seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
    pd.DataFrame: A DataFrame containing the generated population with columns:
        - 'C': Confounder values.
        - 'A': Treatment assignment.
        - 'Y': Outcome values.
        - 'p_A_given_C': Probability of treatment given confounder C.
        - 'X': Image proxy for outcome Y.
    """

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate confounder C ~ N(0, 1)
    C = np.random.normal(0, 1, n_samples)

    # Generate treatment A ~ Bernoulli(p(C)) where p(C) is a logistic function of C
    p_A_given_C = 1 / (1 + np.exp(-C))  # Logistic function
    A = np.random.binomial(1, p_A_given_C)

    # Generate outcome Y ~ N(tau * A + alpha * C, sigma_Y)
    Y = tau * A + alpha * C + np.random.normal(0, sigma_Y, n_samples)

    # Generate proxy X = img_proxy_func(Y) + noise
    X = img_proxy_func(Y)
    X += np.random.normal(0, sigma_X, X.shape)

    population = pd.DataFrame({
        'C': C,
        'A': A,
        'Y': Y,
        'p_A_given_C': p_A_given_C,
        'X': list(X)
    })

    return population

# -----------------------------------------------------------------------------
# 3. Regression model, training, and calibration functions
# -----------------------------------------------------------------------------
class CDF_model:
    
    def __init__(self):
        self.biased_model = LinearRegression()
        
    def fit(self, X_train, y_train):
        
        # Use unconstrained OLS as initial guess for beta
        beta_ols = self.biased_model.fit(X_train, y_train)
        
    def calibrate(self, X_cal, y_cal):
        
        y_pred = self.biased_model.predict(X_cal)
        
        # Fit slope model
        slope_model = LinearRegression().fit(y_cal.reshape(-1, 1), y_pred)
        self.m = slope_model.intercept_
        self.k = slope_model.coef_[0]
        
        # Correct predictions
        y_pred = (y_pred - self.m) / self.k
        
        # Calculate and sort the residuals
        self.residuals = np.sort(y_cal - y_pred)
        
    def predict(self, X_test):
        # Predict the outcomes for the test set
        y_pred = self.biased_model.predict(X_test)
        
        # Correct predictions
        y_pred = (y_pred - self.m) / self.k
        
        # Repeat the predictions to match the number of residuals
        y_pred = np.repeat(y_pred[:, np.newaxis], len(self.residuals), axis=1)
        
        # Add the sorted residuals to the predictions. This will give us a distribution of predictions
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

def fit_and_calibrate_model(X_train, Y_train, seed=42):
    """
    Fits and calibrates the CDF_model on training data.
    """
    X_train, X_cal, Y_train, Y_cal = train_test_split(X_train, Y_train, test_size=0.25, random_state=seed)
    model = CDF_model()
    model.fit(X_train, Y_train)
    model.calibrate(X_cal, Y_cal)
    return model

def estimate_ate_with_pred_dists(y_pred_dists, t_trial, p_A_trial, n_cdf_rounds=1000, seed=42):
    """
    Estimate the ATE using draws from the predictive distribution.
    """
    n_samples, n_quantiles = y_pred_dists.shape
    ate_estimates = []
    np.random.seed(seed)
    for i in range(n_cdf_rounds):
        indices = np.random.randint(0, n_quantiles, n_samples)
        draw_y_preds = y_pred_dists[np.arange(n_samples), indices]
        ate_estimate = get_ate(t_trial, draw_y_preds, p_A_trial)
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
def main(RANDOM_STATE, sigma_X):
    """
    Main function to run the simulation and estimate the Average Treatment Effect (ATE).

    Parameters:
    RANDOM_STATE (int): Random seed for reproducibility.
    sigma_X (float): Standard deviation of the noise added to the proxy X.

    Returns:
    None
    """

    # Simulation parameters
    tau = 0.2
    n_train_samples = 60000
    n_trial_samples = 10000

    print("Generating training population with seed =", RANDOM_STATE)
    train_pop = generate_population(img_proxy_func, tau=tau, n_samples=n_train_samples, sigma_X=sigma_X, seed=RANDOM_STATE)
    print("Generating trial population with seed =", RANDOM_STATE + 1)
    trial_pop = generate_population(img_proxy_func, tau=tau, n_samples=n_trial_samples, sigma_X=sigma_X, seed=RANDOM_STATE + 1)
    
    X_train = np.vstack(train_pop['X'].values)
    Y_train = train_pop['Y'].values
    
    X_trial = np.vstack(trial_pop['X'].values)
    A_trial = trial_pop['A'].values
    p_A_trial = trial_pop['p_A_given_C'].values
    Y_trial = trial_pop['Y'].values  # Only used for evaluating R^2
    
    sample_tau = get_ate(A_trial, Y_trial, p_A_trial)
    print("Estimated true ATE (using true labels) of trial data:", sample_tau)

    # Fit and calibrate the model on training data.
    model = fit_and_calibrate_model(X_train, Y_train, seed=RANDOM_STATE)
    
    # Evaluate R^2 score on trial set
    y_pred_trial = model.biased_model.predict(X_trial)
    y_pred_trial = (y_pred_trial - model.m) / model.k

    r2_score = 1 - np.sum((Y_trial - y_pred_trial) ** 2) / np.sum((Y_trial - np.mean(Y_trial)) ** 2)
    print("R^2 score on trial set:", r2_score)
    
    # Run bootstrap procedure to obtain a distribution of ATE estimates on trial data.
    y_pred_dists = model.predict(X_trial)
    
    n_boots = 1000  # Number of bootstrap samples
    ate_estimates_all = [] # ATE estimates using predictive distributions
    point_ate_estimates = [] # ATE estimates using point predictions
    sample_ate_estimates = [] # ATE estimates using true labels
    for i in tqdm(range(n_boots)):
        y_pred_dists_i, y_pred_trial_i, A_trial_i, p_A_trial_i, Y_trial_i = resample(y_pred_dists, y_pred_trial, A_trial, p_A_trial, Y_trial, random_state=i, n_samples=n_trial_samples)
        boot_ate_ests = estimate_ate_with_pred_dists(y_pred_dists_i, A_trial_i, p_A_trial_i, seed=i)
        boot_point_ate_est = get_ate(A_trial_i, y_pred_trial_i, p_A_trial_i)
        boot_sample_ate_est = get_ate(A_trial_i, Y_trial_i, p_A_trial_i)
        ate_estimates_all.append(boot_ate_ests)
        point_ate_estimates.append(boot_point_ate_est)
        sample_ate_estimates.append(boot_sample_ate_est)
    
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
    # Create a unique run id based on the random state, sigma_X, and current timestamp.
    run_id = f"run_{RANDOM_STATE}_sigmaX_{sigma_X}_{int(time.time())}"
    run_dir = os.path.join(DATA_DIR, 'simulation_with_true_runs', run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save ATE estimates
    output_path_ate = os.path.join(run_dir, "ate_estimates.npy")
    np.save(output_path_ate, ate_estimates_all)
    print("Saved ATE estimates to", output_path_ate)
    
    # Save additional results
    results = {
        'true_tau': tau,
        'n_train_samples': n_train_samples,
        'n_trial_samples': n_trial_samples,
        'sample_tau': sample_tau,
        'r2_score': r2_score,
        'sigma_X': sigma_X,
        'k': model.k,
        'm': model.m,
        'modeling_error': modeling_error,
        'sampling_error': sampling_error,
        'global_mean': global_mean,
        'point_ate_estimates': point_ate_estimates,
        'sample_ate_estimates': sample_ate_estimates
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
        description="Run a simulation to estimate the Average Treatment Effect (ATE) with and without modeling error using our pipeline."
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility and parallel runs (default: 42)"
    )
    parser.add_argument(
        "--sigma_X", type=float, default=5.0,
        help="Standard deviation of the noise added to the proxy X (default: 5.0)"
    )
    args = parser.parse_args()
    main(args.random_state, args.sigma_X)
