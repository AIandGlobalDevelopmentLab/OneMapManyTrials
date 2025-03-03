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
from utils import generate_population, CDF_model, get_ate, estimate_ate_with_pred_dists, calculate_errors

# Load configuration for data directory
config = configparser.ConfigParser()
config.read('config.ini')
DATA_DIR = config['PATHS']['DATA_DIR']

def img_proxy_func(Y):
    return np.vstack([2 * Y, 3 * Y, -4 * Y]).T

def fit_and_calibrate_model(X_train: np.ndarray, Y_train: np.ndarray, seed: int = 42) -> CDF_model:
    """
    Fits and calibrates the CDF_model on training data.

    Parameters:
    X_train (np.ndarray): Training data features.
    Y_train (np.ndarray): Training data labels.
    seed (int): Random seed for reproducibility (default: 42).

    Returns:
    CDF_model: The fitted and calibrated CDF_model.
    """
    X_train, X_cal, Y_train, Y_cal = train_test_split(X_train, Y_train, test_size=0.25, random_state=seed)
    model = CDF_model()
    model.fit(X_train, Y_train)
    model.calibrate(X_cal, Y_cal)
    return model

def generate_populations(RANDOM_STATE, sigma_X, n_train_samples, n_trial_samples, tau):
    """
    Generates training and trial populations.

    Parameters:
    RANDOM_STATE (int): Random seed for reproducibility.
    sigma_X (float): Standard deviation of the noise added to the proxy X.
    n_train_samples (int): Number of training samples to generate.
    n_trial_samples (int): Number of trial samples to generate.
    tau (float): True Average Treatment Effect (ATE) of treatment A.

    Returns:
    tuple: A tuple containing the training population and trial population.
    """
    print("Generating training population with seed =", RANDOM_STATE)
    train_pop = generate_population(img_proxy_func, tau=tau, n_samples=n_train_samples, sigma_X=sigma_X, seed=RANDOM_STATE)
    print("Generating trial population with seed =", RANDOM_STATE + 1)
    trial_pop = generate_population(img_proxy_func, tau=tau, n_samples=n_trial_samples, sigma_X=sigma_X, seed=RANDOM_STATE + 1)
    return train_pop, trial_pop

def extract_data(train_pop, trial_pop):
    """
    Extracts data from the training and trial populations.

    Parameters:
    train_pop (pd.DataFrame): Training population.
    trial_pop (pd.DataFrame): Trial population.

    Returns:
    tuple: A tuple containing the training data and trial data.
    """

    X_train = np.vstack(train_pop['X'].values)
    Y_train = train_pop['Y'].values
    X_trial = np.vstack(trial_pop['X'].values)
    A_trial = trial_pop['A'].values
    p_A_trial = trial_pop['p_A_given_C'].values
    Y_trial = trial_pop['Y'].values
    return X_train, Y_train, X_trial, A_trial, p_A_trial, Y_trial

def evaluate_model(model, X_trial, Y_trial):
    """
    Evaluates the model on the trial set using R^2 score.

    Parameters:
    model (CDF_model): The fitted and calibrated CDF_model.
    X_trial (np.ndarray): Trial data features.
    Y_trial (np.ndarray): Trial data labels.

    Returns:
    float: R^2 score of the model on the trial set.
    """
    y_pred_trial = model.biased_model.predict(X_trial)
    y_pred_trial = (y_pred_trial - model.m) / model.k
    r2_score = 1 - np.sum((Y_trial - y_pred_trial) ** 2) / np.sum((Y_trial - np.mean(Y_trial)) ** 2)
    return r2_score

def bootstrap_ate_estimates(model, X_trial, A_trial, p_A_trial, Y_trial, n_trial_samples, n_boots=1000):
    """
    Bootstraps the Average Treatment Effect (ATE) estimates using the model.

    Parameters:
    model (CDF_model): The fitted and calibrated CDF_model.
    X_trial (np.ndarray): Trial data features.
    A_trial (np.ndarray): Trial data treatment assignments.
    p_A_trial (np.ndarray): Trial data treatment probabilities.
    Y_trial (np.ndarray): Trial data labels.
    n_trial_samples (int): Number of trial samples to resample.
    n_boots (int): Number of bootstrap samples (default: 1000).

    Returns:
    tuple: A tuple containing the bootstrapped ATE estimates, point ATE estimates, and sample ATE estimates.
    """

    y_pred_dists = model.predict(X_trial)
    y_pred_trial = model.point_predict(X_trial)
    ate_estimates_all = []
    point_ate_estimates = []
    sample_ate_estimates = []

    for i in tqdm(range(n_boots)):
        # Resample trial data
        y_pred_dists_i, y_pred_trial_i, A_trial_i, p_A_trial_i, Y_trial_i = resample(y_pred_dists, y_pred_trial, A_trial, p_A_trial, Y_trial, random_state=i, n_samples=n_trial_samples)

        # Estimate ATE using predicted distributions
        boot_ate_ests = estimate_ate_with_pred_dists(y_pred_dists_i, A_trial_i, p_A_trial_i, seed=i)

        # Estimate ATE using point predictions and true labels
        boot_point_ate_est = get_ate(A_trial_i, y_pred_trial_i, p_A_trial_i)
        boot_sample_ate_est = get_ate(A_trial_i, Y_trial_i, p_A_trial_i)

        ate_estimates_all.append(boot_ate_ests)
        point_ate_estimates.append(boot_point_ate_est)
        sample_ate_estimates.append(boot_sample_ate_est)
    
    return np.asarray(ate_estimates_all), point_ate_estimates, sample_ate_estimates

def print_results(global_mean, sampling_error, modeling_error):
    z = norm.ppf(0.95)
    ci_sampling_only = (global_mean - z * sampling_error, global_mean + z * sampling_error)
    total_error = np.sqrt(sampling_error**2 + modeling_error**2)
    ci_with_both_errors = (global_mean - z * total_error, global_mean + z * total_error)
    print('Mean estimated ATE:', global_mean)
    print('Sampling error:', sampling_error)
    print('Modeling error:', modeling_error)
    print(f'90% CI (sampling error only): ({ci_sampling_only[0]:.3f} - {ci_sampling_only[1]:.3f})')
    print(f'90% CI (both errors): ({ci_with_both_errors[0]:.3f} - {ci_with_both_errors[1]:.3f})')

def save_results(RANDOM_STATE, sigma_X, n_train_samples, n_trial_samples, tau, sample_tau, r2_score, model, 
                 modeling_error, sampling_error, global_mean, point_ate_estimates, sample_ate_estimates, ate_estimates_all):
    """
    Save the results of the simulation to disk.

    Parameters:
    RANDOM_STATE (int): Random seed for reproducibility.
    sigma_X (float): Standard deviation of the noise added to the proxy X.
    n_train_samples (int): Number of training samples to generate.
    n_trial_samples (int): Number of trial samples to generate.
    tau (float): True Average Treatment Effect (ATE) of treatment A.
    sample_tau (float): Estimated true ATE (using true labels) of trial data.
    r2_score (float): R^2 score of the model on the trial set.
    model (CDF_model): The fitted and calibrated CDF_model.
    modeling_error (float): Modeling error of the ATE estimates.
    sampling_error (float): Sampling error of the ATE estimates.
    global_mean (float): Mean estimated ATE.
    point_ate_estimates (list): Point ATE estimates.
    sample_ate_estimates (list): True label ATE estimates.
    ate_estimates_all (np.ndarray): Bootstrapped ATE estimates.

    Returns:
    None
    """
    run_id = f"run_{RANDOM_STATE}_sigmaX_{sigma_X}_ntrain_{n_train_samples}_ntrial_{n_trial_samples}"
    run_dir = os.path.join(DATA_DIR, 'simulation_runs', run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Saving ATE estimate matrix as numpy array
    output_path_ate = os.path.join(run_dir, "ate_estimates.npy")
    np.save(output_path_ate, ate_estimates_all)
    print("Saved ATE estimates to", output_path_ate)

    # Saving additional results as JSON
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

def main(RANDOM_STATE, sigma_X, n_train_samples, n_trial_samples, tau):
    """
    Main function to run the simulation and estimate the Average Treatment Effect (ATE).

    Parameters:
    RANDOM_STATE (int): Random seed for reproducibility.
    sigma_X (float): Standard deviation of the noise added to the proxy X.
    n_train_samples (int): Number of training samples to generate.
    n_trial_samples (int): Number of trial samples to generate.
    tau (float): True Average Treatment Effect (ATE) of treatment A.

    Returns:
    None
    """
    train_pop, trial_pop = generate_populations(RANDOM_STATE, sigma_X, n_train_samples, n_trial_samples, tau)
    X_train, Y_train, X_trial, A_trial, p_A_trial, Y_trial = extract_data(train_pop, trial_pop)
    sample_tau = get_ate(A_trial, Y_trial, p_A_trial)
    print("Estimated true ATE (using true labels) of trial data:", sample_tau)

    model = fit_and_calibrate_model(X_train, Y_train, seed=RANDOM_STATE)
    r2_score = evaluate_model(model, X_trial, Y_trial)
    print("R^2 score on trial set:", r2_score)

    ate_estimates_all, point_ate_estimates, sample_ate_estimates = bootstrap_ate_estimates(model, X_trial, A_trial, p_A_trial, Y_trial, n_trial_samples)
    modeling_error, sampling_error = calculate_errors(ate_estimates_all)
    global_mean = np.mean(ate_estimates_all)
    print_results(global_mean, sampling_error, modeling_error)

    save_results(RANDOM_STATE, sigma_X, n_train_samples, n_trial_samples, tau, sample_tau, r2_score, model, modeling_error, sampling_error, global_mean, point_ate_estimates, sample_ate_estimates, ate_estimates_all)

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
    parser.add_argument(
        "--n_train_samples", type=int, default=60000,
        help="Number of training samples to generate (default: 60000)"
    )
    parser.add_argument(
        "--n_trial_samples", type=int, default=10000,
        help="Number of trial samples to generate (default: 10000)"
    )
    parser.add_argument(
        "--tau", type=float, default=0.2,
        help="True Average Treatment Effect (ATE) of treatment A (default: 0.2)"
    )
    args = parser.parse_args()
    main(args.random_state, args.sigma_X, args.n_train_samples, args.n_trial_samples, args.tau)
