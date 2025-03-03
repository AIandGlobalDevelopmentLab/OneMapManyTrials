import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from sklearn.utils import resample
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Data simulation and population generation functions
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
# 2. Regression model, training, and calibration functions
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

    def point_predict(self, X_test):
        # Predict the outcomes for the test set
        y_pred = self.biased_model.predict(X_test)
        
        # Correct predictions
        y_pred = (y_pred - self.m) / self.k
        
        return y_pred
        
    def predict(self, X_test):
        # Predict the outcomes for the test set
        y_pred = self.point_predict(X_test)
        
        # Repeat the predictions to match the number of residuals
        y_pred = np.repeat(y_pred[:, np.newaxis], len(self.residuals), axis=1)
        
        # Add the sorted residuals to the predictions. This will give us a distribution of predictions
        y_pred += self.residuals
        
        return y_pred

# -----------------------------------------------------------------------------
# 3. ATE estimation helper functions
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
