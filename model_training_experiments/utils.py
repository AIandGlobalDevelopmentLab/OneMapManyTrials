import os
import configparser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sns.set_theme()

RANDOM_STATE = 42
N_COMPONENTS = 232


def load_data():

    # Read config file
    config = configparser.ConfigParser()
    config.read('../config.ini')

    DATA_DIR = config['PATHS']['DATA_DIR']

    embeddings = np.load(os.path.join(DATA_DIR, 'small_ssl4eo_resnet50.npy'))
    df = pd.read_csv(os.path.join(DATA_DIR, 'small_dhs.csv'))

    assert embeddings.shape[0] == df.shape[0], 'Mismatch between embeddings and metadata rows.'
    
    X = embeddings

    # Cheat by doing rescaling and PCA on the whole set
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=N_COMPONENTS)
    X = pca.fit_transform(X)

    y = df['iwi'].values / 100.0 # Scale IWI to [0, 1]
    y = y.clip(0, 1)

    X = torch.tensor(X, dtype=torch.float).cuda()
    y = torch.tensor(y, dtype=torch.float).cuda()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# Main regression model
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

# ================ Training ==================

def train_model(X_train, X_val, y_train, y_val, loss_fn, lr=1e-3, l2=1e-5, model_size='M', model_name=None, verbose=1):
    
    torch.manual_seed(RANDOM_STATE)

    # Instantiate model
    main_model = RegressionModel(model_size).cuda()

    # Optimizer
    main_optimizer = optim.Adam(main_model.parameters(), lr=lr, weight_decay=l2)
    
    train_history = []
    val_history = []
    
    best_model = main_model.state_dict()
    best_loss = np.inf
    best_epoch = 0
    
    # Training loop
    max_epochs = 50000
    for epoch in tqdm(range(max_epochs), disable=verbose<1):

        # Train main model
        main_optimizer.zero_grad()
        y_pred = main_model(X_train)
        loss, log_object = loss_fn(y_pred, y_train)
        loss.backward()
        main_optimizer.step()
        
        train_history.append(log_object)
        
        with torch.no_grad():
            y_pred = main_model(X_val)
            loss, log_object = loss_fn(y_pred, y_val)
        
            val_history.append(log_object)
            
            # Save best model
            if loss < best_loss:
                best_model = main_model.state_dict()
                best_loss = loss.item()
                best_epoch = epoch
            
            # Early stopping
            if epoch - best_epoch > 20:
                break
                
    if best_epoch == epoch:
        print('Terminated before early stopping.')
            
    if verbose:
        plot_history(train_history, val_history, model_name)
    
    # Load best model
    main_model.load_state_dict(best_model)
    return main_model

# ================ Metrics ==================
    
def get_slope_model(targets, predictions):
    
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    slope_model = LinearRegression().fit(targets.reshape(-1, 1), predictions)
    
    return slope_model

# ================ Plotting ==================

def plot_history(train_history, val_history, model_name=None):
    
    loss_types = train_history[0].keys()
    
    train_his_df = pd.DataFrame(train_history)
    train_his_df['Epoch'] = train_his_df.index
    train_his_df['Set'] = 'Training'

    val_his_df = pd.DataFrame(val_history)
    val_his_df['Epoch'] = val_his_df.index
    val_his_df['Set'] = 'Validation'

    history_df = pd.concat([train_his_df, val_his_df], ignore_index=True)
    
    title = 'Losses during Training and Validation'
    if model_name:
        title = title + ' of ' + model_name
    
    n_losses = len(loss_types)
    
    if n_losses > 1:
        fig, axs = plt.subplots(1, n_losses, figsize=(18, 5))
        
        for i, loss_type in enumerate(loss_types):
            sns.lineplot(data=history_df, x='Epoch', y=loss_type, hue='Set', ax=axs[i])
            axs[i].set_yscale('log')
            
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.85)
        plt.tight_layout()
        plt.show()
    else:
        loss_type = next(iter(loss_types))
        sns.lineplot(data=history_df, x='Epoch', y=loss_type, hue='Set')
        plt.yscale('log')
        plt.title(title)
        plt.show()

def parity_plot(targets, predictions, title=None, ax=None):
    slope_model = get_slope_model(targets, predictions)
    
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()
    should_plot = False

    # Create a new figure and axis if none is provided
    if ax is None:
        should_plot = True
        fig, ax = plt.subplots()

    ax.scatter(targets, predictions, alpha=0.2)

    # Plot regression lines
    x = np.asarray([min(targets), max(targets)])
    ideal_y = x
    true_y = slope_model.predict(x.reshape(-1, 1))

    # Compute statistics and format the string
    stats_text = f'b={slope_model.coef_[0]:.2f}, r2={r2_score(targets, predictions):.2f}'

    ax.plot(x, ideal_y, 'k--', label='Ideal')
    ax.plot(x, true_y, c=sns.color_palette()[1], label='Fitted')
    ax.set_xlabel('Targets')
    ax.set_ylabel('Predictions')
    ax.set_title(title)
    ax.legend()

    # Add the text in the bottom-right corner
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)  # Ensure the box is square
    ax.text(
        0.95, 0.05, stats_text,
        transform=ax.transAxes,
        fontsize=10, color='black', ha='right', va='bottom'
    )

    # Show the plot if no external axis was provided
    if should_plot:
        plt.show()
    
def resid_plot(targets, predictions, title, ax=None):
    
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()
    should_plot = False

    # Create a new figure and axis if none is provided
    if ax is None:
        should_plot = True
        fig, ax = plt.subplots()

    ax.scatter(targets, (targets - predictions), alpha=0.2)
    ax.axhline(0, ls=":", c=".2")
    ax.set_xlabel('Targets')
    ax.set_ylabel('Residuals')
    ax.set_title(title)

    if should_plot:
        plt.show()
        
        
def plot_results(targets, predictions, model_name):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    
    parity_plot(targets, predictions, title=f'Parity plot of {model_name}', ax=ax0)
    resid_plot(targets, predictions, f'Residuals of {model_name}', ax=ax1)
    fig.show()