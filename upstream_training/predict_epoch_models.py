import timm
import json
import os
import torch
import pandas as pd
import numpy as np
import configparser
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import gaussian_kde
import argparse
import h5py
from time import time

# Read config file
config = configparser.ConfigParser()
config.read('../config.ini')

DATA_DIR = config['PATHS']['DATA_DIR']

# Set up PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

def get_save_dir_name(args):
    # Define the default values as per argparse
    defaults = {
        'loss': 'mse',
        'batch_size': 128,
        'lr': 1e-4,
        'lambda_r': 1e-4,
        'lambda_b': 5,
        'num_linear_epochs': 3,
        'num_top_epochs': 20,
        'num_full_epochs': 0,
        'seed': 42
    }

    # Start with the loss name
    parts = [args.loss]

    # Iterate over args and compare with defaults
    for key, default_value in defaults.items():
        value = getattr(args, key)
        if value != default_value and key != 'loss':  # Skip 'loss' since it's already added
            parts.append(f"{key}={value}")

    return '_'.join(parts)

def init_model():

    repo_id = "torchgeo/ssl4eo_landsat"
    filename = "resnet50_landsat_etm_sr_moco-1266cde3.pth"

    # Download the model weights
    backbone_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Create backbone model
    state_dict = torch.load(backbone_path)
    model = timm.create_model("resnet50", in_chans=6, num_classes=0)
    model.load_state_dict(state_dict)

    # Define model with regression head
    model.fc = nn.Linear(2048, 1)

    # Use channels_last memory format for better performance on GPUs
    model = model.to(device, memory_format=torch.channels_last)

    return model

class RegressionDataset(Dataset):
    def __init__(self, df, hdf5_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.h5_file = None  # Will be initialized lazily in __getitem__

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        cluster = self.df.iloc[idx]
        img = self.h5_file[cluster['cluster_id']][:].astype(np.float32)
        target = cluster['iwi']
        if self.transform:
            img = self.transform(img)
        return img, target

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

def get_dataloaders(df, hdf5_path, train_folds, val_fold, test_fold, batch_size=128, num_workers=16):

    # Get the indices for each fold
    train_folds = df[df['cv_fold'].isin(train_folds)].index.tolist()
    val_fold = df[df['cv_fold'] == val_fold].index.tolist()
    test_fold = df[df['cv_fold'] == test_fold].index.tolist()

    landsat_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 0.0000275 - 0.2),
        transforms.Lambda(lambda x: torch.clamp(x, 0.0, 0.3)),
        transforms.Lambda(lambda x: x / 0.3)
    ])

    train_transform = transforms.Compose([
        landsat_transform,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    train_dataset = RegressionDataset(df=df.iloc[train_folds], hdf5_path=hdf5_path, transform=train_transform)
    val_dataset = RegressionDataset(df=df.iloc[val_fold], hdf5_path=hdf5_path, transform=landsat_transform)
    test_dataset = RegressionDataset(df=df.iloc[test_fold], hdf5_path=hdf5_path, transform=landsat_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader

def score_function(y, kde, delta=1e-5):
        # Derivative of log density
        log_p_plus = kde.logpdf(y + delta)[0]
        log_p_minus = kde.logpdf(y - delta)[0]
        d_logp = (log_p_plus - log_p_minus) / (2 * delta)
        return d_logp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune SSL4EO-L ResNet50 model on DHS dataset for IWI prediction.')
    parser.add_argument('--loss', type=str, choices=['mse', 'ratledge'],
        help='Which loss function to use', default='mse')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--lambda_r', type=float, help='Regularization weight', default=1e-4)
    parser.add_argument('--lambda_b', type=float, help='Quantile MSE loss weight (Only for Ratledge loss)', default=5)
    parser.add_argument('--num_linear_epochs', type=int, help='Number of epochs for linear probe', default=3)
    parser.add_argument('--num_top_epochs', type=int, help='Number of epochs for unfreezing top ResNet block', default=20)
    parser.add_argument('--num_full_epochs', type=int, help='Number of epochs for full training', default=0)
    parser.add_argument('--save_epoch', type=int, help='Epoch checkpoint to load', default=0)
    parser.add_argument('--seed', type=int, help='Seed for random initialization and shuffling', default=42)
    args = parser.parse_args()

    # Set seeds for reproducibility
    RANDOM_STATE = args.seed
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Create a save directory
    MODEL_DIR = os.path.join(DATA_DIR, 'models', 'mse_num_linear_epochs=0_num_top_epochs=0_num_full_epochs=20')
    SAVE_DIR = os.path.join(MODEL_DIR, f'save_epoch_{args.save_epoch}')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Load the dataset
    df = pd.read_csv(os.path.join(DATA_DIR, 'dhs_with_imgs.csv'))
    hdf5_path = os.path.join(DATA_DIR, 'dhs_images.h5')

    # Normalize the IWI values
    mean_iwi = df['iwi'].mean()
    std_iwi = df['iwi'].std()
    df['iwi'] = (df['iwi'] - mean_iwi) / std_iwi

    # Create equal-sized folds
    folds = ['A', 'B', 'C', 'D', 'E']
    df['cv_fold'] = np.nan  # Initialize the cv_fold column

    # Generate and shuffle indices
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    # Split indices into equal-sized groups and assign folds
    fold_indices = np.array_split(indices, len(folds))
    for fold, idx in zip(folds, fold_indices):
        df.loc[idx, 'cv_fold'] = fold

    metrics = {}

    for fold in folds:
        print(f"Making predictions for fold {fold}...")

        # Define train, validation, and test folds
        test_fold = fold
        val_fold = folds[(folds.index(fold) + 1) % len(folds)]
        train_folds = [f for f in folds if f not in [test_fold, val_fold]]

        # Get dataloaders
        _, _, test_dataloader = get_dataloaders(df, hdf5_path, train_folds, val_fold, test_fold, batch_size=args.batch_size)

        # Load model
        model = init_model()
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'model_fold_{fold}_{args.save_epoch}.pth')))

        # Get test predictions
        model.eval()

        test_predictions = []
        test_targets = []
        with torch.no_grad():
            for inputs, targets in tqdm(test_dataloader):
                inputs = inputs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                outputs = model(inputs).squeeze()
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())
        # Scale back to original range
        test_predictions = np.array(test_predictions) * std_iwi + mean_iwi
        test_targets = np.array(test_targets) * std_iwi + mean_iwi

        # Store predictions in the dataframe
        fold_ixs = df['cv_fold'] == test_fold
        df.loc[fold_ixs, 'iwi_hat'] = test_predictions

        # Get the regression coefficient
        lcc_regressor = LinearRegression()
        lcc_regressor.fit(test_targets.reshape(-1, 1), test_predictions)

        metrics[fold] = {
            'test_r2': r2_score(test_targets, test_predictions),
            'test_mae': mean_absolute_error(test_targets, test_predictions),
            'test_reg_coef': lcc_regressor.coef_[0]
        }

    correction_values = {}

    for fold in folds:
        print(f"Getting correction values for fold {fold}...")

        # Define train, validation, and test folds
        test_fold = fold
        val_fold = folds[(folds.index(fold) + 1) % len(folds)]
        train_folds = [f for f in folds if f not in [test_fold, val_fold]]
        print(f"Train folds: {train_folds}, Validation/Calibration fold: {val_fold}")

        # Get dataloaders
        train_dataloader, val_dataloader, _ = get_dataloaders(df, hdf5_path, train_folds, val_fold, test_fold, batch_size=args.batch_size)

        # Load model
        model = init_model()
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'model_fold_{fold}_{args.save_epoch}.pth')))

        # Get predictions for train and validation sets
        model.eval()

        train_predictions = []
        train_targets = []
        with torch.no_grad():
            for inputs, targets in tqdm(train_dataloader, desc="Training folds predictions"):
                inputs = inputs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                outputs = model(inputs).squeeze()
                train_predictions.extend(outputs.cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
        # Scale back to original range
        train_predictions = np.array(train_predictions) * std_iwi + mean_iwi
        train_targets = np.array(train_targets) * std_iwi + mean_iwi

        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, targets in tqdm(val_dataloader, desc="Validation fold predictions"):
                inputs = inputs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                outputs = model(inputs).squeeze()
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        # Scale back to original range
        val_predictions = np.array(val_predictions) * std_iwi + mean_iwi
        val_targets = np.array(val_targets) * std_iwi + mean_iwi

        # Get value for Linear Correlation Correction (LCC)
        train_lcc_regressor = LinearRegression()
        train_lcc_regressor.fit(train_targets.reshape(-1, 1), train_predictions)

        val_lcc_regressor = LinearRegression()
        val_lcc_regressor.fit(val_targets.reshape(-1, 1), val_predictions)
        
        # Get sigma for Tweedie's correction (using both train and validation predictions)
        train_res_std = (train_predictions - train_targets).std()
        val_res_std = (val_predictions - val_targets).std()
        
        # Store correction values
        correction_values[fold] = {
            'train_lcc_slope': train_lcc_regressor.coef_[0],
            'train_lcc_intercept': train_lcc_regressor.intercept_,
            'val_lcc_slope': val_lcc_regressor.coef_[0],
            'val_lcc_intercept': val_lcc_regressor.intercept_,
            'train_sigma': train_res_std,
            'val_sigma': val_res_std,
            'train_predictions': list(train_predictions.astype(float)),  # Used for score function KDE
            'val_predictions': list(val_predictions.astype(float))  # Used for score function KDE
        }

        # Add metrics for train/val sets
        metrics[fold]['train_r2'] = r2_score(train_targets, train_predictions)
        metrics[fold]['train_mae'] = mean_absolute_error(train_targets, train_predictions)
        metrics[fold]['val_r2'] = r2_score(val_targets, val_predictions)
        metrics[fold]['val_mae'] = mean_absolute_error(val_targets, val_predictions)
        metrics[fold]['train_reg_coef'] = train_lcc_regressor.coef_[0]
        metrics[fold]['val_reg_coef'] = val_lcc_regressor.coef_[0]

    # Store the correction values as a JSON file
    with open(os.path.join(SAVE_DIR, 'correction_values.json'), 'w') as f:
        json.dump(correction_values, f)

    # Store the metrics as a JSON file
    with open(os.path.join(SAVE_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    # ---------- Generate KDE plots for train and validation predictions ----------
    val_kdes = {fold: gaussian_kde(correction_values[fold]['val_predictions']) for fold in folds}
    train_kdes = {fold: gaussian_kde(correction_values[fold]['train_predictions']) for fold in folds}

    df['train_score'] = df.apply(lambda row: score_function(row['iwi_hat'], train_kdes[row['cv_fold']]), axis=1)
    df['val_score'] = df.apply(lambda row: score_function(row['iwi_hat'], val_kdes[row['cv_fold']]), axis=1)

    # Restore original IWI values
    df['iwi'] = df['iwi'] * std_iwi + mean_iwi

    df.to_csv(os.path.join(SAVE_DIR, 'dhs_with_imgs_predictions.csv'), index=False)