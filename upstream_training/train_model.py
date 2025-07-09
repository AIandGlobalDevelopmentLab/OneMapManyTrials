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

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def unfreeze_layers(model, layers: list):
    for name, module in model.named_modules():
        if any(layer in name for layer in layers):
            for param in module.parameters():
                param.requires_grad = True

class RatledgeLoss(nn.Module):
    def __init__(self, quants, lambda_b=5.0):
        """
        Args:
          quants: 1-D Tensor of shape (6,) giving the 0%,20%,…,100% IWI cut-points
                  computed once on the full training set.
          lambda_b: weight on the max-bias penalty
        """
        super().__init__()
        self.register_buffer('quants', quants)  
        self.lambda_b = lambda_b

    def forward(self, y_pred, y_true):
        # 1) Standard MSE on the whole batch
        mse = nn.functional.mse_loss(y_pred, y_true)

        # 2) Compute squared bias per quintile, take the max
        max_bias2 = torch.tensor(0., device=y_true.device)
        for j in range(5):
            lo, hi = self.quants[j], self.quants[j+1]
            if j < 4:
                mask = (y_true >= lo) & (y_true < hi)
            else:
                mask = (y_true >= lo) & (y_true <= hi)

            if mask.any():
                bias_j = (y_pred[mask] - y_true[mask]).mean()
                bias2_j = bias_j.pow(2)
                max_bias2 = torch.max(max_bias2, bias2_j)

        return mse + self.lambda_b * max_bias2


def train_model(model, train_loader, val_loader, num_epochs=20, patience=5, lr=1e-4, 
    loss='mse', lambda_r=1e-5, lambda_b=5, T_0=5, T_mult=2, quantile_values=None,
    device=device, freeze_backbone=False, freeze_until=None):
    
    # Optionally freeze parts of the model
    if freeze_backbone:
        freeze_all(model)
        unfreeze_layers(model, ['fc'])  # just head
    elif freeze_until:
        freeze_all(model)
        unfreeze_layers(model, freeze_until)
    else:
        unfreeze_all(model)

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=lambda_r
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

    # Loss function
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'ratledge':
        if quantile_values is None:
            raise ValueError("Quantile values must be provided for Ratledge loss.")
        quants_tensor = torch.tensor(quantile_values, dtype=torch.float32).to(device)
        criterion = RatledgeLoss(quants=quants_tensor, lambda_b=lambda_b)
    else:
        raise ValueError(f"Unsupported loss function: {loss}")
    
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_model_state = model.state_dict()
    steps_since_improvement = 0
    train_losses = []
    val_losses = []

    progress_bar = tqdm(range(num_epochs), desc="Training Progress")

    data_time = 0.0
    inference_time = 0.0
    backprop_time = 0.0
    val_data_time = 0.0
    val_inference_time = 0.0

    for epoch in progress_bar:
        model.train()
        running_loss = 0.0
        start_data_time = time()
        for inputs, targets in train_loader:
            inputs = inputs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            targets = targets.to(device, dtype=torch.float32)
            end_data_time = time()
            data_time += end_data_time - start_data_time

            optimizer.zero_grad()
            with autocast():
                start_inference_time = time()
                outputs = model(inputs).squeeze()  # remove extra dim if needed
                end_inference_time = time()
                inference_time += end_inference_time - start_inference_time
                loss = criterion(outputs, targets)

            # Backpropagation
            start_backprop_time = time()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            end_backprop_time = time()
            backprop_time += end_backprop_time - start_backprop_time
            start_data_time = time()  # reset for next batch

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            start_val_data_time = time()
            for inputs, targets in val_loader:
                inputs = inputs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
                targets = targets.to(device, dtype=torch.float32)
                end_val_data_time = time()
                val_data_time += end_val_data_time - start_val_data_time
                stat_val_inference_time = time()
                outputs = model(inputs).squeeze()
                end_val_inference_time = time()
                val_inference_time += end_val_inference_time - stat_val_inference_time
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                start_val_data_time = time()  # reset for next batch

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
        progress_bar.set_postfix({
            'Train Loss': epoch_loss,
            'Val Loss': val_loss,
            'Best Val Loss': best_val_loss
        })
        if steps_since_improvement >= patience:
            print(f"Early stopping triggered after {steps_since_improvement} epochs without improvement.")
            break
    
    model.load_state_dict(best_model_state)
    print('Times:')
    total_time = data_time + inference_time + backprop_time + val_data_time + val_inference_time
    print(f"Total (covered) epoch time: {total_time:.2f}s")
    print(f"Data loading time: {data_time:.2f}s, ({100*(data_time/total_time):.2f}%)")
    print(f"Inference time: {inference_time:.2f}s, ({100*(inference_time/total_time):.2f}%)")
    print(f"Backpropagation time: {backprop_time:.2f}s, ({100*(backprop_time/total_time):.2f}%)")
    print(f"Validation data loading time: {val_data_time:.2f}s, ({100*(val_data_time/total_time):.2f}%)")
    print(f"Validation inference time: {val_inference_time:.2f}s, ({100*(val_inference_time/total_time):.2f}%)")
    return model

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
    parser.add_argument('--seed', type=int, help='Seed for random initialization and shuffling', default=42)
    args = parser.parse_args()

    # Set seeds for reproducibility
    RANDOM_STATE = args.seed
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Create a save directory
    SAVE_DIR = os.path.join(DATA_DIR, 'models', get_save_dir_name(args))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Load the dataset
    df = pd.read_csv(os.path.join(DATA_DIR, 'dhs_with_imgs.csv'))
    # img_dir = os.path.join(DATA_DIR, 'dhs_images')
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

    for fold in folds:
        print(f"Training for fold {fold}...")

        # Define train, validation, and test folds
        test_fold = fold
        val_fold = folds[(folds.index(fold) + 1) % len(folds)]
        train_folds = [f for f in folds if f not in [test_fold, val_fold]]
        print(f"Train folds: {train_folds}, Validation fold: {val_fold}, Test fold: {test_fold}")

        # Get quantile values for train set (only used for Ratledge loss)
        quantile_values = np.quantile(df[df['cv_fold'].isin(train_folds)]['iwi'].values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Get dataloaders
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(df, hdf5_path, train_folds, val_fold, test_fold, batch_size=args.batch_size)

        # Initialize model
        model = init_model()

        print('==== Phase 1 — linear probe ====')
        if args.num_linear_epochs == 0:
            print('Skipping linear probe phase as num_linear_epochs is set to 0.')
        else:
            model = train_model(model, train_dataloader, val_dataloader, num_epochs=args.num_linear_epochs, 
            loss=args.loss, lambda_r=args.lambda_r, lambda_b=args.lambda_b, quantile_values=quantile_values,
            freeze_backbone=True, device=device)

        print('==== Phase 2 — unfreeze top ResNet block ====')
        if args.num_top_epochs == 0:
            print('Skipping top block training phase as num_top_epochs is set to 0.')
        else:
            model = train_model(model, train_dataloader, val_dataloader, num_epochs=args.num_top_epochs, 
            loss=args.loss, lambda_r=args.lambda_r, lambda_b=args.lambda_b, quantile_values=quantile_values,
            freeze_until=['layer4', 'fc'], device=device)

        print('==== Phase 3 — unfreeze all layers ====')
        if args.num_full_epochs == 0:
            print('Skipping full training phase as num_full_epochs is set to 0.')
        else:
            model = train_model(model, train_dataloader, val_dataloader, num_epochs=args.num_full_epochs, 
                loss=args.loss, lambda_r=args.lambda_r, lambda_b=args.lambda_b, quantile_values=quantile_values,
                device=device)

        # Save the model
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'model_fold_{fold}.pth'))

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
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'model_fold_{fold}.pth')))

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
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'model_fold_{fold}.pth')))

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