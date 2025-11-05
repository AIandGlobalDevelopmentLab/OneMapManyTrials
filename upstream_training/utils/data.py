import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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

class NoisyRegressionDataset(RegressionDataset):
    def __init__(self, df, hdf5_path, noisy_frac=0.1, transform=None, random_state=42):
        super().__init__(df, hdf5_path, transform)
        self.noisy_frac = noisy_frac
        self.random_state = np.random.RandomState(random_state)
        
        # Copy dataframe to avoid modifying original
        self.df_noisy = self.df.copy()

        # Shuffle a fraction of labels
        self._add_label_noise()

    def _add_label_noise(self):
        n_samples = len(self.df_noisy)
        n_noisy = int(self.noisy_frac * n_samples)
        if n_noisy == 0:
            return
        
        # Select indices to corrupt
        noisy_indices = self.random_state.choice(n_samples, n_noisy, replace=False)
        
        # Shuffle their labels
        shuffled_labels = self.df_noisy.loc[noisy_indices, 'iwi'].sample(
            frac=1, random_state=self.random_state
        ).values
        
        # Apply shuffled labels back to the dataset
        self.df_noisy.loc[noisy_indices, 'iwi'] = shuffled_labels

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        cluster = self.df_noisy.iloc[idx]
        img = self.h5_file[cluster['cluster_id']][:].astype(np.float32)
        target = cluster['iwi']
        if self.transform:
            img = self.transform(img)
        return img, target

def get_dataloaders(df, hdf5_path, train_folds, val_fold, test_fold, noisy_frac=0, batch_size=128, num_workers=16):

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

    if noisy_frac == 0:
        train_dataset = RegressionDataset(df=df.iloc[train_folds], hdf5_path=hdf5_path, transform=train_transform)
        val_dataset = RegressionDataset(df=df.iloc[val_fold], hdf5_path=hdf5_path, transform=landsat_transform)
    else:
        train_dataset = NoisyRegressionDataset(df=df.iloc[train_folds], hdf5_path=hdf5_path, transform=train_transform, noisy_frac=noisy_frac)
        val_dataset = NoisyRegressionDataset(df=df.iloc[val_fold], hdf5_path=hdf5_path, transform=landsat_transform, noisy_frac=noisy_frac)
    test_dataset = RegressionDataset(df=df.iloc[test_fold], hdf5_path=hdf5_path, transform=landsat_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader
