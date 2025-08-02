import h5py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import configparser

# Read config file
config = configparser.ConfigParser()
config.read('../config.ini')

DATA_DIR = config['PATHS']['DATA_DIR']

df = pd.read_csv(os.path.join(DATA_DIR, 'dhs_with_imgs.csv'))
img_dir = os.path.join(DATA_DIR, 'dhs_images')

hdf5_path = os.path.join(DATA_DIR, 'dhs_images.h5')

with h5py.File(hdf5_path, 'w') as h5f:
    for i, row in tqdm(df.iterrows(), total=len(df)):
        cluster_id = row['cluster_id']
        img_path = os.path.join(img_dir, cluster_id, 'landsat.np')
        try:
            img = np.load(img_path)  # shape: (H, W, 6)
            img = img.astype(np.uint16)  # Convert to uint16
            h5f.create_dataset(cluster_id, data=img, compression="gzip")
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

print(f"All images written to {hdf5_path}")

print('Sanity check:')
with h5py.File(hdf5_path, 'r') as h5f:
    print(f"Number of datasets in HDF5 file: {len(h5f)}")
    print(f"Keys in HDF5 file: {list(h5f.keys())[:5]}")  # Show first 5 keys
    print(f"Shape of first dataset: {h5f[h5f.keys()[0]].shape}")  # Shape of the first dataset
    print(f"Data type of first dataset: {h5f[h5f.keys()[0]].dtype}")  # Data type of the first dataset