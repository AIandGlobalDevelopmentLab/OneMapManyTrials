import os
import pandas as pd
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

class RegressionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cluster_id = self.df.iloc[idx]['cluster_id']
        img_name = os.path.join(self.img_dir, cluster_id + '.png')
        image = Image.open(img_name)
        target = (self.df.iloc[idx]['iwi'] / 100)
        if self.transform:
            image = self.transform(image)
        return image, target
    
class CLIPRegression(nn.Module):
    def __init__(self, model_name="flax-community/clip-rsicd"):
        super(CLIPRegression, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.regression_head = nn.Linear(self.clip_model.config.projection_dim, 1)

    def forward(self, images):
        outputs = self.clip_model.get_image_features(images)
        regression_output = self.regression_head(outputs)
        return regression_output

if __name__ == '__main__':
    
    # Read config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    DATA_DIR = config['PATHS']['DATA_DIR']

    df = pd.read_csv(os.path.join(DATA_DIR, 'dhs_data.csv'))

    # ========= Set up dataloader for dataset =========
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])

    csv_file = os.path.join(DATA_DIR, 'dhs_data.csv')
    img_dir = os.path.join(DATA_DIR, 'dhs_images')

    dataset = RegressionDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # ========= Load pretrained CLIP model and add regression head =========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPRegression().to(device)
    processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd")

    # ========= Train model =========
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(5):  # number of epochs
        running_loss = 0.0
        for images, targets in tqdm(dataloader):
            inputs = processor(images=images, return_tensors="pt", padding=True, do_rescale=False, do_normalize=False).to(device)
            pixel_values = inputs['pixel_values']
            
            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = criterion(outputs.squeeze(), targets.to(device).float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

    # ========= Save model =========
    torch.save(model.state_dict(), 'clip_regression.pth')