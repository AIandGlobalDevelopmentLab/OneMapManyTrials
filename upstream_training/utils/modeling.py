import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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