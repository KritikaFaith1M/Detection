import torch
import torch.nn as nn
from torchvision import models


class CIFARResNet(nn.Module):
    def __init__(self, model_type="resnet18", num_classes=10):
        super(CIFARResNet, self).__init__()

        # -------------------------------
        # SELECT MODEL (UPDATED ✅)
        # -------------------------------
        if model_type == "resnet18":
            self.model = models.resnet18(weights="IMAGENET1K_V1")
        elif model_type == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V1")
        else:
            raise ValueError("Unsupported model type")

        # -------------------------------
        # CIFAR MODIFICATION (SAME)
        # -------------------------------
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

        # -------------------------------
        # FINAL LAYER (SAME)
        # -------------------------------
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# -------------------------------
# LOAD MODEL FUNCTION (UPDATED)
# -------------------------------
def load_model(model_path, device="cpu"):
    import torch
    import torch.nn as nn
    from torchvision import models

    # SAME MODEL AS TRAINING
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    # LOAD
    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("✅ Model loaded successfully (fixed)")

    return model