import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# -------------------------------
# CONFIG
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/saved/cifar_model.pth"

# -------------------------------
# TRANSFORM (IMPORTANT SAME AS TRAIN)
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# -------------------------------
# LOAD DATA
# -------------------------------
dataset = datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform)

# -------------------------------
# LOAD MODEL
# -------------------------------
from torchvision import models

model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# -------------------------------
# ATTACKS
# -------------------------------
def fgsm_attack(model, image, label, eps=0.2):
    image.requires_grad = True

    output = model(image)
    loss = nn.functional.cross_entropy(output, label)

    model.zero_grad()
    loss.backward()

    adv = image + eps * image.grad.sign()
    return torch.clamp(adv, -1, 1).detach()

def pgd_attack(model, image, label, eps=0.4, alpha=0.02, iters=20):
    ori = image.clone()

    for _ in range(iters):
        image.requires_grad = True

        output = model(image)
        loss = nn.functional.cross_entropy(output, label)

        model.zero_grad()
        loss.backward()

        adv = image + alpha * image.grad.sign()
        eta = torch.clamp(adv - ori, -eps, eps)
        image = torch.clamp(ori + eta, -1, 1).detach()

    return image

# -------------------------------
# SAVE FOLDER
# -------------------------------
os.makedirs("test_images", exist_ok=True)

# -------------------------------
# GENERATE IMAGES
# -------------------------------
count = 0

for i in range(len(dataset)):
    image, label = dataset[i]

    image = image.unsqueeze(0).to(DEVICE)
    label = torch.tensor([label]).to(DEVICE)

    # SAVE SAFE IMAGE
    save_image((image * 0.5 + 0.5), f"test_images/safe_{count}.png")

    # FGSM
    adv_fgsm = fgsm_attack(model, image, label)
    save_image((adv_fgsm * 0.5 + 0.5), f"test_images/fgsm_{count}.png")

    # PGD
    adv_pgd = pgd_attack(model, image, label)
    save_image((adv_pgd * 0.5 + 0.5), f"test_images/pgd_{count}.png")

    count += 1

    if count == 2:   # 🔥 ONLY 2 IMAGES
        break

print("\n✅ Images Generated in /test_images folder")