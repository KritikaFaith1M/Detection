import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.autoencoder import Autoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Using Device: {DEVICE}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

dataset = datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = Autoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10

# ✅ FIX
os.makedirs("models/saved", exist_ok=True)

for epoch in range(EPOCHS):
    start_time = time.time()
    total_loss = 0

    print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(DEVICE)

        output = model(images)
        loss = criterion(output, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    print(f"Loss: {total_loss:.4f}")
    print(f"⏱ Time: {(time.time() - start_time)/60:.2f} min")

torch.save(model.state_dict(), "models/saved/autoencoder.pth")

print("✅ Autoencoder trained & saved")