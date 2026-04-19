import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import random
import numpy as np

# -------------------------------
# SEED
# -------------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# -------------------------------
# MAIN FUNCTION (WINDOWS FIX ✅)
# -------------------------------
def main():

    set_seed()

    # -------------------------------
    # CONFIG
    # -------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    EPOCHS = 8
    LR = 0.001

    print(f"🚀 Using Device: {DEVICE}")
    print("📂 Working Directory:", os.getcwd())

    # -------------------------------
    # TRANSFORMS
    # -------------------------------
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

    # -------------------------------
    # DATA
    # -------------------------------
    train_dataset = datasets.CIFAR10(
        root="./dataset", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=test_transform
    )

    # 🔥 WINDOWS SAFE (num_workers=0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # -------------------------------
    # MODEL
    # -------------------------------
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)

    # -------------------------------
    # LOSS & OPTIMIZER
    # -------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -------------------------------
    # FGSM
    # -------------------------------
    def fgsm_attack(model, images, labels, eps=0.2):
        images = images.clone().detach().requires_grad_(True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv = images + eps * images.grad.sign()
        return torch.clamp(adv, -1, 1).detach()

    # -------------------------------
    # PGD (FAST VERSION ⚡)
    # -------------------------------
    def pgd_attack(model, images, labels, eps=0.3, alpha=0.01, iters=2):
        ori = images.clone().detach()
        adv = ori.clone()

        for _ in range(iters):
            adv.requires_grad_(True)

            outputs = model(adv)
            loss = criterion(outputs, labels)

            model.zero_grad()
            loss.backward()

            adv = adv + alpha * adv.grad.sign()
            eta = torch.clamp(adv - ori, -eps, eps)
            adv = torch.clamp(ori + eta, -1, 1).detach()

        return adv

    # -------------------------------
    # SAVE PATH
    # -------------------------------
    save_dir = os.path.join(os.getcwd(), "models", "saved")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cifar_model.pth")

    # -------------------------------
    # TRAIN LOOP
    # -------------------------------
    try:
        for epoch in range(EPOCHS):
            model.train()

            total_loss = 0
            correct_clean = 0
            correct_adv = 0
            total = 0

            print(f"\n🔥 EPOCH {epoch+1}/{EPOCHS}")

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()

                # CLEAN
                out_clean = model(images)
                loss_clean = criterion(out_clean, labels)

                _, pred_clean = torch.max(out_clean, 1)
                correct_clean += (pred_clean == labels).sum().item()

                # FGSM
                adv1 = fgsm_attack(model, images, labels)
                out_adv1 = model(adv1)
                loss_adv1 = criterion(out_adv1, labels)

                _, pred_adv = torch.max(out_adv1, 1)
                correct_adv += (pred_adv == labels).sum().item()

                # PGD
                adv2 = pgd_attack(model, images, labels)
                out_adv2 = model(adv2)
                loss_adv2 = criterion(out_adv2, labels)

                # NOISE
                noise = torch.randn_like(images) * 0.1
                adv3 = torch.clamp(images + noise, -1, 1)
                out_adv3 = model(adv3)
                loss_adv3 = criterion(out_adv3, labels)

                # TOTAL LOSS
                loss = loss_clean + loss_adv1 + loss_adv2 + loss_adv3

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total += labels.size(0)

                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

            # -------------------------------
            # ACCURACY
            # -------------------------------
            acc_clean = 100 * correct_clean / total
            acc_adv = 100 * correct_adv / total

            print(f"\n📊 Epoch {epoch+1}")
            print(f"Loss: {total_loss:.4f}")
            print(f"✅ Clean Acc: {acc_clean:.2f}%")
            print(f"⚠️ Adv Acc: {acc_adv:.2f}%")

            # SAVE MODEL
            print("💾 Saving model...")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved at: {save_path}")

    except Exception as e:
        print("❌ Training crashed:", e)
        torch.save(model.state_dict(), "emergency_model.pth")
        print("🚑 Emergency model saved!")

    # -------------------------------
    # TEST
    # -------------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = 100 * correct / total

    print("\n🎯 FINAL RESULT")
    print(f"🏆 Test Accuracy: {test_acc:.2f}%")

    print("\n✅ Training Done")


# -------------------------------
# ENTRY POINT (WINDOWS FIX ✅)
# -------------------------------
if __name__ == "__main__":
    main()