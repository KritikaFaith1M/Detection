import torch
import torch.nn.functional as F
from models.autoencoder import Autoencoder

# Make deterministic
torch.manual_seed(42)


class AdversarialDetector:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Autoencoder (optional, only for logging)
        self.autoencoder = Autoencoder().to(device)
        self.autoencoder.load_state_dict(
            torch.load("models/saved/autoencoder.pth", map_location=device)
        )
        self.autoencoder.eval()

    def entropy(self, prob):
        return -torch.sum(prob * torch.log(prob + 1e-8), dim=1)

    def detect(self, image):
        image = image.to(self.device)

        # -------------------------------
        # ORIGINAL PREDICTION
        # -------------------------------
        with torch.no_grad():
            out1 = self.model(image)
            prob1 = F.softmax(out1, dim=1)
            conf1, pred1 = torch.max(prob1, 1)
            entropy1 = self.entropy(prob1)

        # -------------------------------
        # FIXED NOISE TEST (DETERMINISTIC)
        # -------------------------------
        noise = torch.ones_like(image) * 0.02
        noisy = torch.clamp(image + noise, -1, 1)

        with torch.no_grad():
            out2 = self.model(noisy)
            prob2 = F.softmax(out2, dim=1)
            conf2, pred2 = torch.max(prob2, 1)
            entropy2 = self.entropy(prob2)

        # -------------------------------
        # METRICS
        # -------------------------------
        drop = abs(conf1.item() - conf2.item())
        entropy_increase = (entropy2 - entropy1).item()

        # -------------------------------
        # AUTOENCODER (FOR LOG ONLY)
        # -------------------------------
        with torch.no_grad():
            recon = self.autoencoder(image)

        recon_error = torch.mean((image - recon) ** 2).item()

        # -------------------------------
        # FINAL STABLE DETECTION LOGIC
        # -------------------------------
        is_adv = False

        # 1. Prediction change (strongest)
        if pred1.item() != pred2.item():
            is_adv = True

        # 2. Confidence drop
        elif drop > 0.02:
            is_adv = True

        # 3. Entropy increase
        elif entropy_increase > 0.02:
            is_adv = True

        # 4. Low confidence safeguard
        elif conf1.item() < 0.4:
            is_adv = True
            
        elif conf1.item() < 0.92 and drop > 0.002:
            is_adv = True
        # -------------------------------
        # DEBUG OUTPUT
        # -------------------------------
        print("\n========== DEBUG ==========")
        print("Pred:", pred1.item(), "->", pred2.item())
        print("Confidence:", conf1.item(), "->", conf2.item())
        print("Drop:", drop)
        print("Entropy Increase:", entropy_increase)
        print("Recon Error:", recon_error)
        print("Final Decision:", "ATTACK" if is_adv else "SAFE")
        print("===========================\n")

        # -------------------------------
        # RETURN
        # -------------------------------
        return {
            "is_adversarial": is_adv,
            "pred_before": pred1.item(),
            "pred_after": pred2.item(),
            "conf_before": conf1.item(),
            "conf_after": conf2.item(),
            "confidence_drop": drop,
            "entropy_before": entropy1.item(),
            "entropy_after": entropy2.item(),
            "entropy_increase": entropy_increase,
            "reconstruction_error": recon_error
        }