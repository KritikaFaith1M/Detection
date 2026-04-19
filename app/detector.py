import torch
import torch.nn.functional as F
from models.autoencoder import Autoencoder


class AdversarialDetector:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # LOAD AUTOENCODER
        self.autoencoder = Autoencoder().to(device)
        self.autoencoder.load_state_dict(
            torch.load("models/saved/autoencoder.pth", map_location=device)
        )
        self.autoencoder.eval()

    # ENTROPY FUNCTION
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
        # NOISE TEST (REDUCED NOISE ✅)
        # -------------------------------
        noise = torch.randn_like(image) * 0.03
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
        # AUTOENCODER TEST
        # -------------------------------
        with torch.no_grad():
            recon = self.autoencoder(image)

        recon_error = torch.mean((image - recon) ** 2).item()

        # -------------------------------
        # IMPROVED DETECTION LOGIC ✅
        # -------------------------------
        score = 0

        if pred1.item() != pred2.item():
            score += 1

        if drop > 0.15:
            score += 1

        if entropy_increase > 0.05:
            score += 1

        if recon_error > 0.03:
            score += 1

        is_adv = score >= 2

        # OPTIONAL STABILITY CHECK
        if conf1.item() < 0.5:
            is_adv = False

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