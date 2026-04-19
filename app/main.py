import sys
import os
from datetime import datetime

# -------------------------------
# ADD PROJECT ROOT TO PATH
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import transforms
from PIL import Image

from models.classifier import load_model
from app.detector import AdversarialDetector
from app.decision_engine import DecisionEngine
from utils.hashing import generate_hash
from utils.ipfs import IPFSClient
from utils.logger import AuditLogger
from app.blockchain_client import AppBlockchain


class AdversarialSystem:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🚀 Using Device: {self.device}")

        # LOAD MODEL
        self.model = load_model(model_path, self.device)

        # DETECTOR
        self.detector = AdversarialDetector(self.model, self.device)

        self.decision_engine = DecisionEngine()
        self.ipfs = IPFSClient()
        self.logger = AuditLogger()
        self.blockchain = AppBlockchain()

        # TRANSFORM (IMPROVED)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ])

    def process_image(self, image_path):
        print(f"\n🖼 Processing Image: {image_path}")

        if not os.path.exists(image_path):
            print("❌ Image not found!")
            return None

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # -------------------------------
        # DETECTION
        # -------------------------------
        detection = self.detector.detect(image_tensor)

        print(
            f"DEBUG → pred1: {detection['pred_before']}, "
            f"pred2: {detection['pred_after']}, "
            f"drop: {detection['confidence_drop']:.4f}, "
            f"entropy Δ: {detection['entropy_increase']:.4f}"
        )

        # -------------------------------
        # AUTOENCODER CLEAN + RECHECK ✅
        # -------------------------------
        cleaned_pred = None
        cleaned_conf = None

        if detection["is_adversarial"]:
            print("🧼 Cleaning image using Autoencoder...")

            with torch.no_grad():
                cleaned = self.detector.autoencoder(image_tensor)

                out_cleaned = self.model(cleaned)
                prob_cleaned = torch.softmax(out_cleaned, dim=1)
                cleaned_conf, cleaned_pred = torch.max(prob_cleaned, 1)

            print(
                f"🧽 Cleaned Prediction: {cleaned_pred.item()}, "
                f"Confidence: {cleaned_conf.item():.4f}"
            )

        # -------------------------------
        # DECISION
        # -------------------------------
        decision = self.decision_engine.evaluate(detection)

        print(f"🔍 Status: {decision['status']}")
        print(f"⚠️ Risk Level: {decision['risk_level']}")
        print(f"⚙️ Action: {decision['action']}")

        cid = None
        ipfs_url = None
        tx_hash = None
        file_hash = None

        # -------------------------------
        # SECURITY ACTIONS
        # -------------------------------
        if decision["status"] == "ALERT":

            file_hash = generate_hash(image_path)
            print(f"🔐 File Hash: {file_hash}")

            try:
                cid, ipfs_url = self.ipfs.upload(image_path)

                if cid:
                    print(f"🌐 IPFS CID: {cid}")
                    print(f"🔗 URL: {ipfs_url}")
                else:
                    print("❌ IPFS Upload Failed")

            except Exception as e:
                print(f"❌ IPFS Error: {e}")
                cid = None

            if cid and decision["risk_level"] == "HIGH":
                try:
                    tx_hash = self.blockchain.store(cid, decision["risk_level"])

                    if tx_hash:
                        print(f"⛓ Blockchain TX: {tx_hash}")
                    else:
                        print("❌ Blockchain failed")

                except Exception as e:
                    print(f"❌ Blockchain Error: {e}")

        # -------------------------------
        # LOG DATA (IMPROVED)
        # -------------------------------
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "status": decision["status"],
            "risk_level": decision["risk_level"],
            "action": decision["action"],

            "pred_before": detection["pred_before"],
            "pred_after": detection["pred_after"],

            "conf_before": detection["conf_before"],
            "conf_after": detection["conf_after"],
            "confidence_drop": detection["confidence_drop"],

            "entropy_before": detection["entropy_before"],
            "entropy_after": detection["entropy_after"],
            "entropy_increase": detection["entropy_increase"],

            "reconstruction_error": detection["reconstruction_error"],

            "cleaned_prediction": cleaned_pred.item() if cleaned_pred is not None else None,
            "cleaned_confidence": cleaned_conf.item() if cleaned_conf is not None else None,

            "file_hash": file_hash,
            "cid": cid,
            "ipfs_url": ipfs_url,
            "tx_hash": tx_hash
        }

        log_path = self.logger.save_log(log_data)

        print(f"📄 Log saved at: {log_path}")

        return log_data


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    system = AdversarialSystem("models/saved/cifar_model.pth")

    test_image = "adv_test.png"

    result = system.process_image(test_image)

    print("\n📊 FINAL RESULT:")
    print(result)