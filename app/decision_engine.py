class DecisionEngine:
    def evaluate(self, d):

        if d["is_adversarial"]:

            # 🔥 HIGH RISK (PGD / strong attacks)
            if (
                d["confidence_drop"] > 0.15 or
                d["entropy_increase"] > 0.05 or
                d["reconstruction_error"] > 0.03
            ):
                return {
                    "status": "ALERT",
                    "risk_level": "HIGH",
                    "action": "STORE_BLOCKCHAIN"
                }

            # ⚠️ MEDIUM RISK
            elif d["confidence_drop"] > 0.05:
                return {
                    "status": "ALERT",
                    "risk_level": "MEDIUM",
                    "action": "STORE_IPFS"
                }

            # 🟡 LOW RISK
            else:
                return {
                    "status": "ALERT",
                    "risk_level": "LOW",
                    "action": "LOG_ONLY"
                }

        else:
            return {
                "status": "SAFE",
                "risk_level": "SAFE",
                "action": "ALLOW"
            }