import json
import os
import time

class AuditLogger:
    def __init__(self, log_dir="audit_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def save_log(self, data, index=None):
        # -------------------------------
        # FILE NAME
        # -------------------------------
        if index is None:
            index = int(time.time())

        file_path = os.path.join(self.log_dir, f"log_{index}.json")

        # -------------------------------
        # ADD TIMESTAMP
        # -------------------------------
        data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # -------------------------------
        # SAVE JSON
        # -------------------------------
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"📄 Log saved: {file_path}")

        return file_path