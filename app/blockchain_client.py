import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blockchain.client import BlockchainClient

class AppBlockchain:
    def __init__(self):
        self.client = BlockchainClient()

    def store(self, cid, risk_level):
        # Only store high-risk attacks
        if risk_level == "HIGH" and cid:
            tx_hash = self.client.store_hash(cid)
            return tx_hash
        return None