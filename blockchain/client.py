import json
from web3 import Web3

class BlockchainClient:
    def __init__(self, provider_url="http://127.0.0.1:7545"):
        # CONNECT
        self.w3 = Web3(Web3.HTTPProvider(provider_url))

        if not self.w3.is_connected():
            raise Exception("❌ Blockchain not connected")

        print("✅ Connected to Blockchain")

        # LOAD CONTRACT
        with open("blockchain/contract_info.json", "r") as f:
            contract_info = json.load(f)

        self.contract = self.w3.eth.contract(
            address=contract_info["address"],
            abi=contract_info["abi"]
        )

        # DEFAULT ACCOUNT
        self.account = self.w3.eth.accounts[0]

    def store_hash(self, cid):
        try:
            tx_hash = self.contract.functions.storeHash(cid).transact({
                "from": self.account
            })

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            print(f"✅ Stored on Blockchain: {cid}")
            print(f"🧾 TX Hash: {receipt.transactionHash.hex()}")

            return receipt.transactionHash.hex()

        except Exception as e:
            print("❌ Blockchain Error:", e)
            return None