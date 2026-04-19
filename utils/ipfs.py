import ipfshttpclient

class IPFSClient:
    def __init__(self):
        try:
            self.client = ipfshttpclient.connect("/ip4/127.0.0.1/tcp/5001")
            print("✅ Connected to IPFS")
        except Exception as e:
            print("❌ IPFS connection failed:", e)
            self.client = None

    def upload(self, file_path):
        if self.client is None:
            return None, None

        try:
            result = self.client.add(file_path)
            cid = result["Hash"]
            url = f"https://ipfs.io/ipfs/{cid}"

            return cid, url

        except Exception as e:
            print("❌ Upload failed:", e)
            return None, None