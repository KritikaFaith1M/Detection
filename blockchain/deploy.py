import json
from web3 import Web3
from solcx import compile_standard, install_solc

# CONNECT GANACHE
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))

assert w3.is_connected(), "❌ Ganache not running"
print("✅ Connected to Ganache")

# INSTALL SOLC
install_solc("0.8.0")

# READ CONTRACT
with open("blockchain/AuditNotary.sol", "r") as f:
    source = f.read()

# COMPILE
compiled = compile_standard({
    "language": "Solidity",
    "sources": {"AuditNotary.sol": {"content": source}},
    "settings": {
        "outputSelection": {
            "*": {"*": ["abi", "evm.bytecode"]}
        }
    }
}, solc_version="0.8.0")

abi = compiled["contracts"]["AuditNotary.sol"]["AuditNotary"]["abi"]
bytecode = compiled["contracts"]["AuditNotary.sol"]["AuditNotary"]["evm"]["bytecode"]["object"]

# ACCOUNT
account = w3.eth.accounts[0]

# DEPLOY
Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash = Contract.constructor().transact({"from": account})
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print("✅ Contract deployed at:", receipt.contractAddress)

# SAVE
with open("blockchain/contract_info.json", "w") as f:
    json.dump({
        "address": receipt.contractAddress,
        "abi": abi
    }, f)

print("💾 contract_info.json saved")