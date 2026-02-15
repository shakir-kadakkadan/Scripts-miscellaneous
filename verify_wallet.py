#!/usr/bin/env python3
"""Verify wallet address derived from private key"""

from eth_account import Account

# Private key from the authorization screenshot
PRIVATE_KEY = "0x78c2830a077ec424374be73783c767816ceb54ed3765db225ea2b8a80e4c6d5b"

# Create wallet from private key
wallet = Account.from_key(PRIVATE_KEY)

print("=" * 60)
print("Wallet Address Verification")
print("=" * 60)
print(f"Private Key: {PRIVATE_KEY}")
print(f"Derived Address: {wallet.address}")
print(f"Expected Address: 0x055C8CcaD478A3904520ECfE375D104770C4eba7")
print(f"Match: {wallet.address.lower() == '0x055C8CcaD478A3904520ECfE375D104770C4eba7'.lower()}")
print("=" * 60)
