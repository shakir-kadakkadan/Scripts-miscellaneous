#!/usr/bin/env python3
"""
Hyperliquid order placement script callable from Kotlin
Usage: python hyperliquid_order.py <coin> <side> <size> <price>
Example: python hyperliquid_order.py ETH buy 0.05 2000
"""

import sys
import os
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account

# Credentials from environment or hardcoded
MAIN_WALLET = os.getenv("HL_MAIN_WALLET", "0x9CD9C25EcFf658FCd1dd289cBf44A277874f4C40")
API_WALLET_KEY = os.getenv("HL_API_KEY", "0x78c2830a077ec424374be73783c767816ceb54ed3765db225ea2b8a80e4c6d5b")

def place_order(coin, side, size, price):
    """Place a limit order on Hyperliquid"""
    try:
        # Initialize with API wallet
        account = eth_account.Account.from_key(API_WALLET_KEY)
        
        # Create exchange instance with main wallet address
        exchange = Exchange(
            account,
            constants.MAINNET_API_URL,
            account_address=MAIN_WALLET
        )
        
        # Set leverage to 1x (isolated)
        print(f"Setting leverage to 1x for {coin}...")
        exchange.update_leverage(1, coin, True)
        
        # Place order
        is_buy = side.lower() == "buy"
        print(f"Placing {side.upper()} order: {size} {coin} @ ${price}")
        
        order_result = exchange.order(
            coin,
            is_buy,
            float(size),
            float(price),
            {"limit": {"tif": "Gtc"}}
        )
        
        print(f"✅ Order placed successfully!")
        print(f"Response: {order_result}")
        
        # Extract order ID if available
        if order_result and "status" in order_result:
            if order_result["status"] == "ok":
                data = order_result.get("response", {}).get("data", {})
                statuses = data.get("statuses", [])
                if statuses and len(statuses) > 0:
                    first_status = statuses[0]
                    if "resting" in first_status:
                        oid = first_status["resting"]["oid"]
                        print(f"Order ID: {oid}")
                        return 0
            else:
                print(f"❌ Order failed: {order_result}")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

def get_user_state():
    """Get user account state"""
    try:
        info = Info(constants.MAINNET_API_URL)
        state = info.user_state(MAIN_WALLET)
        
        margin = state.get("marginSummary", {})
        account_value = margin.get("accountValue", "0")
        
        print(f"Account Value: ${account_value}")
        print(f"Full state: {state}")
        return 0
        
    except Exception as e:
        print(f"❌ Error getting user state: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python hyperliquid_order.py status")
        print("  python hyperliquid_order.py order <coin> <side> <size> <price>")
        print("Example:")
        print("  python hyperliquid_order.py order ETH buy 0.05 2000")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "status":
        sys.exit(get_user_state())
    elif command == "order":
        if len(sys.argv) != 6:
            print("Error: order command requires: <coin> <side> <size> <price>")
            sys.exit(1)
        coin = sys.argv[2]
        side = sys.argv[3]
        size = sys.argv[4]
        price = sys.argv[5]
        sys.exit(place_order(coin, side, size, price))
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
