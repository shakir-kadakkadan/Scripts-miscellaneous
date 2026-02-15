#!/usr/bin/env python3
"""
Hyperliquid Test Order Script
Place a limit buy order for ETH at $2000 with 1x leverage
Balance: ~$100 USDC
"""

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
import json

# API Wallet credentials from the authorization
WALLET_ADDRESS = "0x055C8CcaD478A3904520ECfE375D104770C4eba7"
PRIVATE_KEY = "0x78c2830a077ec424374be73783c767816ceb54ed3765db225ea2b8a80e4c6d5b"

def main():
    print("=" * 60)
    print("Hyperliquid Order Placement Test")
    print("=" * 60)
    
    # Initialize the exchange with mainnet API using the private key
    # The SDK will derive the wallet from the private key
    from eth_account import Account
    
    # Create wallet from private key
    wallet = Account.from_key(PRIVATE_KEY)
    
    # Initialize the exchange
    exchange = Exchange(
        wallet=wallet,
        base_url=constants.MAINNET_API_URL
    )
    
    # Initialize Info API to check account state
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    print("\n1. Checking account state...")
    try:
        user_state = info.user_state(WALLET_ADDRESS)
        print(f"   Account Address: {WALLET_ADDRESS}")
        
        # Display margin summary
        if 'marginSummary' in user_state:
            margin = user_state['marginSummary']
            print(f"   Account Value: ${float(margin.get('accountValue', 0)):.2f}")
            print(f"   Total Margin Used: ${float(margin.get('totalMarginUsed', 0)):.2f}")
            print(f"   Withdrawable: ${float(margin.get('withdrawable', 0)):.2f}")
        
        # Display positions
        if 'assetPositions' in user_state:
            positions = user_state['assetPositions']
            print(f"\n   Current Positions: {len(positions)}")
            for pos in positions:
                if 'position' in pos:
                    p = pos['position']
                    print(f"     - {p.get('coin', 'N/A')}: Size={p.get('szi', 0)}, Entry=${p.get('entryPx', 0)}")
    
    except Exception as e:
        print(f"   Error fetching account state: {e}")
        return
    
    print("\n2. Preparing order parameters...")
    
    # Order parameters
    asset = "ETH"  # Asset to trade
    is_buy = True  # Buy order
    limit_price = "2000"  # Limit price at $2000
    size = "0.05"  # Size in ETH (~$100 at $2000)
    reduce_only = False  # Not a reduce-only order
    
    print(f"   Asset: {asset}")
    print(f"   Side: {'BUY' if is_buy else 'SELL'}")
    print(f"   Price: ${limit_price}")
    print(f"   Size: {size} {asset}")
    print(f"   Notional: ~${float(limit_price) * float(size):.2f}")
    print(f"   Leverage: 1x (using full collateral)")
    
    print("\n3. Setting leverage to 1x...")
    try:
        # First, get the asset index for ETH
        meta = info.meta()
        eth_index = None
        for i, universe_item in enumerate(meta['universe']):
            if universe_item['name'] == asset:
                eth_index = i
                break
        
        if eth_index is None:
            print(f"   Error: Could not find asset index for {asset}")
            return
        
        print(f"   Asset index for {asset}: {eth_index}")
        
        # Update leverage to 1x (isolated mode)
        leverage_result = exchange.update_leverage(
            1,  # leverage
            asset,  # coin
            False  # is_cross (use isolated margin)
        )
        print(f"   Leverage update result: {leverage_result}")
        
    except Exception as e:
        print(f"   Error setting leverage: {e}")
        print("   Continuing with current leverage settings...")
    
    print("\n4. Placing limit buy order...")
    print(f"   Order: BUY {size} {asset} @ ${limit_price}")
    
    # Uncomment the following lines to actually place the order
    # WARNING: This will place a real order on mainnet!
    
    try:
        order_result = exchange.order(
            asset,  # coin
            is_buy,  # is_buy
            float(size),  # sz
            float(limit_price),  # limit_px
            {"limit": {"tif": "Gtc"}},  # order_type - Good-til-canceled
            reduce_only  # reduce_only
        )
        
        print(f"\n   ✅ Order placed successfully!")
        print(f"   Result: {json.dumps(order_result, indent=2)}")
        
        # Check if order was successful
        if 'status' in order_result:
            if order_result['status'] == 'ok':
                print(f"\n   Order Status: SUCCESS")
                if 'response' in order_result and 'data' in order_result['response']:
                    data = order_result['response']['data']
                    if 'statuses' in data:
                        for status in data['statuses']:
                            if 'resting' in status:
                                print(f"   Order ID: {status['resting']['oid']}")
            else:
                print(f"\n   Order Status: {order_result['status']}")
        
    except Exception as e:
        print(f"\n   ❌ Error placing order: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)

if __name__ == "__main__":
    main()
