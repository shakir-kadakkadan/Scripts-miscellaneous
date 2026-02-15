#!/usr/bin/env python3
"""
Nifty vs Global Indices - Instagram Reel Video Generator
Compares Nifty 50 with SPX, DJI, NDX, FTSE, and NIKKEI using yfinance data.
Creates a 9:16 aspect ratio animated comparison video.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
from datetime import datetime
import os
import subprocess
import requests

# Configuration
START_DATE = "2000-01-01"  # Common start date
MONTHLY_SIP = 100  # $100 per month

# Indices to compare
INDICES = {
    'NIFTY': {'symbol': '^NSEI', 'color': '#00ff88', 'box': True, 'label': 'Nifty'},
    'NDX': {'symbol': '^NDX', 'color': '#00ffff', 'box': True, 'label': 'Nasdaq'},  # Cyan
    'SPX': {'symbol': '^GSPC', 'color': '#ffcc00', 'box': False, 'label': 'S&P 500'},  # Yellow
    'DJI': {'symbol': '^DJI', 'color': '#ff00cc', 'box': False, 'label': 'Dow'},    # Pink/Magenta
    'FTSE': {'symbol': '^FTSE', 'color': '#ff0055', 'box': False, 'label': 'FTSE'},   # Red/Pink
    'NIKKEI': {'symbol': '^N225', 'color': '#ffffff', 'box': False, 'label': 'Nikkei'} # White
}

def fetch_data():
    """Fetch monthly data for all indices using yfinance."""
    print("Fetching data from yfinance...")
    
    # Currencies to fetch
    CURRENCIES = {
        'INR': 'INR=X',     # INR per USD
        'JPY': 'JPY=X',     # JPY per USD
        'GBP': 'GBPUSD=X'   # USD per GBP (Inverse)
    }
    
    # Store currency data
    currency_data = {}
    for code, symbol in CURRENCIES.items():
        print(f"Fetching currency {code} ({symbol})...")
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max", interval="1mo")
        if not hist.empty:
            df = hist.reset_index()[['Date', 'Close']]
            df.columns = ['date', 'rate']
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df['year_month'] = df['date'].dt.to_period('M')
            currency_data[code] = df[['year_month', 'rate']]
            
    data_frames = {}
    
    for name, info in INDICES.items():
        print(f"Fetching {name} ({info['symbol']})...")
        ticker = yf.Ticker(info['symbol'])
        # Fetch history with some buffer before start date to ensure we have data
        hist = ticker.history(period="max", interval="1mo")
        
        if hist.empty:
            print(f"Warning: No data found for {name}")
            continue
            
        # Reset index to get Date as column
        df = hist.reset_index()
        
        # Standardize columns
        df = df[['Date', 'Close']]
        df.columns = ['date', 'price']
        
        # Ensure date is timezone-naive for comparison
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        # Create year_month for merging/aligning
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Convert to USD if needed
        # NIFTY (INR), NIKKEI (JPY), FTSE (GBP)
        
        if name == 'NIFTY':
            # Divide by INR rate
            df = pd.merge(df, currency_data['INR'], on='year_month', how='left')
            df['price'] = df['price'] / df['rate'] # Convert to USD terms
            df = df.dropna(subset=['price'])
        elif name == 'NIKKEI':
            # Divide by JPY rate
            df = pd.merge(df, currency_data['JPY'], on='year_month', how='left')
            df['price'] = df['price'] / df['rate'] # Convert to USD terms
            df = df.dropna(subset=['price'])
        elif name == 'FTSE':
            # Multiply by GBPUSD rate
            df = pd.merge(df, currency_data['GBP'], on='year_month', how='left')
            df['price'] = df['price'] * df['rate'] # Convert to USD terms
            df = df.dropna(subset=['price'])
        # SPX, NDX, DJI are already in USD
        
        # Keep only necessary columns
        data_frames[name] = df[['date', 'price', 'year_month']]

    return data_frames

def prepare_data(data_frames):
    """Align all datasets to a common timeframe."""
    
    # deep copy to avoid modifying originals
    dfs = {k: v.copy() for k, v in data_frames.items()}
    
    # Find common start date (latest of all start dates or configured start date)
    start_dates = [df['date'].min() for df in dfs.values()]
    max_start_date = max(start_dates)
    
    config_start = pd.to_datetime(START_DATE)
    effective_start = max(max_start_date, config_start)
    
    print(f"Aligning data from: {effective_start.strftime('%Y-%m-%d')}")
    
    # Filter and re-index all DFs
    aligned_data = {}
    
    # We need a common date range to step through
    # Create a monthly frequency date range from start to now
    common_dates = pd.date_range(start=effective_start, end=datetime.now(), freq='MS') # Month Start
    period_range = common_dates.to_period('M')
    
    # Create a master dataframe with the period range
    master_df = pd.DataFrame({'year_month': period_range})
    master_df['date'] = master_df['year_month'].dt.to_timestamp()
    
    for name, df in dfs.items():
        # Merge each index data into master
        # We rename 'price' to the index name
        temp_df = df[['year_month', 'price']].rename(columns={'price': name})
        
        # Merge
        master_df = pd.merge(master_df, temp_df, on='year_month', how='left')
        
        # Forward fill missing data (e.g. holidays)
        master_df[name] = master_df[name].ffill()
    
    # Drop any rows that still have NaNs (if start dates didn't perfectly align)
    master_df = master_df.dropna().reset_index(drop=True)
    
    print(f"Final data points: {len(master_df)}")
    return master_df

def calculate_sip(df):
    """Calculate SIP returns for all indices."""
    
    n_points = len(df)
    results = {name: [] for name in INDICES.keys()}
    total_invested = []
    
    print("Calculating SIP returns...")
    
    prices = {name: df[name].values for name in INDICES.keys()}
    
    for i in range(n_points):
        # Current invested amount
        current_invested = (i + 1) * MONTHLY_SIP
        total_invested.append(current_invested)
        
        for name in INDICES.keys():
            # Current price
            curr_price = prices[name][i]
            
            # Calculate value: Sum of (Monthly Investment * (Current Price / Price at Investment))
            # Vectorized calculation for speed
            # slice of prices from 0 to i+1
            past_prices = prices[name][:i+1]
            
            # Growth factors for each installment: curr_price / past_prices
            growth_factors = curr_price / past_prices
            
            # Total value
            value = np.sum(growth_factors * MONTHLY_SIP)
            results[name].append(value)
            
    # Add results to dataframe
    for name, values in results.items():
        df[f'{name}_value'] = values
        
    df['total_invested'] = total_invested
    return df

def create_reel_video(
    df: pd.DataFrame,
    output_path: str = "nifty_vs_global_indices_reel.mp4",
    fps: int = 30,
    duration_seconds: int = 30
):
    """Create the animated video."""
    
    # Dimensions
    width_px = 1080
    height_px = 1920
    dpi = 100
    figsize = (width_px / dpi, height_px / dpi)
    
    pause_seconds = 3
    animation_frames = fps * duration_seconds
    pause_frames = fps * pause_seconds
    total_frames = animation_frames + pause_frames
    
    # Setup plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Common text settings
    text_scale = 1.3
    
    # Initialize lines
    lines = {}
    
    # Plot Total Invested Line
    line_invested, = ax.plot([], [], lw=2, color='#888888', linestyle='--', label='Invested', alpha=0.7)
    
    # Create graphic objects for indices
    for name, info in INDICES.items():
        # Line
        line, = ax.plot([], [], lw=2.5, color=info['color'], label=name)
        lines[name] = line
        
        # Remove floating Line Labels (as per previous request)
        
    # Text List Logic
    # Position: Left Center
    # We will use two columns: Amount (Right Aligned) | Name (Left Aligned)
    
    text_objects = {} # key -> {'amount_obj': Text, 'name_obj': Text}
    
    # Display configuration
    display_items = [
        {'key': 'Invested', 'name': 'Total Invested', 'color': '#888888'},
        {'key': 'NIFTY', 'name': 'Nifty 50 (India)', 'color': INDICES['NIFTY']['color']},
        {'key': 'SPX', 'name': 'S&P 500 (US)', 'color': INDICES['SPX']['color']},
        {'key': 'NDX', 'name': 'Nasdaq 100 (US)', 'color': INDICES['NDX']['color']},
        {'key': 'DJI', 'name': 'Dow Jones (US)', 'color': INDICES['DJI']['color']},
        {'key': 'FTSE', 'name': 'FTSE 100 (UK)', 'color': INDICES['FTSE']['color']},
        {'key': 'NIKKEI', 'name': 'Nikkei 225 (Japan)', 'color': INDICES['NIKKEI']['color']},
    ]
    
    # Coordinates
    start_y = 0.70  # Start from upper middle
    step_y = 0.06   # Space between lines
    col_amount_x = 0.20 # Right align amount to here (Shifted Left)
    col_name_x = 0.23   # Left align name from here (Shifted Left)
    
    for i, item in enumerate(display_items):
        y_pos = start_y - (i * step_y)
        key = item['key']
        color = item['color']
        
        # Amount Text (Right Aligned)
        t_amt = ax.text(col_amount_x, y_pos, '$0', transform=ax.transAxes,
                       fontsize=14 * text_scale, color=color, fontweight='bold',
                       ha='right', va='center')
                       
        # Name Text (Left Aligned)
        t_name = ax.text(col_name_x, y_pos, item['name'], transform=ax.transAxes,
                        fontsize=14 * text_scale, color=color, fontweight='bold',
                        ha='left', va='center')
                        
        # Store display name in the object so we can restore it in animate
        text_objects[key] = {'amt': t_amt, 'name': t_name, 'display_name': item['name']}

    # Note about USD (Below first text "Total Invested")
    # First item is at start_y (0.70). Next is at start_y - step_y (0.64).
    # Place note at 0.67
    ax.text(col_name_x, start_y - 0.03, '(All values in USD)', transform=ax.transAxes,
            fontsize=10 * text_scale, color='#aaaaaa', ha='left', va='center', style='italic')

    # Date Display - Top Center
    date_text = ax.text(0.5, 0.85, '', transform=ax.transAxes,
                       fontsize=18 * text_scale, color='white',
                       ha='center', va='center', fontweight='bold')
                       
    # SIP Subtitle
    start_date_str = df['date'].iloc[0].strftime('%B %Y')
    subtitle = ax.text(0.5, 1.05, f'${MONTHLY_SIP} SIP Monthly Since {start_date_str}',
                      transform=ax.transAxes,
                      fontsize=18 * text_scale, color='#cccccc', fontweight='bold',
                      ha='center', va='top')
    
    # Format Helpers
    def format_usd(v):
        if v >= 1000000: return f'${v/1000000:.2f}M'
        elif v >= 1000: return f'${v/1000:.0f}K'
        else: return f'${v:.0f}'
        
    # Axis Setup
    ax.tick_params(axis='y', length=0)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.grid(True, alpha=0.1, color='white')
    for spine in ax.spines.values(): spine.set_visible(False)
    
    # Calculate Axis Limits
    max_val = 0
    for name in INDICES.keys():
        max_val = max(max_val, df[f'{name}_value'].max())
        
    ax.set_xlim(0, len(df)) 
    ax.set_ylim(0, max_val * 1.1)
    
    # Instagram Handle
    ax.text(0.5, -0.05, '@algo_vs_discretionary_trader', transform=ax.transAxes,
            fontsize=12 * text_scale, color='#666666', ha='center')

    # Layout
    plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.1)

    # Init Function
    def init():
        for line in lines.values(): line.set_data([], [])
        line_invested.set_data([], [])
        for obj in text_objects.values():
            obj['amt'].set_text('')
            obj['name'].set_text('')
        date_text.set_text('')
        
        # Flatten list of artists
        artists = list(lines.values()) + [line_invested, date_text]
        for obj in text_objects.values():
            artists.append(obj['amt'])
            artists.append(obj['name'])
        return artists

    # Animate Function
    def animate(frame):
        # Progress
        effective_frame = min(frame, animation_frames - 1)
        progress = (effective_frame + 1) / animation_frames
        n_show = max(1, int(progress * len(df)))
        
        current_date = df['date'].iloc[n_show - 1]
        date_text.set_text(current_date.strftime('%Y %B'))
        
        # Update lines and text
        x_data = np.arange(n_show)
        
        # Update Invested Line
        y_invested = df['total_invested'].iloc[:n_show].values
        line_invested.set_data(x_data, y_invested)
        
        # Update Invested Text
        curr_invested = y_invested[-1]
        text_objects['Invested']['amt'].set_text(format_usd(curr_invested))
        text_objects['Invested']['name'].set_text(text_objects['Invested']['display_name'])
        
        # Update Indices
        for name in INDICES.keys():
            # Update Line
            y_data = df[f'{name}_value'].iloc[:n_show].values
            lines[name].set_data(x_data, y_data)
            
            # Update Text
            val = y_data[-1]
            text_objects[name]['amt'].set_text(format_usd(val))
            text_objects[name]['name'].set_text(text_objects[name]['display_name'])
            
        # Return artists
        artists = list(lines.values()) + [line_invested, date_text]
        for obj in text_objects.values():
            artists.append(obj['amt'])
            artists.append(obj['name'])
        return artists

    print(f"Creating animation ({total_frames} frames)...")
    anim = FuncAnimation(fig, animate, init_func=init, frames=int(total_frames), interval=1000/fps, blit=True)
    
    print(f"Saving to {output_path}...")
    writer = FFMpegWriter(fps=fps, metadata={'title': 'Nifty vs Global'}, bitrate=8000)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print("Done!")
    return output_path

def add_audio(video_path):
    """Add audio if available."""
    audio_path = os.path.expanduser("~/X-AURA [QL8eq5KRWbQ].mp3")
    output_path = video_path.replace('.mp4', '_with_audio.mp4')
    
    if not os.path.exists(audio_path):
        print("Audio file not found.")
        return

    print("Adding audio...")
    cmd = [
        'ffmpeg', '-y', '-i', video_path, '-ss', '16', '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
        '-shortest', output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Audio added: {output_path}")

def main():
    # 1. Fetch
    dfs = fetch_data()
    
    # 2. Prepare
    df = prepare_data(dfs)
    
    # 3. Calculate
    df = calculate_sip(df)
    
    # 4. Create Video
    video_path = create_reel_video(df, duration_seconds=16)
    
    # 5. Add Audio
    add_audio(video_path)

if __name__ == "__main__":
    main()
