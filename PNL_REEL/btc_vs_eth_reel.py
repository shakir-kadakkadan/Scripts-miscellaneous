#!/usr/bin/env python3
"""
BTC vs ETH - Instagram Reel Video Generator
Uses Binance API for monthly candles.
Creates a 9:16 aspect ratio animated comparison video.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
from datetime import datetime
import json
import subprocess
import os

# Configuration
START_DATE = "19700101"  # YYYYMMDD format
MONTHLY_SIP = 100  # $100 per month


def fetch_binance_monthly_candles(symbol: str):
    """Fetch monthly candles from Binance API."""
    url = "https://api.binance.com/api/v3/klines"

    params = {
        'symbol': symbol,
        'interval': '1M',  # Monthly candles
        'limit': 1000
    }

    print(f"Fetching {symbol} monthly data from Binance...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch {symbol} data: {response.text}")

    data = response.json()

    # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['year_month'] = df['date'].dt.to_period('M')

    return df[['date', 'close', 'year_month']]


def fetch_btc_data():
    """Fetch BTCUSDT data from Binance."""
    df = fetch_binance_monthly_candles('BTCUSDT')
    df = df.rename(columns={'close': 'btc'})
    return df


def fetch_eth_data():
    """Fetch ETHUSDT data from Binance."""
    df = fetch_binance_monthly_candles('ETHUSDT')
    df = df.rename(columns={'close': 'eth'})
    return df


def prepare_data():
    """Fetch and merge BTC and ETH data. Falls back to cached CSV if API fails."""
    csv_path = 'btc_vs_eth_data.csv'

    try:
        btc_df = fetch_btc_data()
        eth_df = fetch_eth_data()

        # Merge on year_month
        merged = pd.merge(
            btc_df[['year_month', 'btc']],
            eth_df[['year_month', 'eth']],
            on='year_month',
            how='inner'
        )

        merged['date'] = merged['year_month'].dt.to_timestamp()
        merged = merged.sort_values('date').reset_index(drop=True)
        print("Data fetched from Binance API successfully.")

    except Exception as e:
        print(f"API fetch failed: {e}")
        if os.path.exists(csv_path):
            print(f"Loading cached data from {csv_path}...")
            merged = pd.read_csv(csv_path)
            merged['date'] = pd.to_datetime(merged['date'])
            merged['year_month'] = merged['date'].dt.to_period('M')
        else:
            raise Exception(f"No cached data available at {csv_path}")

    # Apply START_DATE filter
    requested_start = pd.to_datetime(START_DATE, format='%Y%m%d')
    earliest_date = merged['date'].min()

    # Use requested start date if available, otherwise use earliest available
    if requested_start >= earliest_date:
        effective_start = requested_start
        print(f"Using requested start date: {effective_start.strftime('%Y-%m-%d')}")
    else:
        effective_start = earliest_date
        print(f"Requested start date {requested_start.strftime('%Y-%m-%d')} is earlier than available data.")
        print(f"Using earliest available date: {effective_start.strftime('%Y-%m-%d')}")

    # Filter data from effective start date
    merged = merged[merged['date'] >= effective_start].reset_index(drop=True)

    print(f"Data range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"Data points: {len(merged)}")

    return merged


def create_reel_video(
    df: pd.DataFrame,
    output_path: str = "btc_vs_eth_reel.mp4",
    fps: int = 30,
    duration_seconds: int = 30
):
    """
    Create Instagram Reel sized video (1080x1920, 9:16 aspect ratio).
    """

    # Instagram Reel dimensions
    width_px = 1080
    height_px = 1920
    dpi = 100
    figsize = (width_px / dpi, height_px / dpi)

    pause_seconds = 3  # Pause at end
    animation_frames = fps * duration_seconds
    pause_frames = fps * pause_seconds
    total_frames = animation_frames + pause_frames
    n_points = len(df)

    # Calculate SIP investment
    monthly_sip = MONTHLY_SIP  # $100 per month

    # Calculate SIP portfolio value for each point in time
    btc_sip = []
    eth_sip = []
    total_invested = []

    for i in range(len(df)):
        # Portfolio value at time i = sum of all previous investments grown to time i
        btc_value = 0
        eth_value = 0
        for j in range(i + 1):
            # Each $100 invested at time j, grown to time i
            btc_growth = df['btc'].iloc[i] / df['btc'].iloc[j]
            eth_growth = df['eth'].iloc[i] / df['eth'].iloc[j]
            btc_value += monthly_sip * btc_growth
            eth_value += monthly_sip * eth_growth
        btc_sip.append(btc_value)
        eth_sip.append(eth_value)
        total_invested.append((i + 1) * monthly_sip)

    btc_normalized = pd.Series(btc_sip)
    eth_normalized = pd.Series(eth_sip)
    invested_series = pd.Series(total_invested)

    # Setup figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Text scale multiplier (change this to resize all text)
    text_scale = 1.5

    # Colors
    btc_color = '#f7931a'     # Bitcoin orange
    eth_color = '#627eea'     # Ethereum blue

    # Create line objects
    line_btc, = ax.plot([], [], lw=3, color=btc_color, label='BTC')
    line_eth, = ax.plot([], [], lw=3, color=eth_color, label='ETH')

    # Set axis limits
    x_data = np.arange(len(df))
    y_min = 0  # Start from 0 so graph is visible from beginning
    y_max = max(btc_normalized.max(), eth_normalized.max()) * 1.1

    ax.set_xlim(0, len(df) * 1.15)
    ax.set_ylim(y_min, y_max)

    # Subtitle with SIP info
    start_date_str = df['date'].iloc[0].strftime('%B %Y')
    subtitle = ax.text(0.5, 1.075, f'${MONTHLY_SIP} SIP every month since {start_date_str}',
                      transform=ax.transAxes,
                      fontsize=17 * text_scale, color='#cccccc', fontweight='bold',
                      ha='center', va='top')

    # Value boxes with names inside (moved further apart)
    btc_box = ax.text(0.08, 0.93, '', transform=ax.transAxes,
                       fontsize=17 * text_scale, color=btc_color, ha='center',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                                edgecolor=btc_color, linewidth=2))

    eth_box = ax.text(0.92, 0.93, '', transform=ax.transAxes,
                      fontsize=17 * text_scale, color=eth_color, ha='center',
                      fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                               edgecolor=eth_color, linewidth=2))

    # Percentage labels (smaller text below boxes)
    btc_pct_text = ax.text(0.08, 0.86, '', transform=ax.transAxes,
                            fontsize=13 * text_scale, color=btc_color, ha='center',
                            fontweight='bold')
    eth_pct_text = ax.text(0.92, 0.86, '', transform=ax.transAxes,
                           fontsize=13 * text_scale, color=eth_color, ha='center',
                           fontweight='bold')

    # Total invested display - above date
    invested_text = ax.text(0.32, 0.95, '', transform=ax.transAxes,
                           fontsize=12 * text_scale, color='#aaaaaa',
                           ha='left', va='center', fontweight='bold')

    # Date display - left aligned after BTC box
    date_text = ax.text(0.32, 0.91, '', transform=ax.transAxes,
                       fontsize=16 * text_scale, color='white',
                       ha='left', va='center', fontweight='bold')

    # Instagram handle at bottom with icon
    insta_icon_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/132px-Instagram_logo_2016.svg.png'
    try:
        icon_response = requests.get(insta_icon_url)
        icon_img = Image.open(BytesIO(icon_response.content))
        icon_img = icon_img.resize((40, 40), Image.LANCZOS)
        imagebox = OffsetImage(icon_img, zoom=1)
        ab = AnnotationBbox(imagebox, (0.42, -0.08), transform=ax.transAxes,
                           frameon=False, box_alignment=(1, 0.5))
        ax.add_artist(ab)
    except Exception:
        pass  # Skip icon if download fails

    insta_text = ax.text(0.44, -0.08, '@algo_vs_discretionary_trader',
                        transform=ax.transAxes,
                        fontsize=12 * text_scale, color='#888888',
                        ha='left', va='center', fontweight='bold')

    # Value labels at line ends
    btc_label = ax.text(0, 0, '', fontsize=14 * text_scale, fontweight='bold',
                         color=btc_color, va='center')
    eth_label = ax.text(0, 0, '', fontsize=14 * text_scale, fontweight='bold',
                        color=eth_color, va='center')

    # Style axes (no y-axis label)

    # Format y-axis to show USD
    def format_usd(x, pos):
        if x == 0:
            return '$0'
        elif x >= 1000000:
            return f'${x/1000000:.1f}M'
        elif x >= 1000:
            return f'${x/1000:.1f}K'
        else:
            return f'${x:.0f}'
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_usd))
    ax.tick_params(axis='y', length=0)  # Hide Y-axis tick marks
    ax.set_yticklabels([])  # Hide Y-axis scale labels
    ax.grid(True, alpha=0.2, color='white')

    # Ensure 0 is included in y-ticks
    ax.set_yticks([0] + list(ax.get_yticks()[ax.get_yticks() > 0]))

    # Hide x-axis ticks (we show date separately)
    ax.set_xticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Adjust layout for mobile (graph with space for header and bottom padding)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.16)

    def init():
        line_btc.set_data([], [])
        line_eth.set_data([], [])
        date_text.set_text('')
        invested_text.set_text('')
        btc_label.set_text('')
        eth_label.set_text('')
        btc_box.set_text('BTC\n$0')
        eth_box.set_text('ETH\n$0')
        btc_pct_text.set_text('0%')
        eth_pct_text.set_text('0%')
        return [line_btc, line_eth, date_text, invested_text, btc_label, eth_label,
                btc_box, eth_box, btc_pct_text, eth_pct_text]

    def animate(frame):
        # Calculate progress (clamp to animation_frames for pause at end)
        effective_frame = min(frame, animation_frames - 1)
        progress = (effective_frame + 1) / animation_frames
        n_show = max(1, int(progress * n_points))

        # Update lines
        x_show = x_data[:n_show]
        btc_show = btc_normalized.iloc[:n_show].values
        eth_show = eth_normalized.iloc[:n_show].values

        line_btc.set_data(x_show, btc_show)
        line_eth.set_data(x_show, eth_show)

        # Update date (year month format for fixed width)
        current_date = df['date'].iloc[n_show - 1]
        date_text.set_text(current_date.strftime('%Y %B'))

        # Update value labels at line ends (show portfolio value)
        x_end = x_show[-1]
        btc_end = btc_show[-1]
        eth_end = eth_show[-1]

        # Format values in USD
        def fmt_val(v):
            if v >= 1000000:
                return f'${v/1000000:.2f}M'
            elif v >= 1000:
                return f'${v/1000:.1f}K'
            else:
                return f'${v:.0f}'

        offset = len(df) * 0.02
        btc_label.set_position((x_end + offset, btc_end))
        btc_label.set_text(fmt_val(btc_end))

        eth_label.set_position((x_end + offset, eth_end))
        eth_label.set_text(fmt_val(eth_end))

        # Calculate total invested and % returns
        invested = invested_series.iloc[n_show - 1]
        btc_pct = ((btc_end - invested) / invested) * 100
        eth_pct = ((eth_end - invested) / invested) * 100

        # Update total invested text
        invested_text.set_text(f'Total Invested: {fmt_val(invested)}')

        # Update boxes with name and value
        btc_box.set_text(f'BTC\n{fmt_val(btc_end)}')
        eth_box.set_text(f'ETH\n{fmt_val(eth_end)}')

        # Update percentage labels (smaller text below boxes)
        btc_pct_text.set_text(f'{btc_pct:+.0f}%')
        eth_pct_text.set_text(f'{eth_pct:+.0f}%')

        return [line_btc, line_eth, date_text, invested_text, btc_label, eth_label,
                btc_box, eth_box, btc_pct_text, eth_pct_text]

    print(f"Creating animation with {total_frames} frames...")

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000/fps, blit=True
    )

    # Save video
    print(f"Saving video to: {output_path}")
    writer = FFMpegWriter(fps=fps, metadata={'title': 'BTC vs ETH'}, bitrate=8000)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Video saved successfully: {output_path}")
    return output_path


def add_background_music(
    video_path: str,
    audio_path: str,
    output_path: str = None,
    audio_start_sec: int = 16
):
    """
    Add background music to the video using ffmpeg.

    Args:
        video_path: Path to the input video
        audio_path: Path to the audio file
        output_path: Path for output video (default: replaces original)
        audio_start_sec: Start position in the audio file (seconds)
    """
    if output_path is None:
        output_path = video_path.replace('.mp4', '_with_audio.mp4')

    print(f"\nAdding background music from {audio_start_sec}s...")

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-ss', str(audio_start_sec),
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Video with audio saved: {output_path}")
        return output_path
    else:
        print(f"Error adding audio: {result.stderr}")
        return None


def main():
    # Fetch and prepare data
    df = prepare_data()

    # Save data for reference
    df.to_csv('btc_vs_eth_data.csv', index=False)
    print(f"Data saved to: btc_vs_eth_data.csv")

    # Create video
    video_path = create_reel_video(
        df=df,
        output_path="btc_vs_eth_reel.mp4",
        fps=30,
        duration_seconds=17
    )

    # Add background music
    audio_path = os.path.expanduser(
        "~/X-AURA [QL8eq5KRWbQ].mp3"
    )
    if os.path.exists(audio_path):
        add_background_music(
            video_path=video_path,
            audio_path=audio_path,
            audio_start_sec=16
        )
    else:
        print(f"Audio file not found: {audio_path}")


if __name__ == '__main__':
    main()
