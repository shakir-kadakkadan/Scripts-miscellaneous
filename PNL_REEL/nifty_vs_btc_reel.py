#!/usr/bin/env python3
"""
Nifty (USD) vs BTCUSD - Instagram Reel Video Generator
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
START_DATE = "20200101"  # YYYYMMDD format - will use common start date if this is earlier than available data
MONTHLY_SIP = 100  # $100 per month


def fetch_nifty_usd_data():
    """Fetch Nifty USD data from Investing.com API."""
    url = "https://tvc4.investing.com/ff8d3e148bb69cb57471e08ad54e598d/1770876582/56/56/23/history?symbol=17944&resolution=M&from=837756612&to=1770876672"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.investing.com/',
    }

    print("Fetching Nifty (USD) data...")
    response = requests.get(url, headers=headers)

    if response.status_code != 200 or not response.text:
        print(f"Error: Status {response.status_code}, Response: {response.text[:200] if response.text else 'Empty'}")
        raise Exception("Failed to fetch Nifty data - URL token may have expired")

    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame({
        'timestamp': data['t'],
        'nifty': data['c']  # Using close prices
    })
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['year_month'] = df['date'].dt.to_period('M')

    return df[['date', 'nifty', 'year_month']]


def fetch_btc_data():
    """Fetch BTCUSD data from Investing.com API."""
    url = "https://tvc4.investing.com/f82926a90b7b613bb317d5358fafc045/1770876684/56/56/23/history?symbol=1057391&resolution=M&from=837756696&to=1770876756"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.investing.com/',
    }

    print("Fetching BTCUSD data...")
    response = requests.get(url, headers=headers)

    if response.status_code != 200 or not response.text:
        print(f"Error: Status {response.status_code}, Response: {response.text[:200] if response.text else 'Empty'}")
        raise Exception("Failed to fetch BTC data - URL token may have expired")

    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame({
        'timestamp': data['t'],
        'btc': data['c']  # Using close prices
    })
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['year_month'] = df['date'].dt.to_period('M')

    return df[['date', 'btc', 'year_month']]


def prepare_data():
    """Fetch and merge Nifty USD and BTC data. Falls back to cached CSV if API fails."""
    csv_path = 'nifty_vs_btc_data.csv'

    try:
        nifty_df = fetch_nifty_usd_data()
        btc_df = fetch_btc_data()

        # Merge on year_month
        merged = pd.merge(
            nifty_df[['year_month', 'nifty']],
            btc_df[['year_month', 'btc']],
            on='year_month',
            how='inner'
        )

        merged['date'] = merged['year_month'].dt.to_timestamp()
        merged = merged.sort_values('date').reset_index(drop=True)
        print("Data fetched from API successfully.")

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
    output_path: str = "nifty_vs_btc_reel.mp4",
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
    nifty_sip = []
    btc_sip = []
    total_invested = []

    for i in range(len(df)):
        # Portfolio value at time i = sum of all previous investments grown to time i
        nifty_value = 0
        btc_value = 0
        for j in range(i + 1):
            # Each $100 invested at time j, grown to time i
            nifty_growth = df['nifty'].iloc[i] / df['nifty'].iloc[j]
            btc_growth = df['btc'].iloc[i] / df['btc'].iloc[j]
            nifty_value += monthly_sip * nifty_growth
            btc_value += monthly_sip * btc_growth
        nifty_sip.append(nifty_value)
        btc_sip.append(btc_value)
        total_invested.append((i + 1) * monthly_sip)

    nifty_normalized = pd.Series(nifty_sip)
    btc_normalized = pd.Series(btc_sip)
    invested_series = pd.Series(total_invested)

    # Setup figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Text scale multiplier (change this to resize all text)
    text_scale = 1.5

    # Colors
    nifty_color = '#00ff88'   # Green for Nifty
    btc_color = '#f7931a'     # Bitcoin orange

    # Create line objects
    line_nifty, = ax.plot([], [], lw=3, color=nifty_color, label='NIFTY')
    line_btc, = ax.plot([], [], lw=3, color=btc_color, label='BTC')

    # Set axis limits
    x_data = np.arange(len(df))
    y_min = 0  # Start from 0 so graph is visible from beginning
    y_max = max(nifty_normalized.max(), btc_normalized.max()) * 1.1

    ax.set_xlim(0, len(df) * 1.15)
    ax.set_ylim(y_min, y_max)

    # Subtitle with SIP info
    start_date_str = df['date'].iloc[0].strftime('%B %Y')
    subtitle = ax.text(0.5, 1.075, f'${MONTHLY_SIP} SIP every month since {start_date_str}',
                      transform=ax.transAxes,
                      fontsize=17 * text_scale, color='#cccccc', fontweight='bold',
                      ha='center', va='top')

    # Value boxes with names inside (moved further apart)
    nifty_box = ax.text(0.08, 0.93, '', transform=ax.transAxes,
                       fontsize=17 * text_scale, color=nifty_color, ha='center',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                                edgecolor=nifty_color, linewidth=2))

    btc_box = ax.text(0.92, 0.93, '', transform=ax.transAxes,
                      fontsize=17 * text_scale, color=btc_color, ha='center',
                      fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                               edgecolor=btc_color, linewidth=2))

    # Percentage labels (smaller text below boxes)
    nifty_pct_text = ax.text(0.08, 0.86, '', transform=ax.transAxes,
                            fontsize=13 * text_scale, color=nifty_color, ha='center',
                            fontweight='bold')
    btc_pct_text = ax.text(0.92, 0.86, '', transform=ax.transAxes,
                           fontsize=13 * text_scale, color=btc_color, ha='center',
                           fontweight='bold')

    # Total invested display - above date
    invested_text = ax.text(0.32, 0.95, '', transform=ax.transAxes,
                           fontsize=12 * text_scale, color='#aaaaaa',
                           ha='left', va='center', fontweight='bold')

    # Date display - left aligned after Nifty box
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
    nifty_label = ax.text(0, 0, '', fontsize=14 * text_scale, fontweight='bold',
                         color=nifty_color, va='center')
    btc_label = ax.text(0, 0, '', fontsize=14 * text_scale, fontweight='bold',
                        color=btc_color, va='center')

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
        line_nifty.set_data([], [])
        line_btc.set_data([], [])
        date_text.set_text('')
        invested_text.set_text('')
        nifty_label.set_text('')
        btc_label.set_text('')
        nifty_box.set_text('NIFTY\n$0')
        btc_box.set_text('BTC\n$0')
        nifty_pct_text.set_text('0%')
        btc_pct_text.set_text('0%')
        return [line_nifty, line_btc, date_text, invested_text, nifty_label, btc_label,
                nifty_box, btc_box, nifty_pct_text, btc_pct_text]

    def animate(frame):
        # Calculate progress (clamp to animation_frames for pause at end)
        effective_frame = min(frame, animation_frames - 1)
        progress = (effective_frame + 1) / animation_frames
        n_show = max(1, int(progress * n_points))

        # Update lines
        x_show = x_data[:n_show]
        nifty_show = nifty_normalized.iloc[:n_show].values
        btc_show = btc_normalized.iloc[:n_show].values

        line_nifty.set_data(x_show, nifty_show)
        line_btc.set_data(x_show, btc_show)

        # Update date (year month format for fixed width)
        current_date = df['date'].iloc[n_show - 1]
        date_text.set_text(current_date.strftime('%Y %B'))

        # Update value labels at line ends (show portfolio value)
        x_end = x_show[-1]
        nifty_end = nifty_show[-1]
        btc_end = btc_show[-1]

        # Format values in USD
        def fmt_val(v):
            if v >= 1000000:
                return f'${v/1000000:.2f}M'
            elif v >= 1000:
                return f'${v/1000:.1f}K'
            else:
                return f'${v:.0f}'

        offset = len(df) * 0.02
        nifty_label.set_position((x_end + offset, nifty_end))
        nifty_label.set_text(fmt_val(nifty_end))

        btc_label.set_position((x_end + offset, btc_end))
        btc_label.set_text(fmt_val(btc_end))

        # Calculate total invested and % returns
        invested = invested_series.iloc[n_show - 1]
        nifty_pct = ((nifty_end - invested) / invested) * 100
        btc_pct = ((btc_end - invested) / invested) * 100

        # Update total invested text
        invested_text.set_text(f'Total Invested: {fmt_val(invested)}')

        # Update boxes with name and value
        nifty_box.set_text(f'NIFTY\n{fmt_val(nifty_end)}')
        btc_box.set_text(f'BTC\n{fmt_val(btc_end)}')

        # Update percentage labels (smaller text below boxes)
        nifty_pct_text.set_text(f'{nifty_pct:+.0f}%')
        btc_pct_text.set_text(f'{btc_pct:+.0f}%')

        return [line_nifty, line_btc, date_text, invested_text, nifty_label, btc_label,
                nifty_box, btc_box, nifty_pct_text, btc_pct_text]

    print(f"Creating animation with {total_frames} frames...")

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000/fps, blit=True
    )

    # Save video
    print(f"Saving video to: {output_path}")
    writer = FFMpegWriter(fps=fps, metadata={'title': 'Nifty vs BTC'}, bitrate=8000)
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
    df.to_csv('nifty_vs_btc_data.csv', index=False)
    print(f"Data saved to: nifty_vs_btc_data.csv")

    # Create video
    video_path = create_reel_video(
        df=df,
        output_path="nifty_vs_btc_reel.mp4",
        fps=30,
        duration_seconds=30
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
