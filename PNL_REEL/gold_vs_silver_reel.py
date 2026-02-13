#!/usr/bin/env python3
"""
Gold vs Silver - Instagram Reel Video Generator
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


def fetch_gold_data():
    """Fetch Gold data from Zerodha API."""
    url = "https://api.zerodhafundhouse.com/api/v1/index/historical"
    params = {
        "code": "GOLD995",
        "duration": "max",
        "aggregate": "true"
    }

    print("Fetching Gold data...")
    response = requests.get(url, params=params)
    data = response.json()

    # Convert to DataFrame
    points = data['data']['points']
    df = pd.DataFrame(points)
    df['date'] = pd.to_datetime(df['ts'])
    df['gold'] = df['val']
    df['year_month'] = df['date'].dt.to_period('M')

    return df[['date', 'gold', 'year_month']]


def fetch_silver_data():
    """Fetch Silver data from Zerodha API."""
    url = "https://api.zerodhafundhouse.com/api/v1/index/historical"
    params = {
        "code": "SILVER",
        "duration": "max",
        "aggregate": "true"
    }

    print("Fetching Silver data...")
    response = requests.get(url, params=params)
    data = response.json()

    # Convert to DataFrame
    points = data['data']['points']
    df = pd.DataFrame(points)
    df['date'] = pd.to_datetime(df['ts'])
    df['silver'] = df['val']
    df['year_month'] = df['date'].dt.to_period('M')

    return df[['date', 'silver', 'year_month']]


def prepare_data():
    """Fetch and merge Gold and Silver data."""
    gold_df = fetch_gold_data()
    silver_df = fetch_silver_data()

    # Merge on year_month
    merged = pd.merge(
        gold_df[['year_month', 'gold']],
        silver_df[['year_month', 'silver']],
        on='year_month',
        how='inner'
    )

    merged['date'] = merged['year_month'].dt.to_timestamp()
    merged = merged.sort_values('date').reset_index(drop=True)

    print(f"Data range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"Data points: {len(merged)}")

    return merged


def create_reel_video(
    df: pd.DataFrame,
    output_path: str = "gold_vs_silver_reel.mp4",
    fps: int = 30,
    duration_seconds: int = 15
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

    # Calculate SIP investment (Rs.10K per month)
    monthly_sip = 10000  # Rs.10K per month

    # Calculate SIP portfolio value for each point in time
    gold_sip = []
    silver_sip = []
    total_invested = []

    for i in range(len(df)):
        # Portfolio value at time i = sum of all previous investments grown to time i
        gold_value = 0
        silver_value = 0
        for j in range(i + 1):
            # Each Rs.10K invested at time j, grown to time i
            gold_growth = df['gold'].iloc[i] / df['gold'].iloc[j]
            silver_growth = df['silver'].iloc[i] / df['silver'].iloc[j]
            gold_value += monthly_sip * gold_growth
            silver_value += monthly_sip * silver_growth
        gold_sip.append(gold_value)
        silver_sip.append(silver_value)
        total_invested.append((i + 1) * monthly_sip)

    gold_normalized = pd.Series(gold_sip)
    silver_normalized = pd.Series(silver_sip)
    invested_series = pd.Series(total_invested)

    # Setup figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Text scale multiplier (change this to resize all text)
    text_scale = 1.5

    # Colors
    gold_color = '#ffd700'   # Gold color
    silver_color = '#c0c0c0'  # Silver color

    # Create line objects
    line_gold, = ax.plot([], [], lw=3, color=gold_color, label='GOLD')
    line_silver, = ax.plot([], [], lw=3, color=silver_color, label='SILVER')

    # Set axis limits
    x_data = np.arange(len(df))
    y_min = 0  # Start from 0 so graph is visible from beginning
    y_max = max(gold_normalized.max(), silver_normalized.max()) * 1.1

    ax.set_xlim(0, len(df) * 1.15)
    ax.set_ylim(y_min, y_max)

    # Subtitle with SIP info
    start_date = df['date'].iloc[0].strftime('%B %Y')
    subtitle = ax.text(0.5, 1.075, f'Rs.10K SIP every month since {start_date}',
                      transform=ax.transAxes,
                      fontsize=17 * text_scale, color='#cccccc', fontweight='bold',
                      ha='center', va='top')

    # Value boxes with names inside (moved further apart)
    gold_box = ax.text(0.08, 0.93, '', transform=ax.transAxes,
                       fontsize=17 * text_scale, color=gold_color, ha='center',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                                edgecolor=gold_color, linewidth=2))

    silver_box = ax.text(0.92, 0.93, '', transform=ax.transAxes,
                      fontsize=17 * text_scale, color=silver_color, ha='center',
                      fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                               edgecolor=silver_color, linewidth=2))

    # Percentage labels (smaller text below boxes)
    gold_pct_text = ax.text(0.08, 0.86, '', transform=ax.transAxes,
                            fontsize=13 * text_scale, color=gold_color, ha='center',
                            fontweight='bold')
    silver_pct_text = ax.text(0.92, 0.86, '', transform=ax.transAxes,
                           fontsize=13 * text_scale, color=silver_color, ha='center',
                           fontweight='bold')

    # Total invested display - above date
    invested_text = ax.text(0.32, 0.95, '', transform=ax.transAxes,
                           fontsize=12 * text_scale, color='#aaaaaa',
                           ha='left', va='center', fontweight='bold')

    # Date display - left aligned after Gold box
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
    gold_label = ax.text(0, 0, '', fontsize=14 * text_scale, fontweight='bold',
                         color=gold_color, va='center')
    silver_label = ax.text(0, 0, '', fontsize=14 * text_scale, fontweight='bold',
                        color=silver_color, va='center')

    # Style axes (no y-axis label)

    # Format y-axis to show lakhs/crores
    def format_lakhs(x, pos):
        if x == 0:
            return 'Rs.0'
        elif x >= 10000000:  # 1 Crore = 100 Lakhs
            return f'Rs.{x/10000000:.1f}Cr'
        elif x >= 100000:
            return f'Rs.{x/100000:.1f}L'
        else:
            return f'Rs.{x/1000:.0f}K'
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lakhs))
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
        line_gold.set_data([], [])
        line_silver.set_data([], [])
        date_text.set_text('')
        invested_text.set_text('')
        gold_label.set_text('')
        silver_label.set_text('')
        gold_box.set_text('GOLD\nRs.0')
        silver_box.set_text('SILVER\nRs.0')
        gold_pct_text.set_text('0%')
        silver_pct_text.set_text('0%')
        return [line_gold, line_silver, date_text, invested_text, gold_label, silver_label,
                gold_box, silver_box, gold_pct_text, silver_pct_text]

    def animate(frame):
        # Calculate progress (clamp to animation_frames for pause at end)
        effective_frame = min(frame, animation_frames - 1)
        progress = (effective_frame + 1) / animation_frames
        n_show = max(1, int(progress * n_points))

        # Update lines
        x_show = x_data[:n_show]
        gold_show = gold_normalized.iloc[:n_show].values
        silver_show = silver_normalized.iloc[:n_show].values

        line_gold.set_data(x_show, gold_show)
        line_silver.set_data(x_show, silver_show)

        # Update date (year month format for fixed width)
        current_date = df['date'].iloc[n_show - 1]
        date_text.set_text(current_date.strftime('%Y %B'))

        # Update value labels at line ends (show portfolio value)
        x_end = x_show[-1]
        gold_end = gold_show[-1]
        silver_end = silver_show[-1]

        # Format values in lakhs/crores
        def fmt_val(v):
            if v >= 10000000:
                return f'Rs.{v/10000000:.2f}Cr'
            else:
                return f'Rs.{v/100000:.1f}L'

        offset = len(df) * 0.02
        gold_label.set_position((x_end + offset, gold_end))
        gold_label.set_text(fmt_val(gold_end))

        silver_label.set_position((x_end + offset, silver_end))
        silver_label.set_text(fmt_val(silver_end))

        # Calculate total invested and % returns
        invested = invested_series.iloc[n_show - 1]
        gold_pct = ((gold_end - invested) / invested) * 100
        silver_pct = ((silver_end - invested) / invested) * 100

        # Update total invested text
        invested_text.set_text(f'Total Invested: {fmt_val(invested)}')

        # Update boxes with name and value
        gold_box.set_text(f'GOLD\n{fmt_val(gold_end)}')
        silver_box.set_text(f'SILVER\n{fmt_val(silver_end)}')

        # Update percentage labels (smaller text below boxes)
        gold_pct_text.set_text(f'{gold_pct:+.0f}%')
        silver_pct_text.set_text(f'{silver_pct:+.0f}%')

        return [line_gold, line_silver, date_text, invested_text, gold_label, silver_label,
                gold_box, silver_box, gold_pct_text, silver_pct_text]

    print(f"Creating animation with {total_frames} frames...")

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000/fps, blit=True
    )

    # Save video
    print(f"Saving video to: {output_path}")
    writer = FFMpegWriter(fps=fps, metadata={'title': 'Gold vs Silver'}, bitrate=8000)
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
    df.to_csv('gold_vs_silver_data.csv', index=False)
    print(f"Data saved to: gold_vs_silver_data.csv")

    # Create video
    video_path = create_reel_video(
        df=df,
        output_path="gold_vs_silver_reel.mp4",
        fps=30,
        duration_seconds=19
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
