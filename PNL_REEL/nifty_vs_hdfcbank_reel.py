#!/usr/bin/env python3
"""
NIFTY vs HDFCBANK - Instagram Reel Video Generator
Creates a 9:16 aspect ratio animated comparison video.
Uses Yahoo Finance for dividend-adjusted prices.
SIP increases by ₹500 each year.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import requests
from io import BytesIO
import subprocess
import os


def fetch_nifty_data():
    """Fetch NIFTY 50 data from Yahoo Finance (adjusted for dividends)."""
    print("Fetching NIFTY data from Yahoo Finance...")
    ticker = yf.Ticker("^NSEI")  # NIFTY 50 index
    df = ticker.history(period="max")

    df = df.reset_index()
    df['date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df['nifty'] = df['Close']
    df['year_month'] = df['date'].dt.to_period('M')

    # Aggregate to monthly (last value of each month)
    monthly = df.groupby('year_month').agg({
        'date': 'last',
        'nifty': 'last'
    }).reset_index(drop=True)

    return monthly


def fetch_hdfcbank_data():
    """Fetch HDFCBANK data from Yahoo Finance (adjusted for dividends, splits, bonuses)."""
    print("Fetching HDFCBANK data from Yahoo Finance...")
    ticker = yf.Ticker("HDFCBANK.NS")  # HDFC Bank on NSE
    df = ticker.history(period="max")

    df = df.reset_index()
    df['date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df['hdfcbank'] = df['Close']  # Yahoo Finance 'Close' is adjusted
    df['year_month'] = df['date'].dt.to_period('M')

    # Aggregate to monthly (last value of each month)
    monthly = df.groupby('year_month').agg({
        'date': 'last',
        'hdfcbank': 'last'
    }).reset_index(drop=True)

    return monthly


def prepare_data():
    """Fetch and merge NIFTY and HDFCBANK data."""
    nifty_df = fetch_nifty_data()
    hdfcbank_df = fetch_hdfcbank_data()

    # Merge on year_month
    nifty_df['year_month'] = nifty_df['date'].dt.to_period('M')
    hdfcbank_df['year_month'] = hdfcbank_df['date'].dt.to_period('M')

    merged = pd.merge(
        nifty_df[['year_month', 'nifty']],
        hdfcbank_df[['year_month', 'hdfcbank']],
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
    output_path: str = "nifty_vs_hdfcbank_reel.mp4",
    fps: int = 30,
    duration_seconds: int = 16
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

    # SIP configuration - starts at ₹4000, increases ₹400 each year
    base_sip = 4000
    yearly_increase = 400
    start_year = df['date'].iloc[0].year

    # Calculate SIP amount for each month
    def get_sip_amount(date):
        years_elapsed = date.year - start_year
        return base_sip + (years_elapsed * yearly_increase)

    # Calculate SIP portfolio value for each point in time
    nifty_sip = []
    hdfcbank_sip = []
    total_invested = []

    for i in range(len(df)):
        # Portfolio value at time i = sum of all previous investments grown to time i
        nifty_value = 0
        hdfcbank_value = 0
        invested = 0
        for j in range(i + 1):
            # Get SIP amount for that month
            sip_amount = get_sip_amount(df['date'].iloc[j])
            invested += sip_amount

            # Each SIP invested at time j, grown to time i
            nifty_growth = df['nifty'].iloc[i] / df['nifty'].iloc[j]
            hdfcbank_growth = df['hdfcbank'].iloc[i] / df['hdfcbank'].iloc[j]
            nifty_value += sip_amount * nifty_growth
            hdfcbank_value += sip_amount * hdfcbank_growth

        nifty_sip.append(nifty_value)
        hdfcbank_sip.append(hdfcbank_value)
        total_invested.append(invested)

    nifty_normalized = pd.Series(nifty_sip)
    hdfcbank_normalized = pd.Series(hdfcbank_sip)
    invested_series = pd.Series(total_invested)

    # Setup figure with dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # Text scale multiplier (change this to resize all text)
    text_scale = 1.5

    # Colors
    nifty_color = '#00d4aa'  # Teal/green for NIFTY
    hdfcbank_color = '#ed1c24'  # HDFC Bank red

    # Create line objects
    line_nifty, = ax.plot([], [], lw=3, color=nifty_color, label='NIFTY 50')
    line_hdfcbank, = ax.plot([], [], lw=3, color=hdfcbank_color, label='HDFCBANK')

    # Set axis limits
    x_data = np.arange(len(df))
    y_min = 0  # Start from 0 so graph is visible from beginning
    y_max = max(nifty_normalized.max(), hdfcbank_normalized.max()) * 1.1

    ax.set_xlim(0, len(df) * 1.15)
    ax.set_ylim(y_min, y_max)

    # Subtitle with SIP info
    start_year = df['date'].iloc[0].year
    ax.text(0.5, 1.075, f'₹4K Monthly SIP +₹400/yr since {start_year}',
            transform=ax.transAxes,
            fontsize=17 * text_scale, color='#cccccc', fontweight='bold',
            ha='center', va='top')

    # Current SIP display - below title
    sip_text = ax.text(0.5, 1.025, '', transform=ax.transAxes,
                      fontsize=12 * text_scale, color='white',
                      ha='center', va='top', fontweight='bold')

    # Value boxes with names inside (moved further apart)
    nifty_box = ax.text(0.08, 0.93, '', transform=ax.transAxes,
                       fontsize=17 * text_scale, color=nifty_color, ha='center',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                                edgecolor=nifty_color, linewidth=2))

    hdfcbank_box = ax.text(0.92, 0.93, '', transform=ax.transAxes,
                          fontsize=14 * text_scale, color=hdfcbank_color, ha='center',
                          fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                                   edgecolor=hdfcbank_color, linewidth=2))

    # Percentage labels (smaller text below boxes)
    nifty_pct_text = ax.text(0.08, 0.86, '', transform=ax.transAxes,
                            fontsize=13 * text_scale, color=nifty_color, ha='center',
                            fontweight='bold')
    hdfcbank_pct_text = ax.text(0.92, 0.86, '', transform=ax.transAxes,
                               fontsize=13 * text_scale, color=hdfcbank_color, ha='center',
                               fontweight='bold')

    # Total invested display - above date
    invested_text = ax.text(0.32, 0.95, '', transform=ax.transAxes,
                           fontsize=12 * text_scale, color='#aaaaaa',
                           ha='left', va='center', fontweight='bold')

    # Date display - left aligned after NIFTY box
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

    ax.text(0.44, -0.08, '@algo_vs_discretionary_trader',
            transform=ax.transAxes,
            fontsize=12 * text_scale, color='#888888',
            ha='left', va='center', fontweight='bold')

    # Value labels at line ends
    nifty_label = ax.text(0, 0, '', fontsize=14 * text_scale, fontweight='bold',
                         color=nifty_color, va='center')
    hdfcbank_label = ax.text(0, 0, '', fontsize=14 * text_scale, fontweight='bold',
                            color=hdfcbank_color, va='center')

    # Format y-axis to show lakhs/crores
    def format_lakhs(x, _):
        if x == 0:
            return '₹0'
        elif x >= 10000000:  # 1 Crore = 100 Lakhs
            return f'₹{x/10000000:.1f}Cr'
        elif x >= 100000:
            return f'₹{x/100000:.1f}L'
        else:
            return f'₹{x/1000:.0f}K'
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
        line_nifty.set_data([], [])
        line_hdfcbank.set_data([], [])
        date_text.set_text('')
        invested_text.set_text('')
        sip_text.set_text('')
        nifty_label.set_text('')
        hdfcbank_label.set_text('')
        nifty_box.set_text('NIFTY 50\n₹0')
        hdfcbank_box.set_text('HDFCBANK\n₹0')
        nifty_pct_text.set_text('0%')
        hdfcbank_pct_text.set_text('0%')
        return [line_nifty, line_hdfcbank, date_text, invested_text, sip_text, nifty_label, hdfcbank_label,
                nifty_box, hdfcbank_box, nifty_pct_text, hdfcbank_pct_text]

    def animate(frame):
        # Calculate progress (clamp to animation_frames for pause at end)
        effective_frame = min(frame, animation_frames - 1)
        progress = (effective_frame + 1) / animation_frames
        n_show = max(1, int(progress * n_points))

        # Update lines
        x_show = x_data[:n_show]
        nifty_show = nifty_normalized.iloc[:n_show].values
        hdfcbank_show = hdfcbank_normalized.iloc[:n_show].values

        line_nifty.set_data(x_show, nifty_show)
        line_hdfcbank.set_data(x_show, hdfcbank_show)

        # Update date (year month format for fixed width)
        current_date = df['date'].iloc[n_show - 1]
        date_text.set_text(current_date.strftime('%Y %B'))

        # Update current SIP amount
        current_sip = get_sip_amount(current_date)
        sip_text.set_text(f'Monthly SIP: ₹{current_sip:,}')

        # Update value labels at line ends (show portfolio value)
        x_end = x_show[-1]
        nifty_end = nifty_show[-1]
        hdfcbank_end = hdfcbank_show[-1]

        # Format values in lakhs/crores
        def fmt_val(v):
            if v >= 10000000:
                return f'₹{v/10000000:.2f}Cr'
            elif v >= 100000:
                return f'₹{v/100000:.1f}L'
            else:
                return f'₹{v/1000:.1f}K'

        offset = len(df) * 0.02
        nifty_label.set_position((x_end + offset, nifty_end))
        nifty_label.set_text(fmt_val(nifty_end))

        hdfcbank_label.set_position((x_end + offset, hdfcbank_end))
        hdfcbank_label.set_text(fmt_val(hdfcbank_end))

        # Calculate total invested and % returns
        invested = invested_series.iloc[n_show - 1]
        nifty_pct = ((nifty_end - invested) / invested) * 100
        hdfcbank_pct = ((hdfcbank_end - invested) / invested) * 100

        # Update total invested text
        invested_text.set_text(f'Total Invested: {fmt_val(invested)}')

        # Update boxes with name and value
        nifty_box.set_text(f'NIFTY 50\n{fmt_val(nifty_end)}')
        hdfcbank_box.set_text(f'HDFCBANK\n{fmt_val(hdfcbank_end)}')

        # Update percentage labels (smaller text below boxes)
        nifty_pct_text.set_text(f'{nifty_pct:+.0f}%')
        hdfcbank_pct_text.set_text(f'{hdfcbank_pct:+.0f}%')

        return [line_nifty, line_hdfcbank, date_text, invested_text, sip_text, nifty_label, hdfcbank_label,
                nifty_box, hdfcbank_box, nifty_pct_text, hdfcbank_pct_text]

    print(f"Creating animation with {total_frames} frames...")

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000/fps, blit=True
    )

    # Save video
    print(f"Saving video to: {output_path}")
    writer = FFMpegWriter(fps=fps, metadata={'title': 'NIFTY vs HDFCBANK'}, bitrate=8000)
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
    df.to_csv('nifty_vs_hdfcbank_data.csv', index=False)
    print(f"Data saved to: nifty_vs_hdfcbank_data.csv")

    # Create video
    video_path = create_reel_video(
        df=df,
        output_path="nifty_vs_hdfcbank_reel.mp4",
        fps=30,
        duration_seconds=16
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
