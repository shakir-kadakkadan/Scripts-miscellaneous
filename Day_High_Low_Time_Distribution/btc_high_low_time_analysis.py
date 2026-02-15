#!/usr/bin/env python3
"""
BTC High/Low Time Analysis
Downloads Binance BTCUSDT 1m data for 2025-2026 and analyzes when highs/lows occur each day.
"""

import os
import requests
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import matplotlib.pyplot as plt
from collections import Counter
import concurrent.futures
from tqdm import tqdm

# Configuration
BASE_URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m"
DATA_DIR = "btc_kline_data"
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 2, 12)  # Yesterday's data (today is Feb 13, 2026)

# Kline columns from Binance
KLINE_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
]


def download_day_data(date):
    """Download and extract data for a single day."""
    date_str = date.strftime('%Y-%m-%d')
    filename = f"BTCUSDT-1m-{date_str}.zip"
    url = f"{BASE_URL}/{filename}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                # Get the actual filename inside the zip
                zip_contents = z.namelist()
                if zip_contents:
                    csv_name = zip_contents[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
                        return date_str, df
        return date_str, None
    except Exception as e:
        print(f"Error downloading {date_str}: {e}")
        return date_str, None


def find_high_low_times(df):
    """Find the time (hour:minute) when high and low occurred."""
    if df is None or df.empty:
        return None

    try:
        # Convert open_time to datetime (it's in microseconds from Binance)
        df = df.copy()
        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        df = df.dropna(subset=['open_time'])

        # Filter reasonable timestamps (2020-2030 range in microseconds)
        min_ts = 1577836800000000  # 2020-01-01 in microseconds
        max_ts = 1893456000000000  # 2030-01-01 in microseconds
        df = df[(df['open_time'] >= min_ts) & (df['open_time'] <= max_ts)]

        if df.empty:
            return None

        df['datetime'] = pd.to_datetime(df['open_time'], unit='us')
        df['time_str'] = df['datetime'].dt.strftime('%H:%M')
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute

        # Find the row with highest high and lowest low
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')

        high_idx = df['high'].idxmax()
        low_idx = df['low'].idxmin()

        high_time = df.loc[high_idx, 'time_str']
        low_time = df.loc[low_idx, 'time_str']

        high_hour = df.loc[high_idx, 'hour']
        low_hour = df.loc[low_idx, 'hour']

        return {
            'high_time': high_time,
            'high_hour': high_hour,
            'low_time': low_time,
            'low_hour': low_hour
        }
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


def download_all_data():
    """Download data for all dates in range."""
    dates = []
    current = START_DATE
    while current <= END_DATE:
        dates.append(current)
        current += timedelta(days=1)

    print(f"Downloading {len(dates)} days of data...")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_day_data, d): d for d in dates}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(dates)):
            date_str, df = future.result()
            if df is not None:
                results[date_str] = df

    print(f"Successfully downloaded {len(results)} days")
    return results


def analyze_high_low_times(data_dict):
    """Analyze high/low times for all days."""
    analysis = []

    for date_str, df in sorted(data_dict.items()):
        result = find_high_low_times(df)
        if result:
            analysis.append({
                'date': date_str,
                **result
            })
        else:
            print(f"No result for {date_str}")

    print(f"Total analyzed days: {len(analysis)}")
    return pd.DataFrame(analysis)


def create_visualizations(df):
    """Create 3 graphs: High times, Low times, and combined."""

    # Get date range from data
    date_min = df['date'].min()
    date_max = df['date'].max()
    date_range_str = f"{date_min} to {date_max}"

    # Count occurrences by hour
    high_hour_counts = Counter(df['high_hour'])
    low_hour_counts = Counter(df['low_hour'])

    hours = list(range(24))
    high_counts = [high_hour_counts.get(h, 0) for h in hours]
    low_counts = [low_hour_counts.get(h, 0) for h in hours]

    # Calculate percentages
    total_days = len(df)
    high_pct = [c / total_days * 100 for c in high_counts]
    low_pct = [c / total_days * 100 for c in low_counts]

    # Set style
    plt.style.use('dark_background')

    # Graph 1: High Times Distribution
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    bars1 = ax1.bar(hours, high_counts, color='#00ff88', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Hour (UTC)', fontsize=14)
    ax1.set_ylabel('Number of Days', fontsize=14)
    ax1.set_title(f'BTC Daily HIGH Time Distribution\n({date_range_str}, {total_days} days analyzed)\nSource: Binance Vision (data not stored locally)', fontsize=14, fontweight='bold')
    ax1.set_xticks(hours)
    ax1.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for bar, pct in zip(bars1, high_pct):
        if pct > 0:
            ax1.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9, color='white')

    # Highlight top 3 hours
    top3_high = sorted(range(24), key=lambda x: high_counts[x], reverse=True)[:3]
    for h in top3_high:
        bars1[h].set_color('#ff4444')
        bars1[h].set_edgecolor('#ffffff')

    ax1.legend([plt.Rectangle((0,0),1,1, color='#ff4444'), plt.Rectangle((0,0),1,1, color='#00ff88')],
              [f'Top 3 Hours: {top3_high}', 'Other Hours'], loc='upper right')

    plt.tight_layout()
    plt.savefig('btc_high_time_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: btc_high_time_distribution.png")

    # Graph 2: Low Times Distribution
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    bars2 = ax2.bar(hours, low_counts, color='#ff6b6b', edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Hour (UTC)', fontsize=14)
    ax2.set_ylabel('Number of Days', fontsize=14)
    ax2.set_title(f'BTC Daily LOW Time Distribution\n({date_range_str}, {total_days} days analyzed)\nSource: Binance Vision (data not stored locally)', fontsize=14, fontweight='bold')
    ax2.set_xticks(hours)
    ax2.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for bar, pct in zip(bars2, low_pct):
        if pct > 0:
            ax2.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9, color='white')

    # Highlight top 3 hours
    top3_low = sorted(range(24), key=lambda x: low_counts[x], reverse=True)[:3]
    for h in top3_low:
        bars2[h].set_color('#00ccff')
        bars2[h].set_edgecolor('#ffffff')

    ax2.legend([plt.Rectangle((0,0),1,1, color='#00ccff'), plt.Rectangle((0,0),1,1, color='#ff6b6b')],
              [f'Top 3 Hours: {top3_low}', 'Other Hours'], loc='upper right')

    plt.tight_layout()
    plt.savefig('btc_low_time_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: btc_low_time_distribution.png")

    # Graph 3: Combined High & Low
    fig3, ax3 = plt.subplots(figsize=(16, 9))

    bar_width = 0.35
    x = np.array(hours)

    bars_high = ax3.bar(x - bar_width/2, high_counts, bar_width, label='Daily HIGH', color='#00ff88', edgecolor='white', alpha=0.85)
    bars_low = ax3.bar(x + bar_width/2, low_counts, bar_width, label='Daily LOW', color='#ff6b6b', edgecolor='white', alpha=0.85)

    ax3.set_xlabel('Hour (UTC)', fontsize=14)
    ax3.set_ylabel('Number of Days', fontsize=14)
    ax3.set_title(f'BTC Daily HIGH & LOW Time Distribution\n({date_range_str}, {total_days} days analyzed)\nSource: Binance Vision (data not stored locally)', fontsize=14, fontweight='bold')
    ax3.set_xticks(hours)
    ax3.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45, ha='right')
    ax3.legend(loc='upper right', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('btc_high_low_combined_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: btc_high_low_combined_distribution.png")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nTotal days analyzed: {total_days}")

    print("\n--- TOP 5 HOURS FOR DAILY HIGH ---")
    for i, h in enumerate(sorted(range(24), key=lambda x: high_counts[x], reverse=True)[:5]):
        print(f"  {i+1}. {h:02d}:00 UTC - {high_counts[h]} days ({high_counts[h]/total_days*100:.1f}%)")

    print("\n--- TOP 5 HOURS FOR DAILY LOW ---")
    for i, h in enumerate(sorted(range(24), key=lambda x: low_counts[x], reverse=True)[:5]):
        print(f"  {i+1}. {h:02d}:00 UTC - {low_counts[h]} days ({low_counts[h]/total_days*100:.1f}%)")

    # Time ranges analysis
    print("\n--- TIME RANGE ANALYSIS ---")
    ranges = [
        ("00:00-06:00", range(0, 6)),
        ("06:00-12:00", range(6, 12)),
        ("12:00-18:00", range(12, 18)),
        ("18:00-24:00", range(18, 24)),
    ]

    print("\nHIGH occurs in:")
    for name, r in ranges:
        count = sum(high_counts[h] for h in r)
        print(f"  {name}: {count} days ({count/total_days*100:.1f}%)")

    print("\nLOW occurs in:")
    for name, r in ranges:
        count = sum(low_counts[h] for h in r)
        print(f"  {name}: {count} days ({count/total_days*100:.1f}%)")


def main():
    print("="*60)
    print("BTC HIGH/LOW TIME ANALYSIS")
    print("Analyzing BTCUSDT 1m data from Binance (2025-2026)")
    print("="*60)

    # Download data
    data = download_all_data()

    if not data:
        print("No data downloaded. Exiting.")
        return

    # Analyze high/low times
    print("\nAnalyzing high/low times for each day...")
    analysis_df = analyze_high_low_times(data)

    # Save analysis to CSV
    analysis_df.to_csv('btc_high_low_analysis.csv', index=False)
    print(f"Saved analysis to: btc_high_low_analysis.csv")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(analysis_df)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
