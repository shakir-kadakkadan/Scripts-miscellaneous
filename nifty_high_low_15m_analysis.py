#!/usr/bin/env python3
"""
NIFTY High/Low Time Analysis (15m candles)
Processes 1m NIFTY data from zip files, converts to 15m candles, 
and analyzes when highs/lows occur each day.
"""

import os
import json
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# Configuration
DATA_DIR = "/Users/shakir/BhavAppData/DATA/fyers_index"
ZIP_FILES = [
    "fyers_index_all.zip",
    "fyers_index_all_2.zip"
]

def extract_1m_nifty_data_from_zip(zip_path):
    """Extract all 1m NIFTY data from a single zip file."""
    all_data = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Get all files that match pattern: 1m/NIFTY_*.json
            nifty_files = [f for f in z.namelist() if f.startswith('1m/NIFTY_') and f.endswith('.json')]
            
            print(f"Found {len(nifty_files)} NIFTY 1m files in {os.path.basename(zip_path)}")
            
            for file_path in tqdm(nifty_files, desc=f"Processing {os.path.basename(zip_path)}"):
                try:
                    with z.open(file_path) as f:
                        content = f.read()
                        data = json.loads(content)
                        
                        # Extract candles array from JSON
                        if 'candles' in data and isinstance(data['candles'], list):
                            candles = data['candles']
                            
                            # Convert to DataFrame
                            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            
                            # Extract date from filename (e.g., NIFTY_2025_01_07.json)
                            filename = os.path.basename(file_path)
                            date_str = filename.replace('NIFTY_', '').replace('.json', '').replace('_', '-')
                            df['date'] = date_str
                            
                            all_data.append(df)
                            
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    except Exception as e:
        print(f"Error opening zip {zip_path}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_all_nifty_data():
    """Load all NIFTY 1m data from all zip files."""
    all_data = []
    
    for zip_file in ZIP_FILES:
        zip_path = os.path.join(DATA_DIR, zip_file)
        if os.path.exists(zip_path):
            print(f"\nProcessing: {zip_file}")
            df = extract_1m_nifty_data_from_zip(zip_path)
            if not df.empty:
                all_data.append(df)
        else:
            print(f"Warning: {zip_path} not found")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal 1m candles loaded: {len(combined)}")
        return combined
    
    return pd.DataFrame()


def convert_to_15m_candles(df_1m):
    """Convert 1m candles to 15m candles."""
    if df_1m.empty:
        return pd.DataFrame()
    
    # Convert timestamp to datetime
    df_1m['datetime'] = pd.to_datetime(df_1m['timestamp'], unit='s')
    
    # Sort by datetime
    df_1m = df_1m.sort_values('datetime')
    
    # Group by date and 15-minute intervals
    df_1m['date'] = df_1m['datetime'].dt.date
    df_1m['time_15m'] = df_1m['datetime'].dt.floor('15min')
    
    # Aggregate to 15m candles
    df_15m = df_1m.groupby(['date', 'time_15m']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    print(f"Converted to {len(df_15m)} 15m candles")
    return df_15m


def find_high_low_times_15m(df_15m):
    """Find the 15m time slot when high and low occurred for each day."""
    if df_15m.empty:
        return pd.DataFrame()
    
    results = []
    
    # Group by date
    for date, day_data in df_15m.groupby('date'):
        try:
            # Find the 15m slot with highest high
            high_idx = day_data['high'].idxmax()
            high_time = day_data.loc[high_idx, 'time_15m']
            
            # Find the 15m slot with lowest low
            low_idx = day_data['low'].idxmin()
            low_time = day_data.loc[low_idx, 'time_15m']
            
            # Extract hour and minute
            high_hour = high_time.hour
            high_minute = high_time.minute
            low_hour = low_time.hour
            low_minute = low_time.minute
            
            # Create time string (HH:MM)
            high_time_str = f"{high_hour:02d}:{high_minute:02d}"
            low_time_str = f"{low_hour:02d}:{low_minute:02d}"
            
            # Create 15m slot identifier (0-95, representing 96 15-minute slots in a day)
            high_slot = high_hour * 4 + high_minute // 15
            low_slot = low_hour * 4 + low_minute // 15
            
            results.append({
                'date': str(date),
                'high_time': high_time_str,
                'high_hour': high_hour,
                'high_slot': high_slot,
                'low_time': low_time_str,
                'low_hour': low_hour,
                'low_slot': low_slot
            })
            
        except Exception as e:
            print(f"Error processing date {date}: {e}")
    
    return pd.DataFrame(results)


def create_visualizations(df_analysis):
    """Create visualizations for high/low time distribution."""
    if df_analysis.empty:
        print("No data to visualize")
        return
    
    # Get date range
    date_min = df_analysis['date'].min()
    date_max = df_analysis['date'].max()
    date_range_str = f"{date_min} to {date_max}"
    total_days = len(df_analysis)
    
    # Count occurrences by 15m slot
    high_slot_counts = Counter(df_analysis['high_slot'])
    low_slot_counts = Counter(df_analysis['low_slot'])
    
    # Create slot labels (every 15 minutes) - Convert to IST (UTC + 5:30)
    # Slot 0 = 00:00 UTC = 05:30 IST
    def utc_slot_to_ist_label(slot):
        utc_hour = slot // 4
        utc_minute = (slot % 4) * 15
        # Add 5 hours 30 minutes
        total_minutes = utc_hour * 60 + utc_minute + 330  # 330 = 5*60 + 30
        ist_hour = (total_minutes // 60) % 24
        ist_minute = total_minutes % 60
        return f"{ist_hour:02d}:{ist_minute:02d}", ist_hour, ist_minute
    
    slots = list(range(96))  # 96 15-minute slots in 24 hours
    slot_labels_ist = [utc_slot_to_ist_label(s)[0] for s in slots]
    
    # Filter to trading hours: 09:15 to 15:15 IST (exclude 9:00 and 3:30-4:00)
    # 09:15 IST = 03:45 UTC = slot 15
    # 15:15 IST = 09:45 UTC = slot 39
    trading_start_slot = 15  # 03:45 UTC = 09:15 IST
    trading_end_slot = 39    # 09:45 UTC = 15:15 IST
    
    trading_slots = list(range(trading_start_slot, trading_end_slot + 1))
    trading_labels = [slot_labels_ist[s] for s in trading_slots]
    
    high_counts = [high_slot_counts.get(s, 0) for s in slots]
    low_counts = [low_slot_counts.get(s, 0) for s in slots]
    
    # Filter counts for trading hours
    high_counts_trading = [high_slot_counts.get(s, 0) for s in trading_slots]
    low_counts_trading = [low_slot_counts.get(s, 0) for s in trading_slots]
    
    # Calculate percentages
    high_pct = [c / total_days * 100 for c in high_counts]
    low_pct = [c / total_days * 100 for c in low_counts]
    high_pct_trading = [c / total_days * 100 for c in high_counts_trading]
    low_pct_trading = [c / total_days * 100 for c in low_counts_trading]
    
    # Set style
    plt.style.use('dark_background')
    
    # Graph 1: High Times Distribution (Trading Hours Only - IST)
    fig1, ax1 = plt.subplots(figsize=(16, 12))  # 4:3 aspect ratio
    x_positions = list(range(len(trading_slots)))
    bars1 = ax1.bar(x_positions, high_counts_trading, color='#00ff88', edgecolor='white', alpha=0.8, width=0.8)
    ax1.set_xlabel('Time IST (GMT+5:30) - 15-minute intervals', fontsize=14)
    ax1.set_ylabel('Number of Days', fontsize=14)
    ax1.set_title(f'NIFTY Daily HIGH Time Distribution (15m candles) - Trading Hours 09:15-15:15 IST\\n({date_range_str}, {total_days} days analyzed)', 
                  fontsize=14, fontweight='bold')
    
    # Show time label on every bar
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(trading_labels, rotation=90, ha='center', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Highlight top 5 slots within trading hours
    top5_high_trading = sorted(range(len(trading_slots)), key=lambda x: high_counts_trading[x], reverse=True)[:5]
    for idx in top5_high_trading:
        bars1[idx].set_color('#ff4444')
        bars1[idx].set_edgecolor('#ffffff')
        # Add percentage label
        if high_pct_trading[idx] > 0:
            ax1.annotate(f'{high_pct_trading[idx]:.1f}%',
                        xy=(idx, high_counts_trading[idx]),
                        ha='center', va='bottom', fontsize=8, color='white')
    
    top5_times = [trading_labels[idx] for idx in top5_high_trading]
    ax1.legend([plt.Rectangle((0,0),1,1, color='#ff4444'), plt.Rectangle((0,0),1,1, color='#00ff88')],
              [f'Top 5 Times (IST): {", ".join(top5_times)}', 'Other Times'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('nifty_high_time_distribution_15m.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: nifty_high_time_distribution_15m.png")
    
    # Graph 2: Low Times Distribution (Trading Hours Only - IST)
    fig2, ax2 = plt.subplots(figsize=(16, 12))  # 4:3 aspect ratio
    bars2 = ax2.bar(x_positions, low_counts_trading, color='#ff6b6b', edgecolor='white', alpha=0.8, width=0.8)
    ax2.set_xlabel('Time IST (GMT+5:30) - 15-minute intervals', fontsize=14)
    ax2.set_ylabel('Number of Days', fontsize=14)
    ax2.set_title(f'NIFTY Daily LOW Time Distribution (15m candles) - Trading Hours 09:15-15:15 IST\\n({date_range_str}, {total_days} days analyzed)', 
                  fontsize=14, fontweight='bold')
    
    # Show time label on every bar
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(trading_labels, rotation=90, ha='center', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Highlight top 5 slots within trading hours
    top5_low_trading = sorted(range(len(trading_slots)), key=lambda x: low_counts_trading[x], reverse=True)[:5]
    for idx in top5_low_trading:
        bars2[idx].set_color('#00ccff')
        bars2[idx].set_edgecolor('#ffffff')
        # Add percentage label
        if low_pct_trading[idx] > 0:
            ax2.annotate(f'{low_pct_trading[idx]:.1f}%',
                        xy=(idx, low_counts_trading[idx]),
                        ha='center', va='bottom', fontsize=8, color='white')
    
    top5_times = [trading_labels[idx] for idx in top5_low_trading]
    ax2.legend([plt.Rectangle((0,0),1,1, color='#00ccff'), plt.Rectangle((0,0),1,1, color='#ff6b6b')],
              [f'Top 5 Times (IST): {", ".join(top5_times)}', 'Other Times'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('nifty_low_time_distribution_15m.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: nifty_low_time_distribution_15m.png")
    
    # Graph 3: Combined High & Low (Trading Hours Only - IST)
    fig3, ax3 = plt.subplots(figsize=(16, 12))  # 4:3 aspect ratio
    
    bar_width = 0.35
    x = np.array(x_positions)
    
    bars_high = ax3.bar(x - bar_width/2, high_counts_trading, bar_width, label='Daily HIGH', 
                        color='#00ff88', edgecolor='white', alpha=0.85)
    bars_low = ax3.bar(x + bar_width/2, low_counts_trading, bar_width, label='Daily LOW', 
                       color='#ff6b6b', edgecolor='white', alpha=0.85)
    
    ax3.set_xlabel('Time IST (GMT+5:30) - 15-minute intervals', fontsize=14)
    ax3.set_ylabel('Number of Days', fontsize=14)
    ax3.set_title(f'NIFTY Daily HIGH & LOW Time Distribution (15m candles) - Trading Hours 09:15-15:15 IST\\n({date_range_str}, {total_days} days analyzed)', 
                  fontsize=14, fontweight='bold')
    
    # Show time label on every bar
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(trading_labels, rotation=90, ha='center', fontsize=9)
    ax3.legend(loc='upper right', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nifty_high_low_combined_distribution_15m.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: nifty_high_low_combined_distribution_15m.png")
    
    # Print summary statistics
    print("\\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\\nTotal days analyzed: {total_days}")
    
    print("\n--- TOP 10 15-MINUTE SLOTS FOR DAILY HIGH ---")
    for i, s in enumerate(sorted(range(96), key=lambda x: high_counts[x], reverse=True)[:10]):
        print(f"  {i+1}. {slot_labels_ist[s]} - {high_counts[s]} days ({high_counts[s]/total_days*100:.1f}%)")
    
    print("\n--- TOP 10 15-MINUTE SLOTS FOR DAILY LOW ---")
    for i, s in enumerate(sorted(range(96), key=lambda x: low_counts[x], reverse=True)[:10]):
        print(f"  {i+1}. {slot_labels_ist[s]} - {low_counts[s]} days ({low_counts[s]/total_days*100:.1f}%)")
    
    # Time ranges analysis (by hour)
    print("\\n--- TIME RANGE ANALYSIS (by hour) ---")
    
    # Count by hour for high
    high_hour_counts = Counter(df_analysis['high_hour'])
    low_hour_counts = Counter(df_analysis['low_hour'])
    
    ranges = [
        ("00:00-06:00", range(0, 6)),
        ("06:00-12:00", range(6, 12)),
        ("12:00-18:00", range(12, 18)),
        ("18:00-24:00", range(18, 24)),
    ]
    
    print("\\nHIGH occurs in:")
    for name, r in ranges:
        count = sum(high_hour_counts.get(h, 0) for h in r)
        print(f"  {name}: {count} days ({count/total_days*100:.1f}%)")
    
    print("\\nLOW occurs in:")
    for name, r in ranges:
        count = sum(low_hour_counts.get(h, 0) for h in r)
        print(f"  {name}: {count} days ({count/total_days*100:.1f}%)")


def main():
    print("="*80)
    print("NIFTY HIGH/LOW TIME ANALYSIS (15m candles)")
    print("Processing 1m data from zip files and converting to 15m candles")
    print("="*80)
    
    # Load all 1m data
    print("\\nStep 1: Loading all NIFTY 1m data from zip files...")
    df_1m = load_all_nifty_data()
    
    if df_1m.empty:
        print("No data loaded. Exiting.")
        return
    
    # Convert to 15m candles
    print("\\nStep 2: Converting 1m candles to 15m candles...")
    df_15m = convert_to_15m_candles(df_1m)
    
    if df_15m.empty:
        print("No 15m candles created. Exiting.")
        return
    
    # Analyze high/low times
    print("\\nStep 3: Analyzing high/low times for each day...")
    df_analysis = find_high_low_times_15m(df_15m)
    
    if df_analysis.empty:
        print("No analysis results. Exiting.")
        return
    
    # Save analysis to CSV
    csv_file = 'nifty_high_low_analysis_15m.csv'
    df_analysis.to_csv(csv_file, index=False)
    print(f"\\nSaved analysis to: {csv_file}")
    
    # Create visualizations
    print("\\nStep 4: Creating visualizations...")
    create_visualizations(df_analysis)
    
    print("\\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
