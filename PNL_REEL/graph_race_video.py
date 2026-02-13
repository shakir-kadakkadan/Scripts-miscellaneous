#!/usr/bin/env python3
"""
Animated Line Graph Race Video Generator

Creates an animated video showing multiple line graphs "racing" across time.
Lines progressively draw themselves, creating a racing effect.

Usage:
    python graph_race_video.py --input data.csv --output race_video.mp4

Data format (CSV):
    time,Series1,Series2,Series3
    0,10,15,12
    1,20,18,22
    2,35,25,30
    ...

The first column should be the x-axis (time/index), remaining columns are series to compare.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
import sys


def create_sample_data(output_path: str = "sample_data.csv"):
    """Generate sample data for demonstration."""
    np.random.seed(42)
    time_points = 100

    # Create sample time series with different growth patterns
    time = np.arange(time_points)

    data = {
        'time': time,
        'Product A': np.cumsum(np.random.randn(time_points) * 2 + 1.5),
        'Product B': np.cumsum(np.random.randn(time_points) * 2 + 1.2),
        'Product C': np.cumsum(np.random.randn(time_points) * 2 + 1.0),
        'Product D': np.cumsum(np.random.randn(time_points) * 2 + 0.8),
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sample data created: {output_path}")
    return df


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    df = pd.read_csv(filepath)
    return df


def create_race_animation(
    df: pd.DataFrame,
    output_path: str = "race_video.mp4",
    fps: int = 30,
    duration_seconds: int = 10,
    title: str = "Line Graph Race",
    xlabel: str = "Time",
    ylabel: str = "Value",
    figsize: tuple = (12, 7),
    dpi: int = 150,
    show_legend: bool = True,
    show_labels: bool = True,
    color_palette: list = None
):
    """
    Create an animated line graph race video.

    Args:
        df: DataFrame with first column as x-axis, rest as series
        output_path: Output video file path
        fps: Frames per second
        duration_seconds: Total video duration
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        dpi: Resolution
        show_legend: Whether to show legend
        show_labels: Whether to show value labels at line ends
        color_palette: Custom colors for lines
    """

    # Extract x-axis and series data
    x_col = df.columns[0]
    series_cols = df.columns[1:]
    x_data = df[x_col].values

    # Calculate total frames and points per frame
    total_frames = fps * duration_seconds
    n_points = len(x_data)

    # Setup the figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # Color palette
    if color_palette is None:
        color_palette = [
            '#e94560', '#0f4c75', '#3fc1c9', '#f9ed69',
            '#f38181', '#aa96da', '#fcbad3', '#a8d8ea',
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'
        ]

    # Create line objects
    lines = []
    labels = []
    for i, col in enumerate(series_cols):
        color = color_palette[i % len(color_palette)]
        line, = ax.plot([], [], lw=2.5, color=color, label=col)
        lines.append(line)

        if show_labels:
            label = ax.text(0, 0, '', fontsize=10, fontweight='bold',
                          color=color, va='center')
            labels.append(label)

    # Set axis limits with padding
    x_min, x_max = x_data.min(), x_data.max()
    y_min = df[series_cols].min().min()
    y_max = df[series_cols].max().max()
    y_padding = (y_max - y_min) * 0.1
    x_padding = (x_max - x_min) * 0.15  # Extra padding for labels

    ax.set_xlim(x_min, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Style the axes
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_xlabel(xlabel, fontsize=12, color='white')
    ax.set_ylabel(ylabel, fontsize=12, color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, color='white')

    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)

    if show_legend:
        legend = ax.legend(loc='upper left', facecolor='#16213e',
                          edgecolor='white', labelcolor='white')

    # Progress text
    progress_text = ax.text(0.98, 0.02, '', transform=ax.transAxes,
                           fontsize=10, color='white', ha='right', va='bottom',
                           alpha=0.7)

    def init():
        """Initialize animation."""
        for line in lines:
            line.set_data([], [])
        for label in labels:
            label.set_text('')
        progress_text.set_text('')
        return lines + labels + [progress_text]

    def animate(frame):
        """Update animation for each frame."""
        # Calculate how many data points to show
        progress = (frame + 1) / total_frames
        n_show = max(1, int(progress * n_points))

        for i, (line, col) in enumerate(zip(lines, series_cols)):
            x_show = x_data[:n_show]
            y_show = df[col].values[:n_show]
            line.set_data(x_show, y_show)

            if show_labels and len(x_show) > 0:
                labels[i].set_position((x_show[-1] + (x_max - x_min) * 0.02, y_show[-1]))
                labels[i].set_text(f'{col}: {y_show[-1]:.1f}')

        progress_text.set_text(f'{progress*100:.0f}%')

        return lines + labels + [progress_text]

    print(f"Creating animation with {total_frames} frames...")
    print(f"Data points: {n_points}, Series: {len(series_cols)}")

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000/fps, blit=True
    )

    # Save video
    print(f"Saving video to: {output_path}")

    if output_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, metadata={'title': title}, bitrate=5000)

    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Video saved successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create an animated line graph race video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data
  python graph_race_video.py --generate-sample

  # Create video from CSV
  python graph_race_video.py --input data.csv --output race.mp4

  # Customize video
  python graph_race_video.py --input data.csv --output race.mp4 \\
      --title "Sales Comparison" --duration 15 --fps 60
        """
    )

    parser.add_argument('--input', '-i', type=str, help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, default='race_video.mp4',
                       help='Output video file path (default: race_video.mp4)')
    parser.add_argument('--generate-sample', action='store_true',
                       help='Generate sample data CSV')
    parser.add_argument('--title', type=str, default='Line Graph Race',
                       help='Video title')
    parser.add_argument('--xlabel', type=str, default='Time',
                       help='X-axis label')
    parser.add_argument('--ylabel', type=str, default='Value',
                       help='Y-axis label')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Video duration in seconds (default: 10)')
    parser.add_argument('--width', type=int, default=12,
                       help='Figure width in inches (default: 12)')
    parser.add_argument('--height', type=int, default=7,
                       help='Figure height in inches (default: 7)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Resolution DPI (default: 150)')
    parser.add_argument('--no-legend', action='store_true',
                       help='Hide legend')
    parser.add_argument('--no-labels', action='store_true',
                       help='Hide value labels at line ends')

    args = parser.parse_args()

    if args.generate_sample:
        df = create_sample_data()
        print("\nYou can now run:")
        print(f"  python graph_race_video.py --input sample_data.csv --output race.mp4")
        return

    if not args.input:
        # If no input, generate sample and create video
        print("No input file specified. Using sample data...")
        df = create_sample_data()
        args.input = 'sample_data.csv'
    else:
        df = load_data(args.input)

    create_race_animation(
        df=df,
        output_path=args.output,
        fps=args.fps,
        duration_seconds=args.duration,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        figsize=(args.width, args.height),
        dpi=args.dpi,
        show_legend=not args.no_legend,
        show_labels=not args.no_labels
    )


if __name__ == '__main__':
    main()
