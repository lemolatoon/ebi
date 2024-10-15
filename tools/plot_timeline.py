import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def draw_timeline(title, compression_methods, out_dir: str = "results", fontsize=10):
    """
    Draws a timeline of compression methods with adaptive label positioning.
    If there is a large gap, the earlier date is plotted closer but marked with a wave symbol.

    Parameters:
    - title (str): The title of the graph.
    - compression_methods (list of tuples): A list of (name, year) tuples.
    - out_dir (str): Directory to save the output image.
    """
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Group methods by year
    year_to_methods = defaultdict(list)
    for name, year in compression_methods:
        year_to_methods[year].append(name)

    # Detect large gap and adjust display years if necessary
    years = sorted(year_to_methods.keys())
    large_gap_threshold = 10
    replaced_year = 2008
    if years[1] - years[0] > large_gap_threshold:
        display_years = [replaced_year] + years[1:]
        wave_symbol_position = (replaced_year + years[1]) / 2  # Position for the wave symbol
    else:
        display_years = years

    # Set the figure size for a compact timeline
    fig, ax = plt.subplots(figsize=(8, 2))

    # Draw the horizontal timeline
    ax.hlines(1, display_years[0] - 1, display_years[-1] + 1, color='black', linewidth=1)

    # If there is a large gap, add double vertical wave symbols spanning the graph height
    if years[1] - years[0] > large_gap_threshold:
        d1 = 0.1  # Horizontal width of the wave
        d2 = 2.0  # Extended height to span beyond the graph height
        wn = 51    # Number of waves (must be odd)

        # Create the first wave path
        pp = (0, d1, 0, -d1)
        px = np.array([wave_symbol_position + pp[i % 4] for i in range(wn)])
        py = np.linspace(-0.5, 2.6, wn)  # Extended height
        wave_path1 = Path(list(zip(px, py)), [Path.MOVETO] + [Path.CURVE3] * (wn - 1))

        # Create the second wave path with a slight offset
        px2 = px + 0.1  # Shift slightly for the second wave
        wave_path2 = Path(list(zip(px2, py)), [Path.MOVETO] + [Path.CURVE3] * (wn - 1))

        # Add the first wave symbol as a patch to the axes
        wave_patch1 = PathPatch(
            wave_path1, lw=1, edgecolor='black', facecolor='none',
            clip_on=False, zorder=10
        )
        ax.add_patch(wave_patch1)

        # Add the second wave symbol as a patch to the axes
        wave_patch2 = PathPatch(
            wave_path2, lw=1, edgecolor='black', facecolor='none',
            clip_on=False, zorder=10
        )
        ax.add_patch(wave_patch2)

    # Plot a single point for each year and distribute labels with adaptive spacing
    for i, (year, methods) in enumerate(year_to_methods.items()):
        display_year = display_years[i]
        n = len(methods)
        offset_range = 0.4 if n < 4 else 0.6
        start_offset = -offset_range / 2
        step = offset_range / max(n - 1, 1)  # Step size for label distribution

        # Plot a single marker for the display year
        ax.scatter(display_year, 1, color='black', marker='o', zorder=3)

        # Add labels with horizontal offsets
        for j, name in enumerate(methods):
            x_offset = display_year + start_offset + j * step
            ax.text(x_offset, 1.1, name, ha='center', va='bottom', rotation=90, fontsize=fontsize, fontfamily='monospace')

    # Configure the axis and layout
    ax.set_xticks(display_years)
    ax.set_xticklabels([str(y) if y != replaced_year else '1977' for y in display_years], fontsize=fontsize)
    ax.set_yticks([])
    ax.set_xlim(display_years[0] - 1, display_years[-1] + 1)
    ax.set_ylim(-0.2, 2.4)  # Adjusted for extended height

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the graph
    plt.savefig(os.path.join(out_dir, "timeline.png"))
    plt.show()

# Example usage
compression_methods = [
    ("gzip", 1977),
    ("Snappy", 2011),
    ("Gorilla", 2015),
    ("Zstd", 2016),
    ("Sprintz", 2018),
    ("Buff", 2021),
    ("Chimp", 2022),
    ("Elf", 2023),
    ("ALP", 2023),
]

# Draw the timeline
draw_timeline("Timeline of studied compression methods", compression_methods)
