# utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle
from matplotlib.widgets import Slider


def show_visualization(X, filtration, barcode, initial_epsilon=0.0, margin=0.5):
    # Collect the barcode information
    # The barcode is computed by persistence_barcode before this visualization.
    max_dim = len(barcode)
    finite_deaths = [
        death
        for bars in barcode.values()
        for _, death in bars
        if np.isfinite(death)
    ]
    plot_max = max(finite_deaths, default=max(simplex.distance for simplex in filtration))
    initial_epsilon = np.clip(initial_epsilon, 0, plot_max)

    points = np.asarray(X, dtype=float)
    edges = [simplex for simplex in filtration if simplex.dim == 1]
    triangles = [simplex for simplex in filtration if simplex.dim == 2]

    fig = plt.figure(figsize=(10, 13))
    grid = fig.add_gridspec(
        2, 1,
        height_ratios=[5, 1],
        left=0.06, right=0.94, top=0.95, bottom=0.16,
        hspace=0.15,
    )
    ax_complex = fig.add_subplot(grid[0])
    ax_barcode = fig.add_subplot(grid[1])

    # Circles have radius epsilon / 2, so they touch when an edge appears.
    x_min = points[:, 0].min() - margin
    x_max = points[:, 0].max() + margin
    y_min = points[:, 1].min() - margin
    y_max = points[:, 1].max() + margin

    panel = ax_complex.get_position()
    panel_ratio = panel.width * fig.get_figwidth() / (panel.height * fig.get_figheight())
    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range / y_range < panel_ratio:
        x_center = (x_min + x_max) / 2
        x_range = panel_ratio * y_range
        x_min = x_center - x_range / 2
        x_max = x_center + x_range / 2
    else:
        y_center = (y_min + y_max) / 2
        y_range = x_range / panel_ratio
        y_min = y_center - y_range / 2
        y_max = y_center + y_range / 2

    ax_complex.set_xlim(x_min, x_max)
    ax_complex.set_ylim(y_min, y_max)
    ax_complex.set_aspect('equal', adjustable='box')
    ax_complex.set_title('Vietoris-Rips Complex')
    ax_complex.grid(True, linestyle='--', alpha=0.3)

    circles = []
    for point in points:
        circle = Circle(point, 0, color='#4A90E2', alpha=0.15, zorder=1)
        ax_complex.add_patch(circle)
        circles.append(circle)

    # Draw filled triangles before their edges, so the simplex structure is visible.
    triangle_collection = PolyCollection([], facecolors='#E74C3C', alpha=0.35, zorder=2)
    edge_collection = LineCollection([], colors='#2C3E50', linewidths=1.5, zorder=3)
    ax_complex.add_collection(triangle_collection)
    ax_complex.add_collection(edge_collection)
    ax_complex.scatter(points[:, 0], points[:, 1], color='black', zorder=4)

    ax_barcode.set_xlim(0, plot_max)
    ax_barcode.set_title('Persistence Barcode')
    ax_barcode.set_xlabel(r'Scale ($\epsilon$)')
    ax_barcode.grid(True, axis='x', linestyle='--', alpha=0.3)

    colours = ['#4A90E2', '#2ECC71', '#E74C3C']
    ticks = []
    labels = []
    y = 0
    for dim in range(max_dim):
        bars = barcode[dim]
        start = y
        for birth, death in bars:
            death = plot_max if np.isinf(death) else death
            ax_barcode.hlines(y, birth, death, color=colours[dim], linewidth=4)
            y += 1
        ticks.append((start + y - 1) / 2 if bars else y)
        labels.append(rf'$H_{dim}$')
        y += 1

    ax_barcode.set_yticks(ticks)
    ax_barcode.set_yticklabels(labels)
    epsilon_line = ax_barcode.axvline(initial_epsilon, color='gray', linestyle=':', linewidth=2)

    def update(epsilon):
        for circle in circles:
            circle.set_radius(epsilon / 2)

        triangle_collection.set_verts([
            points[list(simplex.vertices)]
            for simplex in triangles
            if simplex.distance <= epsilon
        ])
        edge_collection.set_segments([
            points[list(simplex.vertices)]
            for simplex in edges
            if simplex.distance <= epsilon
        ])
        epsilon_line.set_xdata([epsilon, epsilon])
        fig.canvas.draw_idle()

    update(initial_epsilon)

    slider_axis = fig.add_axes([0.18, 0.04, 0.64, 0.03])
    epsilon_slider = Slider(
        slider_axis,
        '',
        0,
        plot_max,
        valinit=initial_epsilon,
        color='#4A90E2'
    )
    epsilon_slider.on_changed(update)
    plt.show()
