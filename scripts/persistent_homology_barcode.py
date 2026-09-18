# scripts/persistent_homology_barcode.py
# Run from the root folder as: python -m scripts.persistent_homology_barcode

import numpy as np

from classical_algorithms.persistent_homology import persistence_barcode
from utils.data import circle
from utils.visualization import show_visualization


if __name__ == "__main__":

    # Create data
    np.random.seed(7)
    circle1 = circle(10, radius=1.4, noise=0.2, center=(-4.0, 4.0))
    circle2 = circle(6, radius=0.8, noise=0.1, center=(4.0, 4.0))
    circle3 = circle(4, radius=0.55, noise=0.01, center=(-4.0, -4.0))
    circle4 = circle(8, radius=1.85, noise=0.5, center=(4.0, -4.0))
    data = np.vstack((circle1, circle2, circle3, circle4))

    # Compute persistent homology and barcode
    max_dim = 2
    barcode, filtration = persistence_barcode(data, max_dim)

    print("\n--- Persistence Barcodes ---")
    for dim, bars in barcode.items():
        for birth, death in bars:
            if np.isinf(death):
                print(f"H{dim} Feature: Born at {birth:.2f}, Lives Forever (inf)")
            else:
                print(f"H{dim} Feature: Born at {birth:.2f}, Died at {death:.2f}")

    # Visualize
    show_visualization(data, filtration, barcode, margin=0.9)
