#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
from typing import Tuple, List, Dict

from GONet_Wizard.GONet_utils import GONetFile

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def label_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    labels = np.zeros_like(mask, dtype=int)
    current_label = 0
    rows, cols = mask.shape

    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            if labels[r, c] != 0:
                continue
            current_label += 1
            stack = [(r, c)]
            labels[r, c] = current_label
            while stack:
                rr, cc = stack.pop()
                for dr, dc in NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue
                    if mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current_label
                        stack.append((nr, nc))
    return labels, current_label


def measure_sources(sub: np.ndarray, labels: np.ndarray, num_labels: int) -> List[Dict]:
    sources: List[Dict] = []
    for label_id in range(1, num_labels + 1):
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0:
            continue
        fluxes = sub[ys, xs]
        total_flux = float(fluxes.sum())
        if total_flux <= 0:
            continue
        x_centroid = float((xs * fluxes).sum() / total_flux)
        y_centroid = float((ys * fluxes).sum() / total_flux)
        sources.append({
            "label": label_id,
            "x_centroid": x_centroid,
            "y_centroid": y_centroid,
            "flux": total_flux,
            "npix": int(len(xs)),
        })
    return sources


def compute_img_centroids(image_path: Path, N: float = 5.0, top_k: int = 300) -> np.ndarray:
    go = GONetFile.from_file(Path(image_path))
    go.remove_overscan()
    img_gray = go.green

    # Threshold bright pixels
    sub = img_gray
    sub_mean = float(sub.mean())
    sub_std = float(sub.std())
    threshold = sub_mean + N * sub_std
    mask = sub > threshold

    # Label components
    labels, num_labels = label_connected_components(mask)

    # Measure centroids
    sources = measure_sources(sub, labels, num_labels)

    # Sort by flux desc and take top_k
    sources_sorted = sorted(sources, key=lambda s: s["flux"], reverse=True)[:top_k]
    img_xy = np.array([[s["x_centroid"], s["y_centroid"]] for s in sources_sorted], dtype=float)
    return img_xy


def main():
    parser = argparse.ArgumentParser(description="Compute centroids (img_xy) from a GONet image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--N", type=float, default=5.0, help="Threshold multiplier (mean + N*std)")
    parser.add_argument("--top_k", type=int, default=300, help="Number of brightest sources to keep")
    parser.add_argument("--save", action="store_true", help="Save img_xy .npy next to image")
    args = parser.parse_args()

    img_xy = compute_img_centroids(Path(args.image), N=args.N, top_k=args.top_k)
    print(f"Computed {len(img_xy)} centroids.")

    if args.save:
        out_path = Path(args.image).with_suffix("")
        out_file = out_path.parent / f"{out_path.name}_img_xy.npy"
        np.save(out_file, img_xy)
        print(f"Saved img_xy to: {out_file}")
    else:
        # Print a brief preview (first 5 rows)
        print("img_xy (preview):")
        print(img_xy[:5])


if __name__ == "__main__":
    main()
