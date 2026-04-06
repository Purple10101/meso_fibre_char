"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260406

fibre_measure.py
measures individual fibres length and width
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import matplotlib.pyplot as plt

import numpy as np
import cv2
import torch
import os

from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


def _measure_width(fibre, path, n_samples=20, min_dist_from_reconstruction=10):
    """
    Measure average fibre width by casting perpendicular rays at sampled
    points along the skeleton path, ignoring points near reconstructed regions.
    """
    # distance from every pixel to the nearest reconstructed pixel
    # if no reconstruction happened, all points are valid
    if fibre.reconstructed_region is not None:
        dist_from_recon = distance_transform_edt(~fibre.reconstructed_region)
    else:
        dist_from_recon = np.full(fibre.mask.shape, np.inf)

    # evenly spaced sample indices along the path
    indices = np.linspace(0, len(path) - 1, n_samples, dtype=int)

    widths = []

    for idx in indices:
        r, c = path[idx]

        # skip if too close to reconstructed region
        if dist_from_recon[r, c] < min_dist_from_reconstruction:
            continue

        # local direction vector from neighbouring path points
        i0 = max(idx - 1, 0)
        i1 = min(idx + 1, len(path) - 1)
        dr = path[i1][0] - path[i0][0]
        dc = path[i1][1] - path[i0][1]
        norm = np.sqrt(dr**2 + dc**2)
        if norm == 0:
            continue

        # perpendicular direction
        perp_r = -dc / norm
        perp_c =  dr / norm

        # cast rays in both perpendicular directions until mask boundary
        def ray_length(step_r, step_c):
            dist = 0.0
            cr, cc = float(r), float(c)
            while True:
                cr += step_r
                cc += step_c
                ri, ci = int(round(cr)), int(round(cc))
                if ri < 0 or ri >= fibre.mask.shape[0]:
                    break
                if ci < 0 or ci >= fibre.mask.shape[1]:
                    break
                if not fibre.mask[ri, ci]:
                    break
                dist += 1.0
            return dist

        left  = ray_length( perp_r,  perp_c)
        right = ray_length(-perp_r, -perp_c)
        widths.append(left + right)

    if not widths:
        return None   # all points were too close to reconstruction

    return float(np.mean(widths))

def _trace_skeleton(skeleton):
    """
    Trace skeleton pixels from one endpoint to the other.
    Returns an ordered list of (row, col) coordinates along the centreline.
    """
    # get all skeleton pixel coordinates
    ys, xs = np.where(skeleton)
    pixels = set(zip(ys.tolist(), xs.tolist()))

    # find neighbours of a pixel (8-connected)
    def get_neighbours(r, c):
        return [(r + dr, c + dc)
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if (dr != 0 or dc != 0) and (r + dr, c + dc) in pixels]

    # endpoints are pixels with only 1 neighbour
    endpoints = [p for p in pixels if len(get_neighbours(*p)) == 1]

    if len(endpoints) == 0:
        # closed loop — just pick any pixel as start
        start = next(iter(pixels))
    else:
        start = endpoints[0]

    # walk the skeleton
    path = [start]
    visited = {start}

    while True:
        current = path[-1]
        neighbours = [n for n in get_neighbours(*current) if n not in visited]
        if not neighbours:
            break
        path.append(neighbours[0])
        visited.add(neighbours[0])

    return path


def _skeleton_length(path):
    """
    Sum euclidean distances between consecutive points along the traced path.
    """
    total = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i-1][0]
        dc = path[i][1] - path[i-1][1]
        total += np.sqrt(dr**2 + dc**2)
    return total

def dim_measure(fibre, pixel_len_mm):
    mask = fibre.mask

    skeleton = skeletonize(mask)
    path = _trace_skeleton(skeleton)
    length_px = _skeleton_length(path)
    width_px = _measure_width(fibre, path)
    length_mm = length_px * pixel_len_mm
    width_mm = width_px * pixel_len_mm

    return length_mm, width_mm


def main():
    from src.ss4.seg.model import build_model
    from src.ss4.seg.infer import run_inference, preprocess_array
    from src.ss4.recon.fibre_reconstruction import image_fibres_reconstruction
    from src.common.paths import IMAGE_PATH

    # inference setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("maskrcnn_resnet50_fpn_v2", pretrained=False)

    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "seg\\runs\\fibre_maskrcnn\\best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # load in a fibre image
    images = [f for f in os.listdir(IMAGE_PATH) if f.lower().endswith((".png", ".jpg", ".bmp"))]
    image_path_total = IMAGE_PATH / images[4]
    image = cv2.imread(image_path_total)
    fibres = run_inference(model, image, device, debug=True, debug_out_dir="inf_dbg")
    image_fibres_reconstruction(fibres)

    # fake pixel size — 512px image assumed to be 0.5mm across
    pixel_len_mm = 0.5 / 512

    fibre = fibres[14]
    mask  = fibre.mask

    skeleton  = skeletonize(mask)
    path      = _trace_skeleton(skeleton)
    length_px = _skeleton_length(path)
    width_px  = _measure_width(fibre, path)
    length_mm, width_mm = dim_measure(fibre, pixel_len_mm)

    # sample points used for width measurement
    n_samples = 20
    sample_indices = np.linspace(0, len(path) - 1, n_samples, dtype=int)

    # distance from reconstruction for colouring valid/invalid sample points
    if fibre.reconstructed_region is not None:
        dist_from_recon = distance_transform_edt(~fibre.reconstructed_region)
    else:
        dist_from_recon = np.full(mask.shape, np.inf)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.axis("off")

    # ── Panel 0: mask + reconstruction region ────────────────────────────────
    display = np.zeros((*mask.shape, 3), dtype=np.uint8)
    display[mask]  = [80, 190, 100]   # green = real mask
    if fibre.reconstructed_region is not None:
        display[fibre.reconstructed_region] = [220, 50, 50]   # red = reconstructed

    axes[0].imshow(display)
    axes[0].set_title("Mask  (red = reconstructed)", color="white", fontsize=10, fontweight="bold")

    # ── Panel 1: skeleton overlaid on mask ───────────────────────────────────
    axes[1].imshow(display, alpha=0.5)
    axes[1].imshow(skeleton, cmap="hot", alpha=0.9)

    # draw perpendicular width lines at each valid sample point
    for idx in sample_indices:
        r, c = path[idx]
        valid = dist_from_recon[r, c] >= 10

        i0 = max(idx - 1, 0)
        i1 = min(idx + 1, len(path) - 1)
        dr = path[i1][0] - path[i0][0]
        dc = path[i1][1] - path[i0][1]
        norm = np.sqrt(dr**2 + dc**2)
        if norm == 0:
            continue
        perp_r = -dc / norm
        perp_c =  dr / norm

        # cast rays to find boundary
        def ray_end(step_r, step_c):
            cr, cc = float(r), float(c)
            while True:
                cr += step_r
                cc += step_c
                ri, ci = int(round(cr)), int(round(cc))
                if ri < 0 or ri >= mask.shape[0]: break
                if ci < 0 or ci >= mask.shape[1]: break
                if not mask[ri, ci]: break
            return cr, cc

        r1, c1 = ray_end( perp_r,  perp_c)
        r2, c2 = ray_end(-perp_r, -perp_c)

        color = "#44bb66" if valid else "#cc4444"
        alpha = 0.8      if valid else 0.3
        axes[1].plot([c1, c2], [r1, r2], color=color, linewidth=1, alpha=alpha)
        axes[1].plot(c, r, "o", color=color, markersize=2)

    axes[1].set_title("Skeleton + width samples\n(green = used, red = near reconstruction)",
                      color="white", fontsize=10, fontweight="bold")

    # ── Panel 2: measurements summary ────────────────────────────────────────
    axes[2].imshow(display, alpha=0.3)
    axes[2].imshow(skeleton, cmap="hot", alpha=0.6)

    # draw the full skeleton path as a line
    path_cols = [p[1] for p in path]
    path_rows = [p[0] for p in path]
    axes[2].plot(path_cols, path_rows, color="#4499ff", linewidth=1.5, alpha=0.9)

    # annotate length along the skeleton
    mid = len(path) // 2
    axes[2].annotate(
        f"L = {length_mm:.4f}mm\n({length_px:.1f}px)",
        xy=(path[mid][1], path[mid][0]),
        xytext=(path[mid][1] + 15, path[mid][0] - 15),
        color="white", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
    )

    summary = (
        f"length  : {length_mm:.4f} mm  ({length_px:.1f} px)\n"
        f"width   : {width_mm:.4f} mm  ({width_px:.1f} px)\n"
        f"px size : {pixel_len_mm:.6f} mm/px\n"
        f"samples : {n_samples} points\n"
        f"valid   : {sum(dist_from_recon[path[i][0], path[i][1]] >= 10 for i in sample_indices)}/{n_samples}"
    )
    axes[2].text(
        0.02, 0.02, summary,
        transform=axes[2].transAxes,
        color="white", fontsize=8,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#222222", alpha=0.8),
    )
    axes[2].set_title("Measurements", color="white", fontsize=10, fontweight="bold")

    plt.suptitle(f"Fibre #{fibre.instance_id}  —  score={fibre.score:.2f}",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout(pad=1.5)
    plt.savefig("fibre_measurement.png", dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"  length = {length_mm:.4f}mm   width = {width_mm:.4f}mm")
    print()

if __name__ == '__main__':
    main()