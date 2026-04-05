"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260327

fibre_reconstruction.py
reconstruction of fragmented fibres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import PIL
from skimage.morphology import convex_hull_image
from skimage.measure import label

import numpy as np
import cv2
import torch
import os




from skimage.measure import label
from skimage.segmentation import find_boundaries
from skimage.draw import line
import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage.draw import line
from skimage.morphology import binary_dilation, disk


def _find_fragments(mask):
    labeled = label(mask)
    n = labeled.max()
    fragments = []
    for i in range(1, n + 1):
        fragments.append(labeled == i)
    return fragments


def _fragment_proximity(frag_a, frag_b):
    # only compare boundary pixels — much faster than all pixels
    bounds_a = find_boundaries(frag_a, mode='inner')
    bounds_b = find_boundaries(frag_b, mode='inner')

    ys_a, xs_a = np.where(bounds_a)
    ys_b, xs_b = np.where(bounds_b)

    pts_a = np.stack([xs_a, ys_a], axis=1).astype(np.float32)
    pts_b = np.stack([xs_b, ys_b], axis=1).astype(np.float32)

    # find minimum distance between boundary pixel sets
    dists = np.linalg.norm(pts_a[:, None] - pts_b[None, :], axis=-1)
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)

    nearest_a = (ys_a[min_idx[0]], xs_a[min_idx[0]])
    nearest_b = (ys_b[min_idx[1]], xs_b[min_idx[1]])

    return dists.min(), nearest_a, nearest_b


def _fragment_radius(frag):
    dist_transform = distance_transform_edt(frag)
    return dist_transform.max()

def _fragment_merge(mask, frag_a, frag_b):
    _, nearest_a, nearest_b = _fragment_proximity(frag_a, frag_b)

    radius_a = _fragment_radius(frag_a)
    radius_b = _fragment_radius(frag_b)
    bridge_radius = int(round((radius_a + radius_b) / 2))

    # draw the line
    rr, cc = line(nearest_a[0], nearest_a[1], nearest_b[0], nearest_b[1])
    bridge = np.zeros_like(mask, dtype=bool)
    bridge[rr, cc] = True

    # dilate the line by the average radius
    bridge = binary_dilation(bridge, footprint=disk(bridge_radius))

    merged = mask.copy()
    merged[bridge] = True

    return merged


def reconstruct(mask):
    fragments = _find_fragments(mask)

    if len(fragments) == 1:
        return mask   # nothing to do

    current_mask = mask.copy()

    while True:
        fragments = _find_fragments(current_mask)
        if len(fragments) == 1:
            break

        # find the closest pair of fragments
        best_dist  = np.inf
        best_pair  = (0, 1)
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                dist, _, _ = _fragment_proximity(fragments[i], fragments[j])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)

        current_mask = _fragment_merge(
            current_mask,
            fragments[best_pair[0]],
            fragments[best_pair[1]],
        )

    return current_mask



def needs_reconstruction(mask):
    labeled = label(mask)
    return labeled.max() > 1   # more than one connected component = fragmented

def show_reconstruction(fibre, pil_image, out_path="reconstruction.png"):
    from skimage.morphology import convex_hull_image
    from src.ss4.seg.infer import build_instance_overlay
    import matplotlib.pyplot as plt
    from PIL import Image

    original_mask = fibre.mask.copy()
    reconstructed_mask = reconstruct(fibre.mask)

    H, W = original_mask.shape
    black = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))

    before_img, _ = build_instance_overlay(black, [original_mask], draw_boxes=False)
    after_img, _ = build_instance_overlay(black, [reconstructed_mask], draw_boxes=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.axis("off")

    axes[0].imshow(before_img)
    axes[0].set_title(f"Before  #{fibre.instance_id}", color="white", fontsize=11, fontweight="bold")

    axes[1].imshow(after_img)
    axes[1].set_title(f"After", color="white", fontsize=11, fontweight="bold")

    plt.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  reconstruction → {out_path}")

def main():
    from src.ss4.seg.model import build_model
    from src.ss4.seg.infer import run_inference, preprocess_array
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
    image_path_total = IMAGE_PATH / images[0]
    image = cv2.imread(image_path_total)
    fibres = run_inference(model, image, device, debug=True, debug_out_dir="inf_dbg")

    tensor, pil_image = preprocess_array(image, 512)
    #fibres[8].mask = reconstruct(fibres[8].mask)
    show_reconstruction(fibres[8], pil_image, out_path="reconstruction_0.png")
    print()

if __name__ == '__main__':
    main()