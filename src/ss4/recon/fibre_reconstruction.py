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

def needs_reconstruction(mask):
    labeled = label(mask)
    return labeled.max() > 1   # more than one connected component = fragmented

def show_reconstruction(fibre, pil_image, out_path="reconstruction.png"):
    from skimage.morphology import convex_hull_image
    from src.ss4.seg.infer import build_instance_overlay
    import matplotlib.pyplot as plt
    from PIL import Image

    #if needs_reconstruction(fibre.mask):
    reconstructed = convex_hull_image(fibre.mask)
    gap_fill = reconstructed & ~fibre.mask

    H, W = fibre.mask.shape
    black = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))

    before_img, _ = build_instance_overlay(black, [fibre.mask], draw_boxes=False)
    after_img, _ = build_instance_overlay(black, [reconstructed], draw_boxes=False)
    after_img, _ = build_instance_overlay(after_img, [gap_fill], draw_boxes=False, alpha=0.9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.axis("off")

    axes[0].imshow(before_img)
    axes[0].set_title(f"Before  #{fibre.instance_id}", color="white", fontsize=11, fontweight="bold")

    axes[1].imshow(after_img)
    axes[1].set_title("After  (red = reconstructed area)", color="white", fontsize=11, fontweight="bold")

    plt.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  reconstruction -> {out_path}.    Reconstruction required = {needs_reconstruction(fibre.mask)}")

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
    show_reconstruction(fibres[8], pil_image, out_path="reconstruction_0.png")
    print()

if __name__ == '__main__':
    main()