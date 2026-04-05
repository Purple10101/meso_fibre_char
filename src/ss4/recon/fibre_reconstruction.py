"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260327

fibre_reconstruction.py
reconstruction of fragmented fibres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from skimage.morphology import convex_hull_image
import numpy as np
import torch
import os



def main():
    from msg.src.ss4.seg.model import build_model
    from msg.src.ss4.seg.infer import run_inference
    from msg.src.common.paths import IMAGE_PATH

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
    fibres = run_inference(model, images[0], device, debug=False)
    print()

if __name__ == '__main__':
    main()