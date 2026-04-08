"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260326

ss4.py
image processing sub system emulator
TODO: Time everything!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import asyncio
from multiprocessing import Queue
import torch
import cv2
import os
import sqlite3

from src.common.common import Node, cprint, SharedImage
from src.ss4.seg.model import build_model
from src.ss4.seg.infer import run_inference
from src.ss4.recon.fibre_reconstruction import image_fibres_reconstruction
from src.ss4.meas.fibre_measure import dim_measure
from src.common.db import write_ss4_results, read_image_results
from src.ss4.client_comms.websocket import start_server, broadcast


def _get_pixel_side_len(x_mm, y_mm, x_px, y_px):
    x_pixel_size = x_mm / x_px
    y_pixel_size = y_mm / y_px
    return (x_pixel_size + y_pixel_size) / 2


class ImageProcessingSS4:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device : {self.device}")

        self.model = build_model("maskrcnn_resnet50_fpn_v2", pretrained=False)

        ckpt_path = os.path.join(os.path.dirname(__file__), "seg/runs/fibre_maskrcnn/best.pth")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded : {ckpt_path}")

    def run(self, image, metadata):
        """
        run image analysis pipeline according to metadata provided
        TODO: add support for invalid image
        TODO: Add certainty per fibre?
        TODO: Add support for no SS5 presence
        """
        ret_val = {"image_id": metadata["image_id"],
                   "char": []}
        x_px, y_px, _ = image.shape
        fibres = run_inference(self.model, image, self.device,
                               debug=True, debug_stem=f"frame_{metadata['image_id']}")
        image_fibres_reconstruction(fibres)
        px_len = _get_pixel_side_len(metadata["x_mm"], metadata["y_mm"], x_px, y_px)

        ret_val["char"] = [
            {"mesh_id": i,
             "dimensions": {"length": length_mm, "width": width_mm}}
            for i, fibre in enumerate(fibres)
            for length_mm, width_mm in [dim_measure(fibre, px_len)]
        ]
        return ret_val


def run_ss4(inbox: Queue, peers: dict[str, Queue]):

    async def main():
        node = Node("ss4", inbox, peers)
        proc = ImageProcessingSS4()

        await start_server()

        cprint("ss4", f"Image Processor Ready.")

        async def on_publish_ready(msg):
            image_id = msg["data"]["image_id"]
            all_data = read_image_results(image_id)
            await broadcast(all_data)
            cprint("ss4", f"Published: {all_data}")

        async def on_image_data(msg):
            image_path = msg["data"]["image_path"]
            metadata = msg["data"]["metadata"]

            image = cv2.imread(image_path)

            cprint("ss4", f"Loaded {image_path} ({image.shape}) "
                          f"(x={metadata['x_mm']}, y={metadata['y_mm']})")

            result = proc.run(image, metadata)

            # Write SS4 results to database
            write_ss4_results(
                result["image_id"],
                [{"mesh_id": f["mesh_id"],
                  "length_mm": f["dimensions"]["length"],
                  "width_mm": f["dimensions"]["width"]}
                 for f in result["char"]]
            )

            cprint("ss4", f"Processing complete: {result}")

            signal_ready()
            send_analysis(result)

        async def on_no_images(msg):
            cprint("ss4", "ss3 has no more images. Waiting.")

        def send_analysis(result):
            """Give results to ss5"""
            node.send("ss5", "processing_result", {
                "result": result
            })

        def signal_ready():
            node.send("ss3", "ready_message", {})

        node.on("image_data_message", on_image_data)
        node.on("no_images", on_no_images) # this message wont be part of the real system
        node.on("ready_message", on_publish_ready)


        while True:
            await node.poll()
            await asyncio.sleep(8)

    asyncio.run(main())