"""
P2PNet rice planting-hill (hill) detection — inference script.

Given a folder of UAV RGB images, a trained P2PNet predicts the location of
every rice hill. For each input image the script writes:
  - an annotated image (red dots mark detected hills),
  - a .txt file with one "x y" coordinate per line, in the original image
    resolution.

Only inference is distributed with this repository; the training code and the
manual-annotation dataset are not released. The network architecture and the
training procedure are described in the accompanying paper.
"""

import argparse
import os
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image

from models import build_model

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(
        "P2PNet rice-hill detection (inference)", add_help=False)

    # * Model
    parser.add_argument("--backbone", default="vgg16_bn", type=str,
                        help="convolutional backbone to use")
    parser.add_argument("--row", default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument("--line", default=2, type=int,
                        help="line number of anchor points")

    # * Paths
    parser.add_argument("--input_dir", default="./input",
                        help="folder containing the input RGB images")
    parser.add_argument("--output_dir", default="./output",
                        help="folder where annotated images and .txt are saved")
    parser.add_argument("--weight_path", default="./checkpoints/best_mae.pth",
                        help="path to the trained P2PNet checkpoint")

    # * Inference settings
    parser.add_argument("--threshold", default=0.55, type=float,
                        help="confidence threshold; points below it are discarded")
    parser.add_argument("--gpu_id", default=0, type=int,
                        help="id of the GPU to use")

    return parser


def main(args):
    # Select device (falls back to CPU when CUDA is unavailable).
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Running on:", device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Build the model and load the trained weights.
    model = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.weight_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Image pre-processing.
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])

    # Pillow renamed ANTIALIAS to LANCZOS; support both.
    resample = getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", Image.BICUBIC))

    # Collect image files.
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = sorted(os.listdir(args.input_dir))
    files = [f for f in files if f.lower().endswith(valid_ext)]

    with torch.no_grad():
        for file in files:
            img_path = os.path.join(args.input_dir, file)
            img_original = Image.open(img_path).convert("RGB")
            width, height = img_original.size

            # Round the dimensions down to a multiple of 128.
            new_width = width // 128 * 128
            new_height = height // 128 * 128
            img_resized = img_original.resize((new_width, new_height), resample)

            # Run inference.
            samples = transform(img_resized).unsqueeze(0).to(device)
            outputs = model(samples)

            outputs_scores = torch.nn.functional.softmax(
                outputs["pred_logits"], -1)[0, :, 1]
            outputs_points = outputs["pred_points"][0]

            keep = outputs_scores > args.threshold
            points = outputs_points[keep].detach().cpu().numpy()
            predict_cnt = int(keep.sum())

            # Map the points back to the original resolution and draw them.
            scale_x = new_width / width
            scale_y = new_height / height
            canvas = cv2.cvtColor(np.array(img_original), cv2.COLOR_RGB2BGR)

            coords = []
            for p in points:
                x = int(p[0] / scale_x)
                y = int(p[1] / scale_y)
                coords.append((x, y))
                cv2.circle(canvas, (x, y), 8, (0, 0, 255), -1)

            # Save annotated image.
            cv2.imwrite(os.path.join(args.output_dir, file), canvas)

            # Save hill coordinates.
            txt_path = os.path.join(
                args.output_dir, os.path.splitext(file)[0] + ".txt")
            with open(txt_path, "w") as f:
                for x, y in coords:
                    f.write("{} {}\n".format(x, y))

            print("{}: {} hills".format(file, predict_cnt))

    print("Inference finished. Results are in '{}'.".format(args.output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
