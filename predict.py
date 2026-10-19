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

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image

from models import build_model


def map_valid_coordinates(points, width, height, new_width, new_height):
    """Restore original-resolution coordinates and discard invalid detections."""
    scale_x = new_width / width
    scale_y = new_height / height
    coords = []
    for point in points:
        x = float(point[0]) / scale_x
        y = float(point[1]) / scale_y
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height:
            coords.append((x, y))
    return coords


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
    parser.add_argument("--cpu", action="store_true",
                        help="run on CPU even when CUDA is available")

    return parser


def main(args):
    if not 0 <= args.threshold <= 1:
        raise ValueError('--threshold must be between 0 and 1')
    if args.row < 1 or args.line < 1:
        raise ValueError('--row and --line must be positive')
    if not os.path.isfile(args.weight_path):
        raise FileNotFoundError('Trained checkpoint not found: {}'.format(args.weight_path))
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError('Input folder not found: {}'.format(args.input_dir))
    if os.path.realpath(args.input_dir) == os.path.realpath(args.output_dir):
        raise ValueError('Input and output folders must be different')
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    files = sorted(f for f in os.listdir(args.input_dir)
                   if f.lower().endswith(valid_ext)
                   and os.path.isfile(os.path.join(args.input_dir, f)))
    if not files:
        raise ValueError('No supported images found in {}'.format(args.input_dir))
    stems = [os.path.normcase(os.path.splitext(f)[0]) for f in files]
    if len(stems) != len(set(stems)):
        raise ValueError('Input images must have unique names before their extensions '
                         'to avoid overwriting coordinate files')
    # Select device (falls back to CPU when CUDA is unavailable).
    if not args.cpu and torch.cuda.is_available():
        if not 0 <= args.gpu_id < torch.cuda.device_count():
            raise ValueError('--gpu_id is outside the available GPU range')
        device = torch.device('cuda:{}'.format(args.gpu_id))
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
    resample = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS

    with torch.no_grad():
        for file in files:
            img_path = os.path.join(args.input_dir, file)
            with Image.open(img_path) as source:
                img_original = source.convert("RGB")
            width, height = img_original.size
            if width < 128 or height < 128:
                raise ValueError('{}: both image dimensions must be at least 128 pixels'
                                 .format(file))

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

            # Map the points back to the original resolution and draw them.
            coords = map_valid_coordinates(points, width, height, new_width, new_height)
            canvas = cv2.cvtColor(np.array(img_original), cv2.COLOR_RGB2BGR)

            for x, y in coords:
                cv2.circle(canvas, (int(x), int(y)), 8, (0, 0, 255), -1)

            # Save annotated image.
            # imencode + tofile supports non-ASCII Windows paths.
            success, encoded = cv2.imencode(os.path.splitext(file)[1], canvas)
            if not success:
                raise IOError('Could not encode annotated image: {}'.format(file))
            encoded.tofile(os.path.join(args.output_dir, file))

            # Save hill coordinates.
            txt_path = os.path.join(
                args.output_dir, os.path.splitext(file)[0] + ".txt")
            with open(txt_path, "w") as f:
                for x, y in coords:
                    f.write("{} {}\n".format(x, y))

            print("{}: {} hills".format(file, len(coords)))

    print("Inference finished. Results are in '{}'.".format(args.output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
