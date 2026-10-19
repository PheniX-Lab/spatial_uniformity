# Rice Planting-Hill Detection with P2PNet

A **P2PNet** model adapted to detect rice **planting hills** in UAV RGB
images. The model was trained on more than 1,600 manually annotated UAV
images and enables high-throughput localization of rice hills.

The current release provides the trained hill-detection model, inference
code and environment configuration. Spatial uniformity calculation code
will be added separately. The training workflow and manually annotated
training dataset are not released; the network architecture and training
procedure are described in the accompanying manuscript.

| Input UAV image | Detected hills (red dots) |
| --- | --- |
| <img width="1957" height="1448" alt="example_input" src="https://github.com/user-attachments/assets/4f5f3631-d757-41cf-99e0-c4617155a8f1" /> | <img width="1957" height="1448" alt="example_output" src="https://github.com/user-attachments/assets/139d06e0-4202-4f58-900d-3f11f1c50e77" /> |

## Model architecture

P2PNet consists of a **VGG16_bn** backbone, a **feature-pyramid (FPN)
decoder** and two prediction heads:

- A **point-regression head** predicts an (x, y) offset for each reference
  point arranged on a regular grid.
- A **classification head** predicts foreground and background scores
  for each point.

Points with foreground probabilities above the confidence threshold are
retained as detected hills. The default threshold is **0.55**.

<img width="1317" height="569" alt="P2PNet architecture" src="https://github.com/user-attachments/assets/7f8ea809-3b62-4814-9803-7a2b98ea772f" />

## Hardware

A CUDA-compatible NVIDIA GPU is recommended for faster inference.
The original implementation used PyTorch with CUDA 10.2.

The inference script automatically uses CPU when CUDA is unavailable.
CPU inference can also be selected explicitly with `--cpu`.

## Environment configuration

The updated inference code was tested in an existing Windows environment
with Python 3.7.13, PyTorch 1.9.0+cu102, torchvision 0.10.0+cu102,
NumPy 1.21.6, Pillow 9.1.0 and OpenCV 4.7.0.

### Option A: Conda

Run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate rice_p2pnet
```

The environment file uses `requirements.txt` to install the inference
dependencies. Keep both files in the repository root.

### Option B: pip

Inside an existing **Python 3.7** environment, run:

```bash
pip install -r requirements.txt
```

The requirements select the historical CUDA 10.2 PyTorch wheels on
Windows and Linux. These wheels also support CPU inference; GPU execution
requires a compatible NVIDIA driver.

## Prepare the files

1. Download the trained checkpoint
   [best_mae.pth from Google Drive](https://drive.google.com/file/d/1O-xFtfajLZOg8DQ2UBnxmgcZiVzraOd4/view).
2. Create a `checkpoints/` folder in the repository root and place
   `best_mae.pth` inside it.
3. Create an `input/` folder and place the UAV RGB images to process
   inside it.

The expected layout is:

```text
spatial_uniformity/
├── models/
├── util/
├── checkpoints/
│   └── best_mae.pth
├── input/
│   └── your_image.jpg
├── predict.py
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

The trained P2PNet checkpoint includes the backbone parameters.
**No separate ImageNet-pretrained VGG checkpoint is required for inference.**

The supplied checkpoint uses the `vgg16_bn` backbone and a `2 × 2`
reference-point layout. Keep the default `--backbone`, `--row` and `--line`
settings when using this checkpoint.

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`
and `.tiff`. Both image dimensions must be at least 128 pixels.

The script processes each image as a whole and does not implement tiled
inference. Large UAV images or orthomosaics may exceed available memory.

Before inference, each image dimension is rounded down to a multiple
of 128 by resizing. Predicted coordinates are then mapped back to the
original image resolution.

## Run

Run commands from the repository root.

### Default inference

```bash
python predict.py
```

The script reads images from `input/`, loads
`checkpoints/best_mae.pth` and saves predictions in `output/`.
The output folder is created automatically.

### Specify paths and confidence threshold

```bash
python predict.py --input_dir ./input --output_dir ./output --weight_path ./checkpoints/best_mae.pth --threshold 0.55
```

### Force CPU inference

```bash
python predict.py --cpu
```

### Select a GPU

GPU indices refer to the CUDA devices visible to the current process.
For example, to use the second visible GPU:

```bash
python predict.py --gpu_id 1
```

## Outputs

For each input image, the script saves:

- An annotated image with red dots marking detected hills.
- A `.txt` file containing one `x y` coordinate pair per line.

The detected hill count for each image is also printed to the console.

TXT coordinates retain **floating-point precision** and are expressed
in the **original image resolution**. The origin is at the top-left
corner, with x increasing rightwards and y increasing downwards.
Only the positions used to draw red dots are converted to integers.

Detections with non-finite coordinates or coordinates outside the original
image bounds are discarded. The annotated image, TXT file and reported
hill count use the same retained detections. Valid coordinates satisfy
`0 <= x < width` and `0 <= y < height`.

These are **image pixel coordinates**, not geographic coordinates.
Georeferencing is a separate step before spatial uniformity analysis.

Use different input and output folders. Images in the input folder must
have unique basenames: for example, `plot01.jpg` and `plot01.png` would
both produce `plot01.txt`, so they cannot be processed together.

An image with no valid detections above the threshold produces an empty
coordinate file. Reusing an output folder overwrites existing results
with the same filenames.

## Inference verification

The inference code and supplied checkpoint were tested on three UAV images
on Windows using both CPU and GPU. Both devices produced the same detection
counts and integer drawing coordinates.

## Acknowledgements

The detector is adapted from
[P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)
by Tencent Youtu Research and its contributors.

We acknowledge the original authors and retain the applicable copyright,
license conditions and disclaimers. The `models/` directory contains the
network definitions required to load the trained parameters. Some
inherited loss, matching and utility code remains, although this release
does not provide a training workflow.

## Citation

The accompanying manuscript is:

**Spatial uniformity links potential growing space to canopy development
and yield formation in rice**

Ming Shi, Yangmingrui Gao, Yanfeng Ding, Weiwei Li and Shouyang Liu.

Publication details and the final BibTeX entry will be added when available.

## License

The trained checkpoint is available for download and use with the
accompanying inference code.

P2PNet-derived code remains subject to its original license, including
its academic-research-only restriction. Original contributions and
third-party components retain their respective license terms.
See [LICENSE](LICENSE) for the complete terms and notices.
