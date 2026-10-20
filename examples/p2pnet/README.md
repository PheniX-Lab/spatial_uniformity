# P2PNet inference examples

The `images/` folder contains three UAV RGB test images. The files retain
exactly the image content supplied for this release, without additional
rotation, cropping or resampling during release preparation.

Place the downloaded `best_mae.pth` in `checkpoints/`, then run from the
repository root:

```bash
python predict.py --input_dir examples/p2pnet/images --output_dir output/p2pnet_examples --weight_path checkpoints/best_mae.pth --threshold 0.55
```

The output folder will contain an annotated image and a pixel-coordinate
TXT file for each input image.

| Image | Reference detections at threshold 0.55 |
| --- | ---: |
| plot101_DSC05821_7.png | 40 |
| plot109_DSC05770_6.png | 38 |
| plot309_DSC05215_7.png | 33 |

These are model outputs using the supplied checkpoint, rather than manually
annotated ground-truth counts. Machine-readable counts are provided in
`expected_results.csv`.

The spatial uniformity examples in `../spatial_uniformity/` use separate
projected-coordinate datasets. The pixel coordinates predicted from these
three test images do not correspond to those datasets or their boundaries.
