import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import ttach as tta
import time
import os
import os.path as osp

from train import *
import argparse
from utils.config import Config
from tools.mask_convert import mask_save

def get_args():
    parser = argparse.ArgumentParser(description='Semantic segmentation of remote sensing images')
    parser.add_argument("-c", "--config", type=str, default="configs/loveda/logcanplus.py")
    parser.add_argument("--ckpt", type=str, default="work_dirs/logcanplus_loveda/epoch=40.ckpt")
    parser.add_argument("--tta", type=str, default="d4")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    cfg = Config.fromfile(args.config)

    # Set output directory
    masks_output_dir = osp.join("work_dirs", "predictions")
    os.makedirs(masks_output_dir, exist_ok=True)
    print("masks_save_dir: ", masks_output_dir)

    # Load model checkpoint
    model = myTrain.load_from_checkpoint(args.ckpt, cfg=cfg)
    model = model.to('cuda')
    model.eval()

    # Apply TTA if specified
    if args.tta == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip()
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[90]),
            tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)

    # Load test data (without masks)
    test_loader = build_dataloader(cfg.dataset_config, mode='val')  # Make sure this doesn't load masks

    with torch.no_grad():
        for idx, input in enumerate(test_loader):
            if idx >= 3:
                break

            # Skip loading masks (use only image and id)
            if len(input) == 3:
                images, _, img_ids = input
            elif len(input) == 2:
                images, img_ids = input
            else:
                raise ValueError("Unexpected number of elements returned by the dataloader.")

            images = images.cuda()

            # Predict masks
            raw_predictions = model(images, True)
            pred = raw_predictions.argmax(dim=1)

            # Process and save predictions
            mask_pred = pred[0].cpu().numpy()
            mask_name = str(img_ids[0])

            # Create unique timestamp for each image
            timestamp = time.strftime("%H%M%S")
            unique_mask_name = f"{mask_name}_{timestamp}.png"

            results = (True, mask_pred, cfg.dataset, masks_output_dir, unique_mask_name)

            mask_save(results)
            print(f"Saved mask: {unique_mask_name}")

    print("Done — saved masks for first 3 images.")
