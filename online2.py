import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import ttach as tta
import time
import os
import multiprocessing.pool as mpp
import multiprocessing as mp

from train import *
import torchmetrics

import argparse
from utils.config import Config
from tools.mask_convert import mask_save

def get_args():
    parser = argparse.ArgumentParser('description=online test')
    parser.add_argument("-c", "--config", type=str, default="configs/logcan.py")
    parser.add_argument("--ckpt", type=str, default="work_dirs/LoGCAN_ResNet50_Loveda/epoch=45.ckpt")
    parser.add_argument("--tta", type=str, default="d4")
    parser.add_argument("--masks_output_dir", default=None)
    return parser.parse_args()

# Mapping function for mask values to class indices

def map_mask_values(mask):
    values = torch.tensor([0, 32, 64, 128, 160, 192, 255], device=mask.device)
    mapping = torch.arange(len(values), device=mask.device)

    # Use searchsorted / bucketize
    mapped_mask = torch.zeros_like(mask, dtype=torch.long)
    for i in range(len(values)):
        mapped_mask[mask == values[i]] = mapping[i]

    return mapped_mask

if __name__ == "__main__":
    args = get_args()
    cfg = Config.fromfile(args.config)
    print("Config loaded keys:", cfg.keys())

    if args.masks_output_dir is not None:
        masks_output_dir = args.masks_output_dir
    else:
        masks_output_dir = cfg.exp_name + '/online_figs'

    model = myTrain.load_from_checkpoint(args.ckpt, cfg=cfg)
    model = model.to('cuda')
    model.eval()

    # Apply TTA
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

    # Initialize Metrics
    num_classes = cfg.num_class
    device = 'cuda'

    metric_oa = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    metric_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    metric_iou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=num_classes).to(device)
    metric_precision = torchmetrics.classification.MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    metric_recall = torchmetrics.classification.MulticlassRecall(num_classes=num_classes, average='macro').to(device)

    results = []
    mask2RGB = False
    with torch.no_grad():
        test_loader = build_dataloader(cfg.dataset_config, mode='test')
        print(len(test_loader))
        for input in tqdm(test_loader):
            images = input[0].cuda()
            masks = input[1].cuda()
            img_ids = input[2]

            raw_predictions = model(images, True)
            pred = raw_predictions.argmax(dim=1)
            pred = pred.long()

            # Map masks to class indices [0, num_classes-1]
            masks_mapped = map_mask_values(masks)
            print("Unique mask values before mapping:", masks.unique())
            print("Unique mask values after mapping:", masks_mapped.unique())

            # Debug check — optional
            # print("Unique mask values (after mapping):", masks_mapped.unique())

            # Update metrics
            metric_oa(pred, masks_mapped)
            metric_f1(pred, masks_mapped)
            metric_iou(pred, masks_mapped)
            metric_precision(pred, masks_mapped)
            metric_recall(pred, masks_mapped)

            for i in range(raw_predictions.shape[0]):
                mask_pred = pred[i].cpu().numpy()
                mask_name = str(img_ids[i])
                results.append((mask2RGB, mask_pred, cfg.dataset, masks_output_dir, mask_name))

    # Output metrics
    print("\n--- Evaluation Metrics ---")
    print(f"Overall Accuracy (OA): {metric_oa.compute().item():.4f}")
    print(f"F1 Score (Macro): {metric_f1.compute().item():.4f}")
    print(f"Mean IoU: {metric_iou.compute().item():.4f}")
    print(f"Precision (Macro): {metric_precision.compute().item():.4f}")
    print(f"Recall (Macro): {metric_recall.compute().item():.4f}")
    print("--------------------------\n")

    # Skipping mask saving
    print("Masks were processed but not saved.")

    # Simulated image write time
    t0 = time.time()
    t1 = time.time()
    img_write_time = t1 - t0
    print('Image saving skipped. Estimated time skipped: {} s'.format(img_write_time))
