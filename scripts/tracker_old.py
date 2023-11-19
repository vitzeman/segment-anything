import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import argparse

from tqdm import tqdm

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# TODO: Add parsing of arguments
# TODO: get it working from directory with images
# TODO: Save annotations of the bounding boxes
# TODO: Maybe try to select the new mask by comparing the old one and the new one


def init_sam():
    sam_checkpoint = "segment_anything/checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print(f"[INFO]: SAM model loaded {type(sam_predictor)}")
    return sam_predictor


def get_bbox_from_mask(mask: np.ndarray) -> tuple:
    """Get bounding box from mask in x1,y1,x2,y2 format

    Args:
        mask (np.ndarray): binary mask with shape (H,W)
    """
    mask = mask[0, :, :]
    y, x = np.nonzero(mask)
    x1, x2 = np.min(x), np.max(x)
    y1, y2 = np.min(y), np.max(y)
    return (x1, y1, x2, y2)


def expand_bbox(bbox: tuple, factor: float) -> tuple:
    """Expand bounding box by factor

    Args:
        bbox (tuple): Bbox in x1,y1,x2,y2 format
        factor (float): Factor to expand by

    Returns:
        tuple: Expanded bbox in x1,y1,x2,y2 format
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x1 = int(x1 - w * (factor - 1) / 2)
    y1 = int(y1 - h * (factor - 1) / 2)
    x2 = int(x2 + w * (factor - 1) / 2)
    y2 = int(y2 + h * (factor - 1) / 2)
    return (x1, y1, x2, y2)


def tracking_test(init_bbox: tuple):
    vidcap = cv2.VideoCapture("data/videos/test.mp4")

    if not vidcap.isOpened():
        print("Cannot open camera")
        exit()

    num_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO]: Total num of frames:{num_of_frames}")
    bbox = init_bbox
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    sam = init_sam()

    new_frames = []

    for frame_num in tqdm(range(num_of_frames)):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = vidcap.read()
        if not ret:
            print(f"[ERROR]: Cannot read frame {frame_num}")
            exit()
        sam.set_image(frame)
        mask, scores, logits = sam.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(bbox),
            multimask_output=False,
        )
        new_bbox = get_bbox_from_mask(mask)
        expanded_bbox = expand_bbox(new_bbox, 1.1)

        mask_to_show = mask[0, :, :].reshape((mask.shape[1], mask.shape[2], 1))
        mask_to_show = np.repeat(mask_to_show, 3, axis=2)
        mask_to_show = mask_to_show.astype(np.uint8) * 255
        frame = cv2.addWeighted(frame, 0.5, mask_to_show, 0.5, 0)

        cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 2)
        cv2.rectangle(frame, new_bbox[:2], new_bbox[2:], (0, 0, 255), 2)
        cv2.rectangle(frame, expanded_bbox[:2], expanded_bbox[2:], (255, 0, 0), 2)

        bbox = expanded_bbox
        cv2.imshow("Tracking", frame)
        new_frames.append(frame)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break

    vidcap.release()
    cv2.destroyAllWindows()
    # Save video
    height, width, layers = new_frames[0].shape
    size = (width, height)
    fps = 30
    out = cv2.VideoWriter(
        "data/videos/test_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    for i in tqdm(range(len(new_frames))):
        out.write(new_frames[i])
    out.release()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--image_dir", type=str, required=True)
    # args.add_argument("--save_folder", type=str, required=True)
    # args.add_argument("--sam_checkpoint", type=str, required=True)
    # args.add_argument("--model_type", type=str, required=True)
    args.add_argument("--init_bbox", nargs="+", type=int, required=True)

    args = args.parse_args()
    if len(args.init_bbox) != 4:
        raise ValueError(
            f"[ERROR]: init_bbox must have 4 elements, got {len(args.init_bbox)}"
        )
    tracking_test(tuple(args.init_bbox))
