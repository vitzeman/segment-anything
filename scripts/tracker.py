import numpy as np

# import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import argparse
import json

from tqdm import tqdm

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# TODO: get it working from directory with images
# TODO: Save annotations of the bounding boxes
# TODO: Maybe try to select the new mask by comparing the old one and the new one
LABELS = {
    1: "d01_controller",
    2: "d02_servo",
    3: "d03_main",
    4: "d04_motor",
    5: "d05_axle_front",
    6: "d06_battery",
    7: "d07_axle_rear",
    8: "d08_chassis",
}
ASCII_NUM_SHIFT = 48
LABELS_NUMS_KEY = [x + ASCII_NUM_SHIFT for x in LABELS.keys()]

MODELS_CHECKPOINTS = {
    "vit_h": "segment_anything/checkpoints/sam_vit_h_4b8939.pth",
    "vit_b": "segment_anything/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "segment_anything/checkpoints/sam_vit_l_0b3195.pth",
}


def init_sam():
    sam_checkpoint = "segment_anything/checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam_checkpoint = MODELS_CHECKPOINTS[model_type]

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
    # mask = mask[0, :, :]
    y, x = np.nonzero(mask)
    x1, x2 = np.min(x), np.max(x)
    y1, y2 = np.min(y), np.max(y)
    return (x1, y1, x2, y2)


def expand_bbox(bbox: tuple, factor: float, resolution=tuple) -> tuple:
    """Expand bounding box by factor

    Args:
        bbox (tuple): Bbox in x1,y1,x2,y2 format
        factor (float): Factor to expand by

    Returns:
        tuple: Expanded bbox in x1,y1,x2,y2 format
    """
    height, width = resolution
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x1 = int(x1 - w * (factor - 1) / 2)
    y1 = int(y1 - h * (factor - 1) / 2)
    x2 = int(x2 + w * (factor - 1) / 2)
    y2 = int(y2 + h * (factor - 1) / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)

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
        expanded_bbox = expand_bbox(new_bbox, 1.1, frame.shape[:2])
        print(frame.shape[:2])
        mask_to_show = mask[0, :, :].reshape((mask.shape[1], mask.shape[2], 1))
        mask_to_show = np.repeat(mask_to_show, 3, axis=2)
        mask_to_show = mask_to_show.astype(np.uint8) * 255
        frame = cv2.addWeighted(frame, 0.5, mask_to_show, 0.5, 0)

        cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 2)
        cv2.rectangle(frame, new_bbox[:2], new_bbox[2:], (0, 0, 255), 2)
        cv2.rectangle(frame, expanded_bbox[:2], expanded_bbox[2:], (255, 0, 0), 2)

        bbox = new_bbox
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


def erode_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Erode mask

    Args:
        mask (np.ndarray): mask to erode
        kernel_size (int, optional): erode kernmel size. Defaults to 5.

    Returns:
        np.ndarray: eroded mask
    """
    mask = mask.astype(np.uint8)[0, :, :]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def undistort_image(image: np.ndarray, camera_dict: dict) -> np.ndarray:
    """Undistort image

    Args:
        image (np.ndarray): image to undistort
        camera_dict (dict): _description_

    Returns:
        np.ndarray: _description_
    """


def track_and_cut(
    init_bbox: tuple,
    path2video: str,
    item: str,
    step: int,
    stop_frame: int,
    create_masks: bool,
    camera_dict: dict,
) -> None:
    # TODO: Add continuing from specofic frame and ending at specific frame
    """Track the object in video and cut it into frames for further processing.
        Also saves the bounding boxes for each frame.

    Args:
        init_bbox (tuple): Initial bounding box in x1,y1,x2,y2 format
        path2video (str): Path to video
        item (str): Item name
        step (int): 0 to go frame by frame
        stop_frame (int): Stop frame, if None, then go to the end
        create_masks (bool): Create binary masks for each frame
        camera_dict (dict): Camera parameters
    """
    vidcap = cv2.VideoCapture(path2video)
    if not vidcap.isOpened():
        print(f"[ERROR]Cannot open video {path2video}")
        exit()

    num_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO]: Total num of frames:{num_of_frames}")
    bbox = init_bbox
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    sam = init_sam()

    new_frames = []
    frames = []
    frames_names = []
    frames_bboxes = []
    os.makedirs("data/" + item + "/inputs/", exist_ok=True)
    os.makedirs("data/" + item + "/images/", exist_ok=True)
    if create_masks:
        os.makedirs("data/" + item + "/masks/", exist_ok=True)

    mtx = np.array(camera_dict["camera_matrix"])
    dist = np.array(camera_dict["distortion_coefficients"])
    h = camera_dict["height"]
    w = camera_dict["width"]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    x, y, w, h = roi
    camera_data = {
        "K": newcameramtx.tolist(),
        "resolution": [h, w],
    }
    with open("data/" + item + "/camera_data.json", "w") as f:
        json.dump(camera_data, f, indent=2)

    for frame_num in tqdm(range(num_of_frames), desc="Tracking"):
        if frame_num == num_of_frames - 1 or frame_num == stop_frame:
            break
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = vidcap.read()
        if not ret:
            print(f"[ERROR]: Cannot read frame {frame_num}")
            break

        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        frame = frame[y : y + h, x : x + w]

        frames.append(frame)
        sam.set_image(frame)
        mask, scores, logits = sam.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(bbox),
            multimask_output=False,
        )
        eroded_mask = erode_mask(mask)
        new_bbox = get_bbox_from_mask(eroded_mask)

        size_new_bbox = new_bbox[2] - new_bbox[0] * new_bbox[3] - new_bbox[1]
        size_old_bbox = bbox[2] - bbox[0] * bbox[3] - bbox[1]

        expanded_bbox = expand_bbox(new_bbox, 1.1, frame.shape[:2])
        frame_name = "img_" + str(frame_num).zfill(4)

        # Save frame
        cv2.imwrite("data/" + item + "/images/" + frame_name + ".png", frame)
        # Save bbox into json
        with open("data/" + item + "/inputs/" + frame_name + ".json", "w") as f:
            d = [{"label": item, "bbox_modal": expanded_bbox}]
            json.dump(d, f, indent=2)

        mask_to_show = mask[0, :, :].reshape((mask.shape[1], mask.shape[2], 1))
        mask_to_show = np.repeat(mask_to_show, 3, axis=2)
        mask_to_show = mask_to_show.astype(np.uint8) * 255

        if create_masks:
            cv2.imwrite("data/" + item + "/masks/" + frame_name + ".png", mask_to_show)

        frame = cv2.addWeighted(frame, 0.5, mask_to_show, 0.5, 0)

        cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 2)
        # cv2.rectangle(frame, new_bbox[:2], new_bbox[2:], (0, 0, 255), 2)
        cv2.rectangle(frame, expanded_bbox[:2], expanded_bbox[2:], (0, 0, 255), 2)
        cv2.putText(
            frame,
            str(frame_num).zfill(4),
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        expanded_bbox_size = (expanded_bbox[2] - expanded_bbox[0]) * (
            expanded_bbox[3] - expanded_bbox[1]
        )
        new_bbox_size = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])

        bbox = expanded_bbox
        cv2.imshow("Tracking", frame)
        frames_names.append("img_" + str(frame_num).zfill(4) + ".png")
        frames_bboxes.append(bbox)
        new_frames.append(frame)

        # Press Q on keyboard to  exit
        k = cv2.waitKey(step)
        if k == ord("q"):
            break

    vidcap.release()
    cv2.destroyAllWindows()
    # Save video
    height, width, layers = new_frames[0].shape
    size = (width, height)
    fps = 30
    os.makedirs("data/" + item + "/videos/", exist_ok=True)
    out = cv2.VideoWriter(
        "data/" + item + "/videos/" + item + "_bbox.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )

    new_frames = []
    for file in tqdm(
        sorted(os.listdir("data/" + item + "/images/")),
        desc="Saving video",
        unit="frame",
    ):
        frame = cv2.imread("data/" + item + "/images/" + file)
        with open("data/" + item + "/inputs/" + file[:-4] + ".json", "r") as f:
            bbox = json.load(f)[0]["bbox_modal"]
        mask = cv2.imread("data/" + item + "/masks/" + file)
        frame = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)

        cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 2)
        cv2.putText(
            frame,
            file[:-4],
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)

    out.release()


class Tracker:
    def __init__(self) -> None:
        self.sam = init_sam()
        self.bbox = None
        self.frame_num = 0
        self.start_frame = 0
        self.stop_frame = None

        self.init_bbox = None
        self.item = None
        self.frame = None

    def set_frame(self, frame: np.ndarray) -> None:
        self.frame = frame

    def set_frame_num(self, frame_num: int) -> None:
        self.frame_num = frame_num

    def extract_coordinates(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.init_bbox = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.init_bbox.extend([x, y])
            self.bbox = np.array(self.init_bbox)

    def set_bbox(self) -> None:
        cv2.namedWindow("input_bbox", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("input_bbox", self.extract_coordinates)

        while True:
            frame2show = self.frame.copy()
            key = cv2.waitKey(1)
            if key == 13:
                break
            elif key in LABELS_NUMS_KEY:
                self.item = LABELS[key - ASCII_NUM_SHIFT]

            if self.item:
                cv2.putText(
                    frame2show,
                    f"Selected: {self.item}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if self.bbox:
                cv2.rectangle(
                    frame2show,
                    tuple(self.bbox[:2]),
                    tuple(self.bbox[2:]),
                    (0, 255, 0),
                    2,
                )
                if self.item:
                    cv2.putText(
                        frame2show,
                        f"Selected: {self.item}",
                        tuple(self.bbox[:2]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            cv2.imshow("input_bbox", frame2show)

        cv2.destroyAllWindows()
        print(f"[INFO]: Selected bbox: {self.bbox}")

    def get_next_frame_images(self) -> np.ndarray:
        if self.frame_num == 0 or self.frame_num == self.start_frame:
            self.images = sorted(os.listdir("data/" + self.item + "/images/"))
            self.num_of_frames = len(self.images)

        frame = cv2.imread(
            "data/" + self.item + "/images/" + self.images[self.frame_num]
        )
        self.frame_num += 1

        return frame

    def get_next_frame_video(self) -> np.ndarray:
        if self.frame_num == 0 or self.frame_num == self.start_frame:
            self.vidcap = cv2.VideoCapture(self.path2video)
            self.num_of_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame = self.vidcap.read()

        if ret:
            self.frame_num += 1
            return frame

    def track(self, type: str = "images") -> np.ndarray:
        if type == "images":
            get_next_frame = self.get_next_frame_images

        elif type == "video":
            get_next_frame = self.get_next_frame_video

        # TODO:Replace
        camera_dict = json.load(open(args.camera_dict, "r"))
        mtx = np.array(camera_dict["camera_matrix"])
        dist = np.array(camera_dict["distortion_coefficients"])
        h = camera_dict["height"]
        w = camera_dict["width"]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        x, y, w, h = roi
        camera_data = {
            "K": newcameramtx.tolist(),
            "resolution_format": "height_width for some reason",
            "resolution": [h, w],
        }

        with open("data/" + self.item + "/camera_data.json", "w") as f:
            json.dump(camera_data, f, indent=2)

        frames = []
        while True:
            frame = get_next_frame()

            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            frame = frame[y : y + h, x : x + w]

            frames.append(frame)
            self.sam.set_image(frame)
            mask, scores, logits = self.sam.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(self.bbox),
                multimask_output=False,
            )
            eroded_mask = erode_mask(mask)
            new_bbox = get_bbox_from_mask(eroded_mask)

            bbox_expanded = expand_bbox(new_bbox, 1.1, frame.shape[:2])
            frame_name = "img_" + str(self.frame_num).zfill(4)

            new_label = {"label": self.item, "bbox_modal": bbox_expanded}

            json_path = "data/" + self.scene + "/inputs/" + frame_name + ".json"
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    new_label = json.load(f)
                new_label.append(new_label)

                with open(json_path, "w") as f:
                    json.dump(new_label, f, indent=2)

            else:
                with open(json_path, "w") as f:
                    json.dump([new_label], f, indent=2)

        output_video = cv2.VideoWriter(
            "data/" + self.item + "/videos/" + self.item + "_bbox.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            size,
        )

        for frame in tqdm(frames):
            output_video.write(frame)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--init_bbox", nargs="+", type=int, required=True)
    args.add_argument("--path2video", type=str, required=True)
    args.add_argument("--item", type=str, required=True)
    args.add_argument("--step", type=int, default=1)  # 0 to go frame by frame
    args.add_argument("--stop", type=int, default=None)
    args.add_argument("--masks", type=bool, default=False)
    args.add_argument("--camera_dict", type=str, default="data/camera_parameters.json")

    args = args.parse_args()
    if len(args.init_bbox) != 4:
        raise ValueError(
            f"[ERROR]: init_bbox must have 4 elements, got {len(args.init_bbox)}"
        )

    camera_dict = json.load(open(args.camera_dict, "r"))
    track_and_cut(
        args.init_bbox,
        args.path2video,
        args.item,
        args.step,
        args.stop,
        args.masks,
        camera_dict,
    )
    # 740 250 1030 800
