import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import argparse
import random as rng

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# TODO: Add parsing of arguments
# TODO: Add saving of annotations
# TODO: Add saving masks with color and stuff


class MaskByPromt:
    def __init__(self, image_dir: str, save_folder: str) -> None:
        self.image_dir = image_dir
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.save_folder_imgs = os.path.join(self.save_folder, "images")
        os.makedirs(self.save_folder_imgs, exist_ok=True)
        self.save_folder_masks = os.path.join(self.save_folder, "masks")
        os.makedirs(self.save_folder_masks, exist_ok=True)
        self.save_folder_cutouts = os.path.join(self.save_folder, "cutouts")
        os.makedirs(self.save_folder_cutouts, exist_ok=True)
        print(f"[INFO]: Saving folder is {self.save_folder}")

        self.images = sorted(os.listdir(self.image_dir))
        self.max_images = len(self.images)
        self.promt_points = []
        self.labels = []
        self.input_box = None
        self.current_image_idx = 0
        self.change_image = True
        self.image = None
        self.image_to_show = None
        self.selected_label = 1
        self.binary_mask = None
        self.bbox = None
        self.resolution = None
        self.inputig_bbox = False
        self.input_bbox = None
        self.drawing_bbox = False

        self.label_int = 255

        self.color_dict = {
            -3: (0, 255, 0),  # Include GREEN
            -2: (0, 0, 255),  # Exclude RED
            -1: (0, 0, 0),  # Unlabeled BLACK
            255: (255, 255, 255),  # Unlabeled WHITE
        }

        # SAM
        print("[INFO]: Loading SAM model...")
        sam_checkpoint = "segment_anything/checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.sam_predictor = SamPredictor(sam)
        print(f"[INFO]: SAM model loaded; Running on {device}")
        print("[INFO]: Press h for help")

    def visualize_promt(self):
        self.image_to_show = self.image.copy()
        for point, label in zip(self.promt_points, self.labels):
            if label == 0:  # Exclude
                color = (0, 0, 255)
            else:  # Include
                color = (0, 255, 0)

            if min(self.resolution) < 1000:
                thickness = 5
            else:
                thickness = 10
            cv2.circle(self.image_to_show, tuple(point), thickness, color, -1)

        if self.binary_mask is not None:
            mask_to_show = np.tile(
                self.binary_mask[:, :, None], (1, 1, 3)
            )  # Transform to 3 channel image

            if self.label_int in self.color_dict.keys(): # Already assigned color
                color = self.color_dict[self.label_int]
            else:
                color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
                while color in self.color_dict.values(): # Ensure unique color
                    color = (
                        rng.randint(0, 255),
                        rng.randint(0, 255),
                        rng.randint(0, 255),
                    )
                self.color_dict[self.label_int] = color

            mask_to_show[:, :, 0] = mask_to_show[:, :, 0] * color[0]
            mask_to_show[:, :, 1] = mask_to_show[:, :, 1] * color[1]
            mask_to_show[:, :, 2] = mask_to_show[:, :, 2] * color[2]

            self.image_to_show = cv2.addWeighted(
                self.image_to_show, 1, mask_to_show, 0.7, 0
            )

        if self.input_bbox is not None and len(self.input_bbox) == 4:
            x, y, x2, y2 = self.input_bbox
            cv2.rectangle(self.image_to_show, (x, y), (x2, y2), (0, 255, 0), 5 // 2)

        if self.bbox is not None:
            x, y, w, h = self.bbox
            cv2.rectangle(
                self.image_to_show, (x, y), (x + w, y + h), (255, 0, 0), 5 // 2
            )

        cv2.imshow("image", self.image_to_show)

    def mouse_click(self, event, x, y, flags, param):
        # Add Bbox somehow
        self.change_image = True
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(f"Clicked at {x}, {y}")
            if self.inputig_bbox:
                self.drawing_bbox = True
                self.input_bbox = [x, y]
            else:
                self.promt_points.append([x, y])
                self.labels.append(self.selected_label)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_bbox:
            # TODO: Implement drawing of bbox on image
            pass

        elif event == cv2.EVENT_LBUTTONUP:
            if self.inputig_bbox:
                self.input_bbox.extend([x, y])
                self.drawing_bbox = False

        elif event == cv2.EVENT_MBUTTONDOWN:
            if len(self.promt_points) != 0:
                # print(f"Removed point {self.promt_points[-1]}")
                _ = self.promt_points.pop()
                _ = self.labels.pop()
                print(self.promt_points)
        self.visualize_promt()

    def load_image(self):
        print(f"[INFO]: Loading image {self.images[self.current_image_idx]}")
        self.image = cv2.imread(
            os.path.join(self.image_dir, self.images[self.current_image_idx])
        )
        self.image_to_show = self.image.copy()
        self.promt_points = []
        self.labels = []
        self.binary_mask = None
        self.selected_label = 1
        self.visualize_promt()

    def save(self):
        if self.binary_mask is None:
            print("[INFO]: No mask to save, NOT SAVING")
        else:
            # print(f"Saving promt {self.images[self.current_image_idx]}")
            promt_path = os.path.join(
                self.save_folder_imgs, self.images[self.current_image_idx]
            )
            cv2.imwrite(promt_path, self.image)
            mask_path = os.path.join(
                self.save_folder_masks,
                self.images[self.current_image_idx].split(".")[0] + ".png",
            )
            cv2.imwrite(mask_path, self.binary_mask * 255)
            cutout_path = os.path.join(
                self.save_folder_cutouts, self.images[self.current_image_idx]
            )
            cutout = cv2.bitwise_and(self.image, self.image, mask=self.binary_mask)
            for label, point in zip(self.labels, self.promt_points):
                if label == 0:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.circle(cutout, tuple(point), 10, color, -1)
            cv2.imwrite(cutout_path, cutout)

    def next_image(self):
        self.current_image_idx += 1
        if self.current_image_idx >= self.max_images:
            print("[INFO]: No more images")
            self.current_image_idx = self.max_images - 1
        self.load_image()

    def previous_image(self):
        self.current_image_idx -= 1
        if self.current_image_idx < 0:
            print("[INFO]: First image")
            self.current_image_idx = 0
        self.load_image()

    def postprocess_mask(self):
        self.binary_mask = cv2.dilate(self.binary_mask, None, iterations=3)
        self.binary_mask = cv2.erode(self.binary_mask, None, iterations=3)

    def mask_image(self):
        # NOTE: UNUSED
        self.sam_predictor.set_image(self.image)
        if self.input_box is not None:
            print("[INFO]: Masking image with input box")
            mask, scores, logits = self.sam_predictor.predict(
                point_coords=np.array(self.promt_points),
                point_labels=np.array(self.labels),
                box=np.array(self.input_box),
                multimask_output=True,
            )
            mask_input = logits[np.argmax(scores), :, :]
            masks, _, _ = self.sam_predictor.predict(
                point_coords=np.array(self.promt_points),
                point_labels=np.array(self.labels),
                mask_input=mask_input[None, :, :],
                box=np.array(self.input_box),
                multimask_output=False,
            )
            cv2_mask = np.array(masks[0, :, :])
            cv2_mask = cv2_mask.astype(np.uint8)
            self.binary_mask = cv2_mask
            self.postprocess_mask()
        else:
            print("[INFO]: Masking image")
            mask, scores, logits = self.sam_predictor.predict(
                point_coords=np.array(self.promt_points),
                point_labels=np.array(self.labels),
                multimask_output=True,
            )
            mask_input = logits[np.argmax(scores), :, :]
            masks, _, _ = self.sam_predictor.predict(
                point_coords=np.array(self.promt_points),
                point_labels=np.array(self.labels),
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )
            cv2_mask = np.array(masks[0, :, :])
            cv2_mask = cv2_mask.astype(np.uint8)
            self.binary_mask = cv2_mask
            self.postprocess_mask()

    def proccess_key(self, key: int) -> bool:
        """Process key press

        Args:
            key (int): key pressed

        Returns:
            bool: True if should quit
        """
        if key == ord("h"):  # Help
            # TODO: UPDATE
            print("[INFO]: Help Keybindings")
            print("\th: Help")
            print("\tq or esc: Quit")
            print("\td: Next image")
            print("\ta: Previous image")
            print("\t1: Include point")
            print("\t0: Exclude point")
            print("\tx: Switch include/exclude point")
            print("\tm or space: Mask image")
            print("\ts: Save promt")
            print("\tr: Reset promt")
            print("\tc: Dilate mask")
            print("\tv: Erode mask")
            print("\te: Visualize mask as cropped image")
            print("\tb: Generate bounding box from mask")

            print("[INFO]: Mouse Keybindings")
            print("\tleft mouse click: Add point")
            print("\tmiddle mouse click: Remove point")

        elif key == ord("q") or key == 27:  # Quit program (q or esc)
            print("[INFO]: Quitting program")
            return True

        elif key in [ord("d"), 83]:  # Next imagen, right arrow
            self.next_image()

        elif key in [ord("a"), 81]:  # Previous imagee, left arrow
            self.previous_image()

        elif key == 82:  # Up arrow
            print("[INFO]: Up arrow")
            self.label_int += 1
            if self.label_int > 255:
                self.label_int = 1
            print(f"[INFO]: Label int is {self.label_int}")

        elif key == 84:  # Down arrow
            print("[INFO]: Down arrow")
            self.label_int -= 1
            if self.label_int < 1:
                self.label_int = 255
            print(f"[INFO]: Label int is {self.label_int}")

        elif key in [ord("1"), ord("i")]:  # Select label 1
            print("[INFO]: Include point")
            self.inputig_bbox = False
            self.selected_label = 1
        elif key in [ord("0"), ord("e")]:  # Select label 2
            print("[INFO]: Exclude point")
            self.inputig_bbox = False
            self.selected_label = 0

        elif key == ord("x"):  # Switch label
            self.inputig_bbox = False
            if self.selected_label == 1:
                print("[INFO]: Exclude point")
                self.selected_label = 0
            else:
                print("[INFO]: Include point")
                self.selected_label = 1

        elif key == ord("b"):  # Input bounding box
            print("[INFO]: Input bounding box")
            self.inputig_bbox = True

        elif key == ord("m") or key == 32:  # Mask image (m or space)
            print("[INFO]: Masking image")

            self.sam_predictor.set_image(self.image)

            if self.promt_points is None or len(self.promt_points) == 0:
                print("[INFO]: No promt points, skipping")
                points = None
            else:
                points = np.array(self.promt_points)

            if self.labels is None or len(self.labels) == 0:
                print("[INFO]: No labels, skipping")
                labels = None
            else:
                labels = np.array(self.labels)

            if self.input_bbox is None or len(self.input_bbox) != 4:
                print("[INFO]: No input box, skipping")
                box = None
            else:
                box = np.array(self.input_bbox)
                box = box[None, :]

            print(f"Points: {points}")
            print(f"Labels: {labels}")
            print(f"Box: {box}")

            mask, scores, logits = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=True,
            )
            mask_input = logits[np.argmax(scores), :, :]

            masks, _, _ = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )
            cv2_mask = np.array(masks[0, :, :])
            cv2_mask = cv2_mask.astype(np.uint8)
            self.binary_mask = cv2_mask
            self.visualize_promt()

        elif key == ord("l"):  # Dilate mask
            print("[INFO]: Dilating mask")
            self.binary_mask = cv2.dilate(self.binary_mask, None, iterations=1)
            self.visualize_promt()

        elif key == ord("k"):  # Erode mask
            print("[INFO]: Eroding mask")
            self.binary_mask = cv2.erode(self.binary_mask, None, iterations=1)
            self.visualize_promt()

        elif key == ord("c"):  # Visualize mask as cropped image
            print("[INFO]: Visualize mask as cropped image")
            if self.binary_mask is None:
                print("[INFO]: No mask to visualize")
            else:
                cutout = cv2.bitwise_and(self.image, self.image, mask=self.binary_mask)
                cv2.imshow("cutout", cutout)

        elif key == ord("n"):  # Generate bounding box from mask
            print("[INFO]: Generate bounding box from mask")
            if self.binary_mask is None:
                print("[INFO]: No mask to generate bounding box")
            else:
                contours, hierarchy = cv2.findContours(
                    self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cnt = contours[0]
                x, y, w, h = cv2.boundingRect(cnt)
                self.bbox = (x, y, w, h)
                # TODO: Add saving bounding box data to file + add classification by hand

        elif key == ord("s"):  # Save promt
            self.save()
            self.next_image()

        elif key == ord("r"):  # Reset promt
            print("[INFO]: Reset prompt")
            self.promt_points = []
            self.labels = []
            self.binary_mask = None
            self.image_to_show = self.image.copy()
            self.bbox = None
            self.input_bbox = None

        else:
            print(f"[WARN] Unknown key press with id {key}")

        cv2.imshow("image", self.image_to_show)

        return False

    def mask_by_promt(self):
        """Open image and mask it by promt

        Args:
            image_dir (str): path to folder with images
        """
        cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cutout", int(2048 / 2), int(2448 / 2))

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", int(2048 / 2), int(2448 / 2))
        cv2.setMouseCallback("image", self.mouse_click)

        self.image = cv2.imread(
            os.path.join(self.image_dir, self.images[self.current_image_idx])
        )
        self.image_to_show = self.image.copy()
        self.resolution = self.image.shape[:2]
        cv2.imshow("cutout", np.zeros_like(self.image_to_show))

        cv2.imshow("image", self.image_to_show)
        while True:
            key = cv2.waitKey(0)

            should_quit = self.proccess_key(key)
            if should_quit:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    mask_by_promt = MaskByPromt("data/images", "data/images/masks")
    mask_by_promt.current_image_idx = 0
    # SUGAR skončil jsem u 206 205 hotová
    mask_by_promt.mask_by_promt()
