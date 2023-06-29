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

from mask_by_promt import MaskByPromt
# TODO: zkusit to s bboxama

class MasKAutomaticPrompt(MaskByPromt):
    def __init__(self, image_dir:str, save_folder:str) -> None:
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

        
        self.images = os.listdir(self.image_dir)
        self.max_images = len(self.images)
        self.promt_points = []
        self.labels = []
        self.current_image_idx = 0
        self.change_image = True
        self.image = None
        self.image_to_show = None
        self.selected_label = 1
        self.binary_mask = None
        self.bbox = None

        #SAM
        print("[INFO]: Loading SAM model...")
        sam_checkpoint = "../segment_anything/checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.sam_predictor = SamPredictor(sam)
        print("[INFO]: SAM model loaded")
        print("[INFO]: Press h for help")

    def procces_all_images(self):
        offset = 100
        # self.promt_points = [   [int(2047/2), int(2447/2)],
        #                         [int(2047/2)-offset, int(2447/2)], 
        #                         [int(2047/2)+offset, int(2447/2)],
        #                         [int(2047/2), int(2447/2)-offset],
        #                         [int(2047/2), int(2447/2)+offset],
        #                         [0,0],
        #                         [0, 2447],
        #                         [2047, 0],
        #                         [2047, 2447]]
        # self.labels = [1, 1, 1, 1, 1, 0, 0, 0, 0]
        for i in tqdm(range(self.max_images)):
            self.promt_points = [
                                [int(2447/2), int(2047/2)],
                                [int(2447/2)-offset, int(2047/2)], 
                                [int(2447/2)+offset, int(2047/2)],
                                [int(2447/2), int(2047/2)-offset],
                                [int(2447/2), int(2047/2)+offset],
                                [int(2447/2)-offset, int(2047/2)-offset],
                                [int(2447/2)+offset, int(2047/2)+offset],
                                [int(2447/2)-offset, int(2047/2)+offset],
                                [int(2447/2)+offset, int(2047/2)-offset],
                                [0+ offset,0 + offset],
                                [0+offset, 2047-offset],
                                [2447-offset, 0+offset],
                                [2447-offset, 2047-offset]
                                ]
            self.input_box = None
            # self.input_box = [0, 0, 2447, 2047] # Worse results
            self.labels = [1, 1, 1, 1, 1, 1, 1,1,1, 0, 0, 0, 0]
            # self.labels = [1, 0, 0, 0, 0]
            self.current_image_idx = i
            self.image = cv2.imread(os.path.join(self.image_dir, self.images[i]))
            self.image_to_show = self.image.copy()
            
            self.mask_image()
            self.save()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    mask_automatic_prompt = MasKAutomaticPrompt("../../3Dreconstuct/02_cracker_box/images/", "../../3Dreconstuct/02_cracker_box_masked2/")
    mask_automatic_prompt.procces_all_images()
    # mask_by_promt = MaskByPromt('../../3Dreconstuct/03_sugar_box/images/', '../../3Dreconstuct/03_sugar_box_v2/')
    # mask_by_promt.mask_by_promt()
