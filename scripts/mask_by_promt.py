import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

class MaskByPromt():
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

    def visualize_promt(self):
        self.image_to_show = self.image.copy()
        for point,label in zip(self.promt_points,self.labels):
            if label == 0:
                color = (0,0,255) 
            else:
                color = (0,255,0)
            cv2.circle(self.image_to_show, tuple(point), 10, color, -1)

        if self.binary_mask is not None:
            mask_to_show = np.tile(self.binary_mask[:,:,None], (1,1,3))*255
            self.image_to_show = cv2.addWeighted(self.image_to_show, 1, mask_to_show, 0.5, 0)

        if self.bbox is not None:
            x,y,w,h = self.bbox
            cv2.rectangle(self.image_to_show,(x,y),(x+w,y+h),(0,255,0),5)

        cv2.imshow("image", self.image_to_show)

    def mouse_click(self, event, x, y, flags, param):
        self.change_image = True
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(f"Clicked at {x}, {y}")
            self.promt_points.append([x, y])
            self.labels.append(self.selected_label)
            
        elif event == cv2.EVENT_MBUTTONDOWN:
            if len(self.promt_points) != 0:
                # print(f"Removed point {self.promt_points[-1]}")
                _ = self.promt_points.pop()
                _ = self.labels.pop()
                print(self.promt_points)
        self.visualize_promt()

    def load_image(self):
        print(f"[INFO]: Loading image {self.images[self.current_image_idx]}")
        self.image = cv2.imread(os.path.join(self.image_dir, self.images[self.current_image_idx]))
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
            print(f"Saving promt {self.images[self.current_image_idx]}")
            promt_path = os.path.join(self.save_folder_imgs, self.images[self.current_image_idx])
            cv2.imwrite(promt_path, self.image)
            mask_path = os.path.join(self.save_folder_masks, self.images[self.current_image_idx].split(".")[0]+".png")
            cv2.imwrite(mask_path, self.binary_mask*255)
            cutout_path = os.path.join(self.save_folder_cutouts, self.images[self.current_image_idx])
            cutout = cv2.bitwise_and(self.image, self.image, mask=self.binary_mask)
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


    def proccess_key(self, key:int)->bool:
        """Process key press

        Args:
            key (int): key pressed

        Returns:
            bool: True if should quit
        """        
        if key == ord('h'): # Help
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

        elif key == ord('q') or key == 27: # Quit program (q or esc)
            print("[INFO]: Quitting program")
            return True    
        
        elif key == ord('d'): # Next imagen
            self.next_image()
    
        elif key == ord('a'): # Previous imagewe
            self.previous_image()

        elif key == ord('1'): # Select label 1
            print("[INFO]: Include point")
            self.selected_label = 1
        elif key == ord('0'): # Select label 2
            print("[INFO]: Exclude point")
            self.selected_label = 0
        
        elif key == ord('x'): # Switch label
            if self.selected_label == 1:
                print("[INFO]: Exclude point")
                self.selected_label = 0
            else:
                print("[INFO]: Include point")
                self.selected_label = 1

        elif key == ord('m') or key == 32: # Mask image (m or space)
            print("[INFO]: Masking image")
            self.sam_predictor.set_image(self.image)
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
            cv2_mask = np.array(masks[0,:,:])
            cv2_mask = cv2_mask.astype(np.uint8)
            self.binary_mask = cv2_mask
            self.visualize_promt()
        
        elif key == ord('c'): # Dilate mask
            print("[INFO]: Dilating mask")
            self.binary_mask = cv2.dilate(self.binary_mask, None, iterations=1)
            self.visualize_promt()
        
        elif key == ord('v'): # Erode mask
            print("[INFO]: Eroding mask")
            self.binary_mask = cv2.erode(self.binary_mask, None, iterations=1)
            self.visualize_promt()

        elif key == ord('e'): # Visualize mask as cropped image
            print("[INFO]: Visualize mask as cropped image")
            if self.binary_mask is None:
                print("[INFO]: No mask to visualize")
            else:
                cutout = cv2.bitwise_and(self.image, self.image, mask=self.binary_mask)
                cv2.imshow("cutout", cutout)

        elif key == ord('b'): # Generate bounding box from mask
            print("[INFO]: Generate bounding box from mask")
            if self.binary_mask is None:
                print("[INFO]: No mask to generate bounding box")
            else:
                contours, hierarchy = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = contours[0]
                x,y,w,h = cv2.boundingRect(cnt)
                self.bbox = (x,y,w,h)
                # TODO: Add bounding box to promt
                # TODO: Add saving bounding box data to file + add classification by hand
                

        elif key == ord('s'): # Save promt
            self.save()
            self.next_image()

        elif key == ord('r'): # Reset promt
            print("[INFO]: Reset prompt")
            self.promt_points = []
            self.labels = []
            self.binary_mask = None
            self.image_to_show = self.image.copy()
            self.bbox = None
        
        cv2.imshow("image", self.image_to_show)
        
        return False

    def mask_by_promt(self):
        """Open image and mask it by promt

        Args:
            image_dir (str): path to folder with images
        """    
        cv2.namedWindow("cutout", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cutout", int(2048/2), int(2448/2))

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", int(2048/2), int(2448/2))
        cv2.setMouseCallback("image", self.mouse_click)

        

        self.image = cv2.imread(os.path.join(self.image_dir, self.images[self.current_image_idx]))
        self.image_to_show = self.image.copy()
        cv2.imshow("cutout", np.zeros_like(self.image_to_show))

        cv2.imshow("image", self.image_to_show)
        while True:
            key = cv2.waitKey(0)
            
            should_quit = self.proccess_key(key)
            if should_quit:
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    mask_by_promt = MaskByPromt('../../3Dreconstuct/03_sugar_box/images/', '../../3Dreconstuct/03_sugar_box_v2/')
    mask_by_promt.mask_by_promt()
