#! /usr/bin/bash

path2video=data/input_videos/Servo2.mp4
item=Servo
step=1 # 0: manual, 1: auto
create_masks=True # True, False save masks for the object
camera_params_path=data/camera_parametersP7Af50.json
# --init_bbox 720 1365 350 790 controller.mp4
# --init_bbox 740 300 1200 760 battery.mp4
# --init_bbox 780 420 1090 780 main.mp4
# --init_bbox 900 460 1080 790 motor.mp4
# --init_bbox 950 240 1150 460 servo1.mp4
# --init_bbox 890 450 1100 680 servo2.mp4

# --init_bbox 783 244 1150 580 Battery2.mp4
# --init_bbox 770 130 1150 740 Chassis2
# --init_bbox 666 300 1075 580 Controller2.mp4
# --init_bbox 575 105 1000 466 Main2.mp4


/home/vit/anaconda3/envs/SAM/bin/python /home/vit/Documents/DP/segment-anything/scripts/tracker.py --init_bbox 710 270 970 520 --path2video $path2video --item $item --masks $create_masks --step $step --camera_dict $camera_params_path --stop -1
