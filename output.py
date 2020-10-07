import os

pth_path = 'weights/yolact_base_61_2144_interrupt.pth'
# make sure that the image is cropped
img_path = '/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/annotations/yolact_train/JPEGImages/mod_1500_curve_3_frame0551.jpg'
out_name = 'metal_img.png'
config   = 'yolact_base_config'

# train
# Local
# python -W ignore train.py --config=yolact_base_config --dataset metal2020_dataset --batch_size=5
# Server
# nohup python -W ignore train.py --config=yolact_base_config --batch_size=48 --batch_alloc=24,24 --dataset metal2020_server_dataset --validation_epoch=16 > train.log 2>&1 &

# Quantitative evaluation
# os.system(f"python -W ignore eval.py --trained_model={pth_path} --config {config}")
#
# Output json file
# os.system(f"python -W ignore eval.py --trained_model={pth_path} --output_coco_json")

# eval, output img
# os.system(f'python -W ignore eval.py --trained_model={pth_path} --score_threshold=0.15 --top_k=15 --image={img_path}:{out_name} --config {config}')


# Evaluate on a folder
# from time import time
# counts = os.listdir('/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/annotations/yolact_val/JPEGImages')
# start = time()
# os.system("python -W ignore eval.py --trained_model=weights/yolact_base_42934_128802_interrupt.pth --score_threshold=0.15 --top_k=15 --images=/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/annotations/yolact_val/JPEGImages:/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/predictions/val")
# print(f'\nSpends {time()-start:.2f} sec')
# print(f'--- {len(counts)/(time()-start):.2f} fps ---')
# resume training
# python -W ignore train.py --config=yolact_base_config --dataset metal2020_dataset --batch_size=5 --resume weights/yolact_base_267_123_interrupt.pth
pass