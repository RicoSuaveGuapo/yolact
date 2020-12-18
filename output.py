import os
import getpass
import torch
from time import time

config   = 'yolact_base_config'
kind = 'U100'
assert kind in ['A30','U100','U150','All']

uid = getpass.getuser()
if uid == 'rico-li':
    # pth_path = 'reserve_weight/300_images/yolact_base_42934_128802_interrupt.pth' # 300 labels result
    # pth_path = 'weights/yolact_base_1249_60000.pth'       # loss ratio (cls,box,mask)  1:1.5:6.125
    # pth_path = 'weights/yolact_base_2777_133333.pth'      # only U100/U150 loss ratio (cls,box,mask) 1:  3:6.125
    pth_path = 'weights/yolact_base_454_30000.pth'          # with all kinds and loss ratio 1:  6:6.125
    # pth_path = 'weights/yolact_base_4761_114285.pth'      # with only A30 and loss ratio 1:  6:6.125
    # pth_path = 'weights/yolact_plus_base_1731_114285.pth' # all kinds yolact++ and 1:6:6.125
    
    # Quantitative evaluation
    if kind != 'All':
        val_dir = f'/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/{kind}/annotations/yolact_val/JPEGImages'
        output_dir = f'/home/rico-li/Job/豐興鋼鐵/Prediction/{kind}/{os.path.basename(pth_path)[:-4]}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    else:
        val_dir = f'/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/yolact_val/JPEGImages'
        output_dir = f'/home/rico-li/Job/豐興鋼鐵/Prediction/{kind}/{os.path.basename(pth_path)[:-4]}/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

# for dell
elif uid == 'aiuser':
    pth_path = '/home/aiuser/Job/yolact/weights/yolact_base_4166_133333.pth'
    val_dir = '/home/aiuser/Job/yolact/data/metal_data/data/clean_data_20frames/U100/annotations/yolact_val/JPEGImages'
    output_dir = '/home/aiuser/Job/yolact/data/metal_data/data/clean_data_20frames/U100/predictions/4000_val'
# for dgx
elif uid == 'root':
    pth_path = '/nfs/Workspace/yolact_base_4166_133333.pth'
    val_dir = '/nfs/Workspace/clean_data_20frames/U100/annotations/yolact_val/JPEGImages'
    output_dir = '/nfs/Workspace/clean_data_20frames/U100/predictions/4000_val'

# single image evaluation
# make sure that the image is cropped
# img_path = '/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/annotations/yolact_train/JPEGImages/mod_1500_curve_3_frame0551.jpg'
# out_name = 'metal_img.png'


# ===== train =====
# Local
# os.system('python -W ignore train_gan_server.py --config=yolact_base_config --dataset metal2020_dataset --batch_size=1 --validation_epoch=16')

# Server Dell
# python -W ignore train.py --config=yolact_base_config --batch_size=24 --batch_alloc=24 --dataset metal2020_server_dataset --validation_epoch=16
# Nohup
# nohup python -W ignore train.py --config=yolact_base_config --batch_size=24 --batch_alloc=24 --dataset metal2020_server_dataset --validation_epoch=16 > train.log 2>&1 &

# Server DGX
# os.system("python -W ignore train.py --config=yolact_base_config --batch_size=48 --batch_alloc=24,24 --dataset metal2020_server_dgx_dataset --validation_epoch=16")
# Nohup
# os.system("nohup python -W ignore train.py --resume --config=yolact_base_config --batch_size=48 --batch_alloc=24,24 --dataset metal2020_server_dgx_dataset --validation_epoch=16 > train.log 2>&1 & ")

# resume training
os.system(f"python -W ignore train_gan_server.py --config=yolact_base_config --dataset metal2020_dataset --batch_size=2 --resume {pth_path} --validation_epoch=16")


# ===== eval =====
# Quantitative evaluation
# os.system(f"python -W ignore eval.py --max_images 1 --no_bar --trained_model={pth_path} --config {config} --display_fps --fast_nms False")
#
# Output json file
# os.system(f"python -W ignore eval.py --trained_model={pth_path} --output_coco_json")
# eval, output img
# os.system(f'python -W ignore eval.py --trained_model={pth_path} --score_threshold=0.15 --top_k=15 --image={img_path}:{out_name} --config {config}')

# Evaluate on a folder
# counts = os.listdir(f'{val_dir}')
# start = time()
# os.system(f"python -W ignore eval.py --trained_model={pth_path} --score_threshold=0.1 --top_k=15 --images={val_dir}:{output_dir} --config {config} --fast_nms False")
# print(f'\nSpends {time()-start:.2f} sec')
# print(f'--- {len(counts)/(time()-start):.2f} fps ---')

# Evaluate on a single image
# os.system(f'python -W ignore eval.py --trained_model={pth_path} --fast_nms False --score_threshold=0.1 --top_k=15 --config {config} --image /home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/A30/images/val/1555_curve_13_frame0314.jpg')
