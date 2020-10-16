import os
import getpass

config   = 'yolact_base_config'

uid = getpass.getuser()
if uid == 'rico-li':
    # pth_path = 'reserve_weight/300_images/yolact_base_42934_128802_interrupt.pth' # 300 labels result
    pth_path = '/home/rico-li/Job/豐興鋼鐵/yolact/weights/yolact_base_4166_133333.pth'
    # Quantitative evaluation
    val_dir = '/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/annotations/yolact_val/JPEGImages'
    output_dir = '/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/predictions/4000_val'

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
img_path = '/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/annotations/yolact_train/JPEGImages/mod_1500_curve_3_frame0551.jpg'
out_name = 'metal_img.png'


# train
# Local
# os.system('python -W ignore train.py --config=yolact_base_config --dataset metal2020_dataset --batch_size=5 --validation_epoch=1')

# Server Dell
# python -W ignore train.py --config=yolact_base_config --batch_size=24 --batch_alloc=24 --dataset metal2020_server_dataset --validation_epoch=16
# Nohup
# nohup python -W ignore train.py --config=yolact_base_config --batch_size=24 --batch_alloc=24 --dataset metal2020_server_dataset --validation_epoch=16 > train.log 2>&1 &

# Server DGX
# python -W ignore train.py --config=yolact_base_config --batch_size=48 --batch_alloc=24, 24 --dataset metal2020_server_dgx_dataset --validation_epoch=16
# Nohup
# nohup python -W ignore train.py --config=yolact_base_config --batch_size=48 --batch_alloc=24, 24 --dataset metal2020_server_dgx_dataset --validation_epoch=16 > train.log 2>&1 &

# resume training
# python -W ignore train.py --config=yolact_base_config --dataset metal2020_dataset --batch_size=5 --resume weights/yolact_base_267_123_interrupt.pth

# Quantitative evaluation
# os.system(f"python -W ignore eval.py --trained_model={pth_path} --config {config} --display_fps")
#
# Output json file
# os.system(f"python -W ignore eval.py --trained_model={pth_path} --output_coco_json")
# eval, output img
# os.system(f'python -W ignore eval.py --trained_model={pth_path} --score_threshold=0.15 --top_k=15 --image={img_path}:{out_name} --config {config}')

# Evaluate on a folder
# from time import time
# counts = os.listdir(f'{val_dir}')
# start = time()
# os.system(f"python -W ignore eval.py --trained_model={pth_path} --score_threshold=0.15 --top_k=15 --images={val_dir}:{output_dir} --config {config}")
# print(f'\nSpends {time()-start:.2f} sec')
# print(f'--- {len(counts)/(time()-start):.2f} fps ---')
