import json
import pickle
from eval import *

# val total 1051 images

# ground truth
with open('/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/annotations/yolact_val/annotations.json') as json_file:
    gt_bbox = json.load(json_file)
    # dict
    # dict_keys(['info', 'licenses', 'images', 'type', 'annotations', 'categories'])
    # 'info': {'description': None, 'url': None, 'version': None, 'year': 2020, 'contributor': None, 'date_created': '2020-10-08 14:34:32.871654'}
    # 'licenses': [{'url': None, 'id': 0, 'name': None}]
    # 'images': [{'license': 0, 'url': None, 'file_name': 'JPEGImages/mod_1745_3_curve_frame1896.jpg', 'height': 410, 'width': 700, 'date_captured': None, 'id': 0}, 
    # {'license': 0, 'url': None, 'file_name': 'JPEGImages/mod_1518_ok_3_frame2707.jpg', 'height': 410, 'width': 700, 'date_captured': None, 'id': 1},...]
    # NOTE
    # -> with 1051 items
    # 'type':instance
    # annotations: [{'id': 0, 'image_id': 0, 'category_id': 3, 'segmentation': [[367.35, 7.20, 380.82, 7.72,... ]], 'area': 7207.0, 'bbox': [227.0, 7.0, 154.0, 268.0], 'iscrowd': 0},
    # {'id': 1, 'image_id': 0, 'category_id': 3, 'segmentation': [[...]], 'area': 7266.0, 'bbox': [...], 'iscrowd': 0},...]
    # NOTE
    # -> with 3844 annotations
    # 'categories'
    # [{'supercategory': None, 'id': 0, 'name': 'background'}, 
    # {'supercategory': None, 'id': 1, 'name': 'U150_curve'}, 
    # {'supercategory': None, 'id': 2, 'name': 'U150_ok'}, 
    # {'supercategory': None, 'id': 3, 'name': 'U100_curve'}, 
    # {'supercategory': None, 'id': 4, 'name': 'U100_ok'}, 
    # {'supercategory': None, 'id': 5, 'name': 'A30_curve'}, 
    # {'supercategory': None, 'id': 6, 'name': 'A30_ok'}]


# detection
# from the code of 
# detections.add_bbox(image_id, classes[i], boxes[i,:],   box_scores[i])
# detections.add_mask(image_id, classes[i], masks[i,:,:], mask_scores[i])
# scores = [scores, scores * maskiou_p]
# scores here is the confident level

# bbox
with open('results/bbox_detections.json') as json_file:
    pred_bbox = json.load(json_file)
    # list
    # [{'image_id': 0, 'category_id': 3, 'bbox': [228.0, 7.0, 152.0, 263.0], 'score': 0.99}, {...}...]
    # total 18624

# mask
with open('results/mask_detections.json') as json_file:
    pred_mask = json.load(json_file)
    # list
    # [{'image_id': 0, 'category_id': 3, 'segmentation': {'size': [410, 700], 
    # 'counts': '...0I8N101O1O1H8N2N3O0NhSP4'}, 'score': 0.99},...]
    # total 18624

# print(gt_bbox['categories'])
# print(len(pred_bbox))
# print(pred_mask[1])

with open('results/ap_data.pkl','rb') as pkl:
     data = pickle.load(pkl)

# ap_data.pkl structure
# 
# 'box', 'mask'
# IoU (50,55,...95) total 10
# classes, total 6
# A30_curve, A30_ok, U100_curve, U100_ok, U150_curve, U150_ok

recalls, precisions, scores = data['box'][0][2].get_recall()

score_sep = []
pred_sep = []
recall_sep =[]
for i in range(0,5):
    score_gt = [score for score in scores if score >= (i+5)*0.1][-1]
    recall_gt = [recalls[j] for j, score in enumerate(scores) if score >= (i+5)*0.1][-1]
    pred_gt = [precisions[j] for j, score in enumerate(scores) if score >= (i+5)*0.1][-1]
    score_sep.append(score_gt)
    recall_sep.append(recall_gt)
    pred_sep.append(pred_gt)

score_sep = [round(score,3) for score in score_sep]
recall_sep = [round(recall,3) for recall in recall_sep]
pred_sep = [round(pred,3) for pred in pred_sep]
print(f'Confidence: {score_sep}')
print(f'Recall    : {recall_sep}')
print(f'Precision : {pred_sep}')

# threshold_bbox = [bbox for bbox in pred_bbox if bbox['score'] > 0.9]
# print(len(threshold_bbox))
