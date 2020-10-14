import json

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
print(pred_mask[1])
