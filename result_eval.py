import json
import pickle
from eval import *
import matplotlib.pyplot as plt

# val total 1051 images

# ground truth
# with open('/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/U100/annotations/yolact_val/annotations.json') as json_file:
#     gt_bbox = json.load(json_file)
    # dict
    # dict_keys(['info', 'licenses', 'images', 'type', 'annotations', 'categories'])
    # 'info': {'description': None, 'url': None, 'version': None, 'year': 2020, 'contributor': None, 'date_created': '2020-10-08 14:34:32.871654'}
    # 'licenses': [{'url': None, 'id': 0, 'name': None}]
    # 'images': [{'license': 0, 'url': None, 'file_name': 'JPEGImages/mod_1745_3_curve_frame1896.jpg', 'height': 410, 'width': 700, 'date_captured': None, 'id': 0}, 
    # {'license': 0, 'url': None, 'file_name': 'JPEGImages/mod_1518_ok_3_frame2707.jpg', 'height': 410, 'width': 700, 'date_captured': None, 'id': 1},...]
    # 'type':instance
    # annotations: [{'id': 0, 'image_id': 0, 'category_id': 3, 'segmentation': [[367.35, 7.20, 380.82, 7.72,... ]], 'area': 7207.0, 'bbox': [227.0, 7.0, 154.0, 268.0], 'iscrowd': 0},
    # {'id': 1, 'image_id': 0, 'category_id': 3, 'segmentation': [[...]], 'area': 7266.0, 'bbox': [...], 'iscrowd': 0},...]
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
# with open('results/bbox_detections.json') as json_file:
#     pred_bbox = json.load(json_file)
    # list
    # [{'image_id': 0, 'category_id': 3, 'bbox': [228.0, 7.0, 152.0, 263.0], 'score': 0.99}, {...}...]
    # total 18624

# mask
# with open('results/mask_detections.json') as json_file:
#     pred_mask = json.load(json_file)
    # list
    # [{'image_id': 0, 'category_id': 3, 'segmentation': {'size': [410, 700], 
    # 'counts': '...0I8N101O1O1H8N2N3O0NhSP4'}, 'score': 0.99},...]
    # total 18624

# print(gt_bbox['categories'])
# print(len(pred_bbox))
# print(pred_mask[1])


# ap_data.pkl structure:
# 1. 'box', 'mask'
# 2. IoU (50,55,...95) total 10
# classes, total 6 in the order like below
# 3. 'U150_curve','U150_ok','U100_curve','U100_ok','A30_curve','A30_ok'

def ap_compare_plot(data:list,output_path,classes, pkl_1_name, pkl_2_name):
    if os.path.exists(output_path):
        os.remove(output_path)
    for i, clss in enumerate(classes):
        try:
            recalls_fast, precisions_fast, scores_fast = data[0]['box'][0][i].get_recall()
            recalls_norm, precisions_norm, scores_norm = data[1]['box'][0][i].get_recall()
            score_fast = []
            predi_fast = []
            recal_fast = []
            score_norm = []
            predi_norm = []
            recal_norm = []
            for i in range(0,10):
                threshold = i*0.1
                _score = [score for score in scores_fast if score >= threshold][-1]
                _recall = [recalls_fast[j] for j, score in enumerate(scores_fast) if score >= threshold][-1]
                _pred = [precisions_fast[j] for j, score in enumerate(scores_fast) if score >= threshold][-1]
                score_fast.append(_score)
                recal_fast.append(_recall)
                predi_fast.append(_pred)

                _score = [score for score in scores_norm if score >= threshold][-1]
                _recall = [recalls_norm[j] for j, score in enumerate(scores_norm) if score >= threshold][-1]
                _pred = [precisions_norm[j] for j, score in enumerate(scores_norm) if score >= threshold][-1]
                score_norm.append(_score)
                recal_norm.append(_recall)
                predi_norm.append(_pred)

            score_fast = [round(score,3) for score in score_fast]
            recal_fast = [round(recall,3) for recall in recal_fast]
            predi_fast = [round(pred,3) for pred in predi_fast]
            score_norm = [round(score,3) for score in score_norm]
            recal_norm = [round(recall,3) for recall in recal_norm]
            predi_norm = [round(pred,3) for pred in predi_norm]

            fig, ax = plt.subplots()
            title_name = clss
            plt.title(title_name)
            recal_curve_fast, = ax.plot(score_fast, recal_fast, marker='x',label=f'Recall ({pkl_1_name})',color='blue',alpha=0.5)
            recal_curve_norm, = ax.plot(score_norm, recal_norm, marker='x',label=f'Recall ({pkl_2_name})',color='red',alpha=0.5)
            predi_curve_fast, = ax.plot(score_fast, predi_fast, marker='o',label=f'Precision ({pkl_1_name})',color='blue',alpha=0.5)
            predi_curve_norm, = ax.plot(score_norm, predi_norm, marker='o',label=f'Precision ({pkl_2_name})',color='red',alpha=0.5)

            ax.set_xlabel('Confidence')
            ax.set_ylabel('Recll/Precision')
            plt.legend(handles=[recal_curve_fast,recal_curve_norm,predi_curve_fast,predi_curve_norm],loc='lower center')
            plt.savefig(f'/home/rico-li/Job/豐興鋼鐵/Prediction/{clss}_Comp_AP.png')
            # plt.show()
            # with open(output_path, 'a') as f:
            #     print(f'Kind: {clss}', file=f)
            #     print(f'Confidence: {score_sep}', file=f)
            #     print(f'Recall    : {recall_sep}', file=f)
            #     print(f'Precision : {pred_sep}\n', file=f)
        except:
            # with open(output_path, 'a') as f:
            print(f'Kind: {clss}:')
                # print(f'Kind: {clss}', file=f)
                # print('no ground truth!', file=f)
            continue
    print('save file save to /home/rico-li/Job/豐興鋼鐵/Prediction')

def ap_plot(data,output_path,classes):
    if os.path.exists(output_path):
        os.remove(output_path)
    for i, clss in enumerate(classes):
        try:
            recalls, precisions, scores = data['box'][0][i].get_recall()
            score_sep = []
            pred_sep = []
            recall_sep =[]
            for i in range(0,10):
                threshold = i*0.1
                _score = [score for score in scores if score >= threshold][-1]
                _recall = [recalls[j] for j, score in enumerate(scores) if score >= threshold][-1]
                _pred = [precisions[j] for j, score in enumerate(scores) if score >= threshold][-1]
                score_sep.append(_score)
                recall_sep.append(_recall)
                pred_sep.append(_pred)

            score_sep = [round(score,3) for score in score_sep]
            recall_sep = [round(recall,3) for recall in recall_sep]
            pred_sep = [round(pred,3) for pred in pred_sep]

            fig, ax = plt.subplots()
            title_name = clss
            plt.title(title_name)
            recall_curve,    = ax.plot(score_sep, recall_sep, marker='o',label='Recall')
            precision_curve, = ax.plot(score_sep, pred_sep, marker='o',label='Precision')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Recll/Precision')
            plt.legend(handles=[recall_curve,precision_curve],loc='right')
            plt.savefig(f'/home/rico-li/Job/豐興鋼鐵/Prediction/{clss}_AP_curve.png')
            # plt.show()
            with open(output_path, 'a') as f:
                print(f'Kind: {clss}', file=f)
                print(f'Confidence: {score_sep}', file=f)
                print(f'Recall    : {recall_sep}', file=f)
                print(f'Precision : {pred_sep}\n', file=f)
        except:
            with open(output_path, 'a') as f:
                print(f'Kind: {clss}:')
                print(f'Kind: {clss}', file=f)
                print('no ground truth!', file=f)
            continue
    print(f'save file save to /home/rico-li/Job/豐興鋼鐵/Prediction/{clss}_AP_curve.png')



if __name__ == "__main__":
    classes = ['U150_curve','U150_ok','U100_curve','U100_ok','A30_curve','A30_ok']
    output_path = '/home/rico-li/Job/豐興鋼鐵/Prediction/Confidence_Recall_Precison_U150.txt'
    # with open('/home/rico-li/Job/豐興鋼鐵/Prediction/U150/U150_ap_data.pkl','rb') as pkl:
    #     data = pickle.load(pkl)
    # ap_plot(data=data,output_path=output_path,classes=classes)

    pkl_1_path = '/home/rico-li/Job/豐興鋼鐵/Prediction/U150/yolact_plus_base_1731_114285/U150_ap_data.pkl'
    pkl_2_path = '/home/rico-li/Job/豐興鋼鐵/Prediction/U150/yolact_base_2777_133333/U150_ap_data.pkl'
    with open(pkl_1_path,'rb') as pkl:
        data_1 = pickle.load(pkl)
    with open(pkl_2_path,'rb') as pkl:
        data_2 = pickle.load(pkl)
    data = [data_1, data_2]
    ap_compare_plot(data=data,output_path=output_path,classes=classes,pkl_1_name='Yolact++',pkl_2_name='Yolact_base')