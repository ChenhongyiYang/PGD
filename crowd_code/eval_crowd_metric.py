import mmcv
import numpy as np
from pycocotools.coco import COCO
from evaluate import compute_JI, compute_APMR
import json
from utils import misc_utils 

anno_path = './data/crowdhuman/annotations/annotation_val.json'
pred_path = './work_dirs/results.pkl'

anno_data = COCO(anno_path)
pred_data = mmcv.load(pred_path)
img_ids = anno_data.getImgIds()

def boxes_dump(boxes):
    if boxes.shape[-1] == 6:
        result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                   'score':round(float(box[4]), 5),
                   'tag': 1} for box in boxes]
    elif boxes.shape[-1] == 5:
        result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                   'tag': 1 if box[4] == 1 else -1} for box in boxes]
    else:
        raise ValueError('Unknown box dim.')
    return result

results_list = []
for ii, img_id in enumerate(img_ids):
    preds = pred_data[ii]
    img_info = anno_data.loadImgs(img_id)[0]
    # parse GT annotations
    anno_ids = anno_data.getAnnIds(img_id)
    annos = anno_data.loadAnns(anno_ids)
    gt_bboxes = []
    for anno in annos:
        gt_bbox = [x for x in anno['bbox']]
        tag = -1 if anno['iscrowd'] else 1
        gt_bbox.append(tag)
        gt_bboxes.append(gt_bbox)
    gt_bboxes = np.array(gt_bboxes).reshape(-1, 5)
    # parse prediction
    preds = preds[0]  # get person prediction (all predictions are person)
    tags = np.ones((preds.shape[0], 1))
    pred_bboxes = np.concatenate([preds, tags], axis=1)
    pred_bboxes[:, 2:4] -= pred_bboxes[:, 0:2]
    result_dict = dict(ID=img_info['file_name'][:-4], height=int(img_info['height']), width=int(img_info['width']),
                dtboxes=boxes_dump(pred_bboxes), gtboxes=boxes_dump(gt_bboxes))
    results_list.append(result_dict)
    
fpath = './crowd_code/eval.json'
misc_utils.save_json_lines(results_list, fpath)
eval_path = './crowd_code/eval_res.json'
eval_fid = open(eval_path, 'w')
res_line, JI = compute_JI.evaluation_all(fpath, 'box')
for line in res_line:
    eval_fid.write(line+'\n')
eval_source = 'data/crowdhuman/annotations/annotation_val.odgt'
AP, MR = compute_APMR.compute_APMR(fpath, eval_source, 'box')
line = 'AP:{:.4f}, MR:{:.4f}, JI:{:.4f}.'.format(AP, MR, JI)
print(line)

    
    
    
