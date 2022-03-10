import torch
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.fcos_head import FCOSHead

from .utils.builder import build_distill_weight
from .utils.background_weight import get_back_weight_anchorfree
from .utils.pgw_anchor_free import PGWAnchorFreeModule
from .utils.pgw_anchor_based import PGWAnchorModule

INF = 1e8

@HEADS.register_module()
class FCOSDistillHead(FCOSHead):
    def __init__(self, **kwargs):
        super(FCOSDistillHead, self).__init__(**kwargs)
        self.w_assigner_cls = build_distill_weight(self.train_cfg.kd_assigner.cls_assigner)
        self.w_assigner_reg = build_distill_weight(self.train_cfg.kd_assigner.reg_assigner)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_kd_value_maps(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          gt_bbox_ignore=None
                          ):
        with torch.no_grad():
            assert len(cls_scores) == len(bbox_preds) == len(centernesses)
            featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
            all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
            num_points_per_levels = [x.shape[0] for x in all_level_points]

            batch_size = cls_scores[0].shape[0]
            num_levels = len(cls_scores)
            all_level_points = [x.reshape(1, -1, 2).repeat(batch_size,1,1) for x in all_level_points]
            all_bbox_preds = [x.permute(0,2,3,1).reshape(batch_size, -1, 4) for x in bbox_preds]
            all_level_bbox_pred = [distance2bbox(x, y) for (x,y) in zip(all_level_points, all_bbox_preds)]

            cls_score_list = []; bbox_pred_list = []  # by image
            for i in range(batch_size):
                tmp_cls_list = []; tmp_bbox_list = []
                for j in range(num_levels):
                    cls_score = cls_scores[j][i].permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                    bbox_pred = all_level_bbox_pred[j][i]
                    tmp_cls_list.append(cls_score); tmp_bbox_list.append(bbox_pred)
                cat_cls_score = torch.cat(tmp_cls_list, dim=0); cat_bbox_pred = torch.cat(tmp_bbox_list, dim=0)
                cls_score_list.append(cat_cls_score); bbox_pred_list.append(cat_bbox_pred)

            cat_points = torch.cat(all_level_points, dim=1)  # [batch_size, all_N, 2]

            kd_value_cls_maps = [[] for _ in range(num_levels)]
            kd_value_reg_maps = [[] for _ in range(num_levels)]
            kd_backmaps = [[] for _ in range(num_levels)]
            for i in range(batch_size):
                kd_value_cls_map = self.w_assigner_cls.assign(cat_points[i], cls_score_list[i], bbox_pred_list[i],
                                                              gt_bboxes=gt_bboxes[i], gt_labels=gt_labels[i])
                kd_value_reg_map = self.w_assigner_reg.assign(cat_points[i], cls_score_list[i], bbox_pred_list[i],
                                                              gt_bboxes=gt_bboxes[i], gt_labels=gt_labels[i])
                back_map = get_back_weight_anchorfree(cat_points[i], gt_bboxes[i])
                for l in range(num_levels):
                    kd_value_cls_level = kd_value_cls_map[sum(num_points_per_levels[:l]):sum(num_points_per_levels[:(l+1)])]
                    kd_value_reg_level = kd_value_reg_map[sum(num_points_per_levels[:l]):sum(num_points_per_levels[:(l+1)])]
                    back_map_level = back_map[sum(num_points_per_levels[:l]):sum(num_points_per_levels[:(l+1)])]
                    kd_value_cls_maps[l].append(kd_value_cls_level.reshape(1, 1, featmap_sizes[l][0], featmap_sizes[l][1]))
                    kd_value_reg_maps[l].append(kd_value_reg_level.reshape(1, 1, featmap_sizes[l][0], featmap_sizes[l][1]))
                    kd_backmaps[l].append(back_map_level.reshape(1, 1, featmap_sizes[l][0], featmap_sizes[l][1]))
            kd_value_cls_maps = [torch.cat(x, dim=0) for x in kd_value_cls_maps]
            kd_value_reg_maps = [torch.cat(x, dim=0) for x in kd_value_reg_maps]
            kd_backmaps = [torch.cat(x, dim=0) for x in kd_backmaps]

            return dict(cls_value_maps=kd_value_cls_maps, reg_value_maps=kd_value_reg_maps, back_maps=kd_backmaps)


