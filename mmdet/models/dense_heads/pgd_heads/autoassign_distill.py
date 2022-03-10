import torch
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, reduce_mean
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.paa_head import levels_to_images
from mmdet.models.dense_heads.autoassign_head import AutoAssignHead

from .utils.builder import build_distill_weight
from .utils.background_weight import get_back_weight_anchorfree
from .utils.pgw_anchor_free import PGWAnchorFreeModule
from .utils.pgw_anchor_based import PGWAnchorModule

INF = 1e8
EPS = 1e-12

@HEADS.register_module()
class AutoAssignDistillHead(AutoAssignHead):
    def __init__(self, **kwargs):
        super(AutoAssignDistillHead, self).__init__(**kwargs)
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
                    kd_value_cls_level = kd_value_cls_map[
                                         sum(num_points_per_levels[:l]):sum(num_points_per_levels[:(l + 1)])]
                    kd_value_reg_level = kd_value_reg_map[
                                         sum(num_points_per_levels[:l]):sum(num_points_per_levels[:(l + 1)])]
                    back_map_level = back_map[sum(num_points_per_levels[:l]):sum(num_points_per_levels[:(l + 1)])]
                    kd_value_cls_maps[l].append(
                        kd_value_cls_level.reshape(1, 1, featmap_sizes[l][0], featmap_sizes[l][1]))
                    kd_value_reg_maps[l].append(
                        kd_value_reg_level.reshape(1, 1, featmap_sizes[l][0], featmap_sizes[l][1]))
                    kd_backmaps[l].append(back_map_level.reshape(1, 1, featmap_sizes[l][0], featmap_sizes[l][1]))
            kd_value_cls_maps = [torch.cat(x, dim=0) for x in kd_value_cls_maps]
            kd_value_reg_maps = [torch.cat(x, dim=0) for x in kd_value_reg_maps]
            kd_backmaps = [torch.cat(x, dim=0) for x in kd_backmaps]

            return dict(cls_value_maps=kd_value_cls_maps, reg_value_maps=kd_value_reg_maps, back_maps=kd_backmaps)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss_offline(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_infos_from_teacher,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        self.center_prior.eval()

        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        all_num_gt = sum([len(item) for item in gt_bboxes])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        inside_gt_bbox_mask_list = gt_infos_from_teacher['inside_gt_bbox_mask_list']
        bbox_targets_list = gt_infos_from_teacher['bbox_targets_list']
        center_prior_weight_list = gt_infos_from_teacher['center_prior_weight_list']
        decoded_target_preds_list = gt_infos_from_teacher['decoded_target_preds_list']
        ious_list = gt_infos_from_teacher['ious_list']


        mlvl_points = torch.cat(all_level_points, dim=0)
        bbox_preds = levels_to_images(bbox_preds)
        cls_scores = levels_to_images(cls_scores)
        objectnesses = levels_to_images(objectnesses)

        reg_loss_list = []
        num_points = len(mlvl_points)

        for bbox_pred, gt_bboxe, inside_gt_bbox_mask, decoded_target_preds in zip(
                bbox_preds, bbox_targets_list, inside_gt_bbox_mask_list, decoded_target_preds_list):
            temp_num_gt = gt_bboxe.size(1)
            expand_mlvl_points = mlvl_points[:, None, :].expand(
                num_points, temp_num_gt, 2).reshape(-1, 2)
            gt_bboxe = gt_bboxe.reshape(-1, 4)
            expand_bbox_pred = bbox_pred[:, None, :].expand(
                num_points, temp_num_gt, 4).reshape(-1, 4)
            decoded_bbox_preds = distance2bbox(expand_mlvl_points, expand_bbox_pred)

            loss_bbox = self.loss_bbox(
                decoded_bbox_preds,
                decoded_target_preds,
                weight=None,
                reduction_override='none')
            reg_loss_list.append(loss_bbox.reshape(num_points, temp_num_gt))

        cls_scores = [item.sigmoid() for item in cls_scores]
        objectnesses = [item.sigmoid() for item in objectnesses]
        pos_loss_list, = multi_apply(self.get_pos_loss_single, cls_scores,
                                     objectnesses, reg_loss_list, gt_labels,
                                     center_prior_weight_list)
        pos_avg_factor = reduce_mean(
            bbox_pred.new_tensor(all_num_gt)).clamp_(min=1)
        pos_loss = sum(pos_loss_list) / pos_avg_factor

        neg_loss_list, = multi_apply(self.get_neg_loss_single, cls_scores,
                                     objectnesses, gt_labels, ious_list,
                                     inside_gt_bbox_mask_list)
        neg_avg_factor = sum(item.data.sum()
                             for item in center_prior_weight_list)
        neg_avg_factor = reduce_mean(neg_avg_factor).clamp_(min=1)
        neg_loss = sum(neg_loss_list) / neg_avg_factor

        center_loss = []
        for i in range(len(img_metas)):

            if inside_gt_bbox_mask_list[i].any():
                center_loss.append(
                    len(gt_bboxes[i]) /
                    center_prior_weight_list[i].sum().clamp_(min=EPS))
            # when width or height of gt_bbox is smaller than stride of p3
            else:
                center_loss.append(center_prior_weight_list[i].sum() * 0)

        center_loss = torch.stack(center_loss).mean() * self.center_loss_weight

        # avoid dead lock in DDP
        if all_num_gt == 0:
            pos_loss = bbox_preds[0].sum() * 0
            dummy_center_prior_loss = self.center_prior.mean.sum(
            ) * 0 + self.center_prior.sigma.sum() * 0
            center_loss = objectnesses[0].sum() * 0 + dummy_center_prior_loss

        loss = dict(
            loss_pos=pos_loss, loss_neg=neg_loss, loss_center=center_loss)

        return loss
