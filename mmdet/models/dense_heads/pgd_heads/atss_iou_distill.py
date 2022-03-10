import torch
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags,
                        images_to_levels, multi_apply,
                        reduce_mean, unmap)
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.atss_iou_head import ATSSIoUHead

from .utils.builder import build_distill_weight
from .utils.background_weight import get_back_weight
from .utils.pgw_anchor_free import PGWAnchorFreeModule
from .utils.pgw_anchor_based import PGWAnchorModule


EPS = 1e-12

@HEADS.register_module()
class ATSSIoUHeadDistill(ATSSIoUHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self, **kwargs):

        super(ATSSIoUHeadDistill, self).__init__(**kwargs)
        self.w_assigner_cls = build_distill_weight(self.train_cfg.kd_assigner.cls_assigner)
        self.w_assigner_reg = build_distill_weight(self.train_cfg.kd_assigner.reg_assigner)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             cls_scores_offline,
             bbox_preds_offline,
             iou_preds_offline,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             ret_label_assign=False):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        atss_cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            cls_scores_offline,
            bbox_preds_offline,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if atss_cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg, kd_value_maps_cls, kd_value_maps_reg, kd_back_map_list) = atss_cls_reg_targets

        atss_num_total_pos = num_total_pos

        num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            iou_preds,
            # centernesses,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        if not ret_label_assign:
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_iou=losses_iou)
        else:
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_iou=losses_iou), \
                   dict(cls_value_maps=kd_value_maps_cls, reg_value_maps=kd_value_maps_reg, back_maps=kd_back_map_list)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs


        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        num_levels = len(cls_scores)
        cls_score_list = []
        bbox_pred_list = []
        for i in range(num_imgs):
            tmp_cls_list = [];
            tmp_bbox_list = []
            for j in range(num_levels):
                cls_score = cls_scores[j][i].permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                bbox_pred = bbox_preds[j][i].permute(1, 2, 0).reshape(-1, 4)
                tmp_cls_list.append(cls_score); tmp_bbox_list.append(bbox_pred)
            cat_cls_score = torch.cat(tmp_cls_list, dim=0); cat_bbox_pred = torch.cat(tmp_bbox_list, dim=0)
            cls_score_list.append(cat_cls_score); bbox_pred_list.append(cat_bbox_pred)

        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list, all_kd_value_cls_map, all_kd_value_reg_map, all_kd_back_map) = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            cls_score_list,
            bbox_pred_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        kd_value_map_cls_list = images_to_levels(all_kd_value_cls_map,
                                                 num_level_anchors)
        kd_value_map_reg_list = images_to_levels(all_kd_value_reg_map,
                                                 num_level_anchors)
        kd_back_map_list = images_to_levels(all_kd_back_map,
                                            num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg, kd_value_map_cls_list, kd_value_map_reg_list, kd_back_map_list)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           cls_scores,
                           bbox_preds,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        anchor_levels = torch.zeros((flat_anchors.size(0),))
        for i in range(len(num_level_anchors)):
            anchor_levels[sum(num_level_anchors[:i]):sum(num_level_anchors[:i + 1])] = i
        anchor_levels = anchor_levels[inside_flags]

        bbox_preds_valid = bbox_preds[inside_flags, :]
        cls_scores_valid = cls_scores[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)
        bbox_preds_valid = self.bbox_coder.decode(anchors, bbox_preds_valid)
        kd_value_cls_map = self.w_assigner_cls.assign(anchors, cls_scores_valid, bbox_preds_valid,
                                                      gt_bboxes, anchor_levels, gt_bboxes_ignore, gt_labels)
        kd_value_reg_map = self.w_assigner_reg.assign(anchors, cls_scores_valid, bbox_preds_valid,
                                                      gt_bboxes, anchor_levels, gt_bboxes_ignore, gt_labels)
        kd_back_map = get_back_weight(anchors, gt_bboxes)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                # used in VFNetHead
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            kd_value_cls_map = unmap(kd_value_cls_map, num_total_anchors, inside_flags)
            kd_value_reg_map = unmap(kd_value_reg_map, num_total_anchors, inside_flags)
            kd_back_map = unmap(kd_back_map, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, kd_value_cls_map, kd_value_reg_map, kd_back_map)
