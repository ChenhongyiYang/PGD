import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

INF = 100000000
eps = 1e-10

def masked_mean(matrix, dim, masked_value=-INF):
    indicator = matrix != masked_value

    matrix_ = matrix.clone()
    matrix_[matrix == masked_value] = 0
    _sum = matrix_.sum(dim=dim)
    _denorm = indicator.sum(dim=dim)
    return _sum / (_denorm + eps)

def masked_std(matrix, dim, mean, masked_value=-INF):
    indicator = matrix != masked_value

    matrix_ = matrix.clone()

    if dim == 0:
        matrix_ = matrix_ - mean.reshape(1, -1)
    elif dim == 1:
        matrix_ = matrix_ - mean.reshape(-1, 1)
    else:
        raise NotImplementedError
    matrix_[matrix == masked_value] = 0

    std = torch.sqrt((matrix_**2).sum(dim=dim) / (eps+indicator.sum(dim=dim)-1).clamp(min=eps))

    return std



@BBOX_ASSIGNERS.register_module()
class DDODCenterTopkAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 alpha,
                 is_atss,
                 center_region=(0.2, 0.2),
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.alpha = alpha
        self.is_atss = is_atss
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        self.center_region = center_region

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               cls_scores,
               bbox_preds,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        # NOTE first convert anchor to prediction bbox
        bboxes = bboxes[:, :4]      # anchor bbox
        bbox_preds = bbox_preds.detach()
        cls_scores = cls_scores.detach()

        # bbox_preds = bbox_coder.decode(bboxes, bbox_preds)      # prediction bbox

        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # NOTE DeFCN style cost function
        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bbox_preds, gt_bboxes)
        # compute cls cost for bbox and GT
        cls_cost = torch.sigmoid(cls_scores[:, gt_labels])

        # make sure that we are in element-wise multiplication
        assert cls_cost.shape == overlaps.shape

        # overlaps is actually is a cost matrix
        overlaps = cls_cost ** (1 - self.alpha) * overlaps ** self.alpha  # num_boxes, num_gt

        gt_center_x1 = gt_bboxes[:, 0] + (1. - self.center_region[0]) * 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0])
        gt_center_y1 = gt_bboxes[:, 1] + (1. - self.center_region[1]) * 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        gt_center_x2 = gt_bboxes[:, 2] - (1. - self.center_region[0]) * 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0])
        gt_center_y2 = gt_bboxes[:, 3] - (1. - self.center_region[1]) * 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1])

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

        bboxes_cx_extend = bboxes_cx.reshape(-1, 1).repeat(1, num_gt)
        bboxes_cy_extend = bboxes_cy.reshape(-1, 1).repeat(1, num_gt)

        l_ = bboxes_cx_extend - gt_center_x1
        t_ = bboxes_cy_extend - gt_center_y1
        r_ = gt_center_x2 - bboxes_cx_extend
        b_ = gt_center_y2 - bboxes_cy_extend

        is_in_center = torch.stack([l_, t_, r_, b_], dim=-1).min(dim=-1)[0] > 0.01
        overlaps[~is_in_center] = -INF

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)


        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            overlaps[ignore_idxs, :] = -INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            overlaps_per_level = overlaps[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = overlaps_per_level.topk(selectable_k, dim=0, largest=True)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)  # [topk * num_level, num_gt]

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        if self.is_atss:
            overlaps_mean_per_gt = masked_mean(candidate_overlaps, dim=0)
            overlaps_std_per_gt = masked_std(candidate_overlaps, dim=0, mean=overlaps_mean_per_gt)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
            is_pos = candidate_overlaps > overlaps_thr_per_gt[None, :]
            # print(overlaps_thr_per_gt)
        else:
            is_pos = candidate_overlaps > -0.1

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)