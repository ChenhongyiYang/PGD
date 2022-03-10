import torch
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from .pgw_anchor_based import mle_2d_gaussian_2
from .builder import DISTILL_WEIGHT

eps = 1e-10

@DISTILL_WEIGHT.register_module()
class PGWAnchorFreeModule(torch.nn.Module):
    def __init__(self,
                 alpha,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1,
                 low_bound=0.,):

        super(PGWAnchorFreeModule, self).__init__()
        self.alpha = alpha
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        self.low_bound = low_bound

    @torch.no_grad()
    def assign(self,
               points,
               cls_scores,
               bbox_preds,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):

        bbox_preds = bbox_preds.detach()
        cls_scores = cls_scores.detach()

        num_gt, num_points = gt_bboxes.size(0), points.size(0)
        if num_gt == 0 or num_points == 0:
            return torch.zeros((num_points,), dtype=cls_scores.dtype)

        overlaps = self.iou_calculator(bbox_preds, gt_bboxes)
        cls_cost = torch.sigmoid(cls_scores[:, gt_labels])
        assert cls_cost.shape == overlaps.shape

        overlaps = cls_cost ** (1 - self.alpha) * overlaps ** self.alpha

        assigned_gt_inds = overlaps.new_full((num_points, ), 0, dtype=torch.long)

        # TODO: deal with the ignore gt_bboxes for anchor-free models
        # if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
        #         and gt_bboxes_ignore.numel() > 0 and points.numel() > 0):
        #     ignore_overlaps = self.iou_calculator(
        #         bboxes, gt_bboxes_ignore, mode='iof')
        #     ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
        #     ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
        #     assigned_gt_inds[ignore_idxs] = -1

        cy = points[:, 0].reshape(-1, 1).repeat(1, num_gt)
        cx = points[:, 1].reshape(-1, 1).repeat(1, num_gt)

        gx1 = gt_bboxes[:, 0][None, :].repeat(num_points, 1)  # [n_bbox, n_gt]
        gy1 = gt_bboxes[:, 1][None, :].repeat(num_points, 1)
        gx2 = gt_bboxes[:, 2][None, :].repeat(num_points, 1)
        gy2 = gt_bboxes[:, 3][None, :].repeat(num_points, 1)

        valid = ((cx - gx1) > eps) * ((cy - gy1) > eps) * ((gx2 - cx) > eps) * ((gy2 - cy) > eps)
        valid = valid.to(dtype=cls_scores.dtype)

        _, topk_idxs = overlaps.topk(self.topk, dim=0, largest=True) # [topk, num_gt]
        candidate_idxs = topk_idxs

        candidate_points = points[candidate_idxs.view(-1)].reshape((self.topk, num_gt, 2)).permute(1, 0, 2)  #[num_gt, topk, 2]

        miu, sigma, inverse, deter = mle_2d_gaussian_2(candidate_points)

        pos_diff = (candidate_points - miu)[:, :, None, :]  # [num_gt, topk, 1, 2]
        candidate_w = torch.exp(-0.5 * torch.matmul(torch.matmul(pos_diff, inverse[:, None, :, :]), pos_diff.transpose(2, 3)))  # [num_gt, topk]
        candidate_w = candidate_w.transpose(0, 1).reshape(self.topk, num_gt)

        w = torch.zeros_like(points[:, 0]).reshape(-1, 1).repeat(1, num_gt)

        for i in range(num_gt):
            w[candidate_idxs[:,i], i] = candidate_w[:, i]

        w = w * valid
        w = torch.max(w, dim=1)[0]
        w[assigned_gt_inds == -1] = 0.
        w[w < self.low_bound] = 0.

        return w

    def forward(self, **kwargs):
        pass












