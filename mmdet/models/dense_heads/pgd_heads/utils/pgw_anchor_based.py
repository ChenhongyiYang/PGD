import torch
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from .builder import DISTILL_WEIGHT

eps = 1e-10

@DISTILL_WEIGHT.register_module()
class PGWAnchorModule(torch.nn.Module):
    def __init__(self,
                 alpha,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1,
                 low_bound=0.,):

        super(PGWAnchorModule, self).__init__()
        self.alpha = alpha
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        self.low_bound = low_bound

    @torch.no_grad()
    def assign(self,
               bboxes,
               cls_scores,
               bbox_preds,
               gt_bboxes,
               bbox_levels,
               gt_bboxes_ignore=None,
               gt_labels=None):

        bboxes = bboxes[:, :4]  # anchor bbox
        bbox_preds = bbox_preds.detach()
        cls_scores = cls_scores.detach()

        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        if num_gt == 0 or num_bboxes == 0:
            return torch.zeros((num_bboxes,), dtype=cls_scores.dtype)

        overlaps = self.iou_calculator(bbox_preds, gt_bboxes)
        cls_cost = torch.sigmoid(cls_scores[:, gt_labels])  # [num_bbox, num_gt]
        assert cls_cost.shape == overlaps.shape

        overlaps = cls_cost ** (1 - self.alpha) * overlaps ** self.alpha

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,), 0, dtype=torch.long)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            assigned_gt_inds[ignore_idxs] = -1

        _, topk_idxs = overlaps.topk(self.topk, dim=0, largest=True)  # [topk, num_gt]
        candidate_idxs = topk_idxs

        candidate_cx = bboxes_cx[candidate_idxs.view(-1)].reshape(candidate_idxs.shape)  # [topk, num_gt]
        candidate_cy = bboxes_cy[candidate_idxs.view(-1)].reshape(candidate_idxs.shape)  # [topk, num_gt]
        candidate_pos = torch.stack((candidate_cx.transpose(0, 1), candidate_cy.transpose(0, 1)),
                                    dim=-1)  # [num_gt, topk, 2]

        if self.topk != 1:
            miu, sigma, inverse, deter = mle_2d_gaussian_2(candidate_pos)

        x1 = bboxes[:, 0][:, None].repeat(1, num_gt)  # [n_bbox, n_gt]
        y1 = bboxes[:, 1][:, None].repeat(1, num_gt)
        x2 = bboxes[:, 2][:, None].repeat(1, num_gt)
        y2 = bboxes[:, 3][:, None].repeat(1, num_gt)

        cx = (x1 + x2) * 0.5  # [n_bbox, n_gt]
        cy = (y1 + y2) * 0.5  # [n_bbox, n_gt]

        gx1 = gt_bboxes[:, 0][None, :].repeat(num_bboxes, 1)  # [n_bbox, n_gt]
        gy1 = gt_bboxes[:, 1][None, :].repeat(num_bboxes, 1)
        gx2 = gt_bboxes[:, 2][None, :].repeat(num_bboxes, 1)
        gy2 = gt_bboxes[:, 3][None, :].repeat(num_bboxes, 1)

        valid = ((cx - gx1) > eps) * ((cy - gy1) > eps) * ((gx2 - cx) > eps) * ((gy2 - cy) > eps)
        valid = valid.to(dtype=cls_scores.dtype)

        if self.topk != 1:
            pos_diff = (candidate_pos - miu)[:, :, None, :]  # [num_gt, topk, 1, 2]
            candidate_w = torch.exp(-0.5 * torch.matmul(torch.matmul(pos_diff, inverse[:, None, :, :]),
                                                            pos_diff.transpose(2, 3)))  # [num_gt, topk]
            candidate_w = candidate_w.transpose(0, 1).reshape(self.topk, num_gt)
            w = torch.zeros_like(bboxes_cx).reshape(-1, 1).repeat(1, num_gt)
            for i in range(num_gt):
                w[candidate_idxs[:, i], i] = candidate_w[:, i]
        else:
            w = torch.zeros_like(bboxes_cx).reshape(-1, 1).repeat(1, num_gt)
            for i in range(num_gt):
                w[candidate_idxs[:, i], i] = 1.

        w = w * valid
        w = torch.max(w, dim=1)[0]
        w[assigned_gt_inds == -1] = 0.
        w[w < self.low_bound] = 0.

        return w

    def forward(self, **kwargs):
        pass


def mle_2d_gaussian_2(sampled_data):
    # sampled_data [N, M, 2]
    data = sampled_data + (torch.rand(sampled_data.size(), device=sampled_data.device) - 0.5) * 0.1  # hack: add noise to avoid zero determint
    miu = data.mean(dim=1, keepdim=True) #[N, 1, 2]
    diff = (data - miu)[:, :, :, None]
    sigma = torch.matmul(diff, diff.transpose(2, 3)).mean(dim=1)  # [N, 2, 2]
    deter = sigma[:, 0, 0] * sigma[:, 1, 1] - sigma[:, 0, 1] * sigma[:, 1, 0]  # [N]

    inverse = torch.zeros_like(sigma)
    inverse[:, 0, 0] = sigma[:, 1, 1]
    inverse[:, 0, 1] = -1. * sigma[:, 0, 1]
    inverse[:, 1, 0] = -1. * sigma[:, 1, 0]
    inverse[:, 1, 1] = sigma[:, 0, 0]
    inverse /= (deter[:,None,None]+1e-10)
    return miu, sigma, inverse, deter










