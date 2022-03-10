import torch

eps = 1e-10

def get_back_weight(bboxes, gt_bboxes):
    with torch.no_grad():
        bboxes = bboxes[:, :4]  # anchor bbox

        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        x1 = bboxes[:, 0][:, None].repeat(1, num_gt)  # [n_bbox, n_gt]
        y1 = bboxes[:, 1][:, None].repeat(1, num_gt)
        x2 = bboxes[:, 2][:, None].repeat(1, num_gt)
        y2 = bboxes[:, 3][:, None].repeat(1, num_gt)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        gx1 = gt_bboxes[:, 0][None, :].repeat(num_bboxes, 1)  # [n_bbox, n_gt]
        gy1 = gt_bboxes[:, 1][None, :].repeat(num_bboxes, 1)
        gx2 = gt_bboxes[:, 2][None, :].repeat(num_bboxes, 1)
        gy2 = gt_bboxes[:, 3][None, :].repeat(num_bboxes, 1)

        valid = ((cx - gx1) > eps) * ((cy - gy1) > eps) * ((gx2 - cx) > eps) * ((gy2 - cy) > eps)
        valid = valid.to(dtype=bboxes.dtype)

        in_box = valid.sum(dim=1)
        back_w = (in_box == 0).to(dtype=bboxes.dtype)

        return back_w

def get_back_weight_anchorfree(points, gt_bboxes):
    with torch.no_grad():
        num_gt, num_bboxes = gt_bboxes.size(0), points.size(0)

        cy = points[:, 0].reshape(-1, 1).repeat(1, num_gt)
        cx = points[:, 1].reshape(-1, 1).repeat(1, num_gt)

        gx1 = gt_bboxes[:, 0][None, :].repeat(num_bboxes, 1)  # [n_bbox, n_gt]
        gy1 = gt_bboxes[:, 1][None, :].repeat(num_bboxes, 1)
        gx2 = gt_bboxes[:, 2][None, :].repeat(num_bboxes, 1)
        gy2 = gt_bboxes[:, 3][None, :].repeat(num_bboxes, 1)

        valid = ((cx - gx1) > eps) * ((cy - gy1) > eps) * ((gx2 - cx) > eps) * ((gy2 - cy) > eps)
        valid = valid.to(dtype=gt_bboxes.dtype)

        in_box = valid.sum(dim=1)
        back_w = (in_box == 0).to(dtype=gt_bboxes.dtype)

        return back_w











