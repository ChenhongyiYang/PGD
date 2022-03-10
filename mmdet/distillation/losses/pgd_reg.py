import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from mmdet.distillation.builder import DISTILL_LOSSES
from mmcv.runner import force_fp32


@DISTILL_LOSSES.register_module()
class PGDRegLoss(nn.Module):
    def __init__(self,
                 name,
                 temp,
                 gamma,
                 delta,
                 **kwargs
                 ):
        super(PGDRegLoss, self).__init__()
        self.temp = temp
        self.gamma = gamma
        self.delta = delta

    @force_fp32(apply_to=('preds_S', 'preds_T'))
    def forward(self,
                preds_S,
                preds_T,
                mask_fg,
                **kwargs):
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'

        N, C, H, W = preds_S.shape

        preds_S = preds_S.permute(0, 2, 3, 1).reshape(-1, C)
        preds_T = preds_T.permute(0, 2, 3, 1).reshape(-1, C)
        target_map = mask_fg.reshape(-1)
        pos_inds = target_map != 0

        preds_S = preds_S[pos_inds]
        preds_T = preds_T[pos_inds]
        target_w = target_map[pos_inds]

        if preds_S.numel() > 0:
            C_attention_s = C * F.softmax(torch.abs(preds_S) / self.temp, dim=1)
            C_attention_t = C * F.softmax(torch.abs(preds_T) / self.temp, dim=1)

            fea_loss = F.mse_loss(preds_S, preds_T, reduction='none') * C_attention_t
            fea_loss = (fea_loss.mean(dim=1) * target_w).mean()

            mask_loss = torch.sum(torch.abs((C_attention_s - C_attention_t))) / C_attention_s.size(0)
            return self.gamma * fea_loss + self.delta * mask_loss

        else:
            return preds_S.sum() * 0.

