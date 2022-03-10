import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from mmdet.distillation.builder import DISTILL_LOSSES
from mmcv.runner import force_fp32

eps = 1e-10

@DISTILL_LOSSES.register_module()
class PGDClsLoss(nn.Module):
    def __init__(self,
                 name,
                 temp_s,
                 temp_c,
                 loss_weight,
                 alpha,
                 beta,
                 delta,
                 **kwargs,
                 ):
        super(PGDClsLoss, self).__init__()
        self.temp_s = temp_s
        self.temp_c = temp_c
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    @force_fp32(apply_to=('preds_S', 'preds_T'))
    def forward(self,
                preds_S,
                preds_T,
                mask_fg,
                mask_bg,
                **kwargs):
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'

        mask_bg = mask_bg / (mask_bg != 0).to(dtype=preds_S.dtype).sum(dim=(2, 3), keepdims=True).clamp(min=eps)
        mask_fg = mask_fg / (mask_fg != 0).to(dtype=preds_S.dtype).sum(dim=(2, 3), keepdims=True).clamp(min=eps)

        S_attention_t, C_attention_t = self.get_attention(preds_T)
        S_attention_s, C_attention_s = self.get_attention(preds_S)

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T,
                                             mask_fg, mask_bg,
                                             C_attention_s, C_attention_t,
                                             S_attention_s, S_attention_t)
        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t,
                                       S_attention_s, S_attention_t)

        loss = self.loss_weight * self.alpha * fg_loss + \
               self.loss_weight * self.beta * bg_loss + \
               self.delta * mask_loss

        return loss

    def get_attention(self, preds):
        """ preds: Bs*C*W*H """
        N, C, H, W = preds.shape
        value = torch.abs(preds)

        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map / self.temp_s).view(N, -1), dim=1)).view(N, H, W)

        # Bs*C
        C_attention = C * F.softmax(value / self.temp_c, dim=1)

        return S_attention, C_attention

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s - C_t))) / (C_s.size(0) * C_s.size(-2) * C_s.size(-1)) \
                    + torch.sum(torch.abs((S_s - S_t))) / S_t.size(0)

        return mask_loss

