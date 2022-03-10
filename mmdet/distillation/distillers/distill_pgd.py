import torch
from mmdet.distillation.builder import DISTILLER
from .distill_base import DistillBaseDetector
from ..losses import PGDClsLoss, PGDRegLoss

from mmdet.models.dense_heads.pgd_heads.atss_iou_distill import ATSSIoUHeadDistill
from mmdet.models.dense_heads.pgd_heads.autoassign_distill import AutoAssignDistillHead
from mmdet.models.dense_heads.pgd_heads.ddod_distill import DDOD_Distill_Head
from mmdet.models.dense_heads.pgd_heads.fcos_distill import FCOSDistillHead
from mmdet.models.dense_heads.pgd_heads.gfl_distill import GFLDistillHead

@DISTILLER.register_module()
class PredictionGuidedDistiller(DistillBaseDetector):
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):

        super(DistillBaseDetector, self).forward_train(img, img_metas)

        with torch.no_grad():
            self.teacher.eval()
            x_teacher = self.teacher.extract_feat(img)
            teacher_preds = self.teacher.bbox_head(x_teacher)

        x_student = self.student.extract_feat(img)
        student_preds = self.student.bbox_head(x_student)
        if type(self.student.bbox_head) is DDOD_Distill_Head:
            student_loss, label_assign = self.student.bbox_head.loss(*student_preds, *teacher_preds,
                                                                     gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore,
                                                                     ret_label_assign=True)
        elif type(self.student.bbox_head) is ATSSIoUHeadDistill:
            student_loss, label_assign = self.student.bbox_head.loss(*student_preds, *teacher_preds,
                                                                     gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore,
                                                                     ret_label_assign=True)
        elif type(self.student.bbox_head) is GFLDistillHead:
            student_loss, label_assign = self.student.bbox_head.loss(*student_preds, *teacher_preds,
                                                                     gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore,
                                                                     ret_label_assign=True)
        elif type(self.student.bbox_head) is FCOSDistillHead:
            student_loss = self.student.bbox_head.loss(*student_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
            label_assign = self.student.bbox_head.get_kd_value_maps(*teacher_preds,gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
        elif type(self.student.bbox_head) is AutoAssignDistillHead:
            student_loss = self.student.bbox_head.loss(*student_preds, gt_bboxes, gt_labels, img_metas,gt_bboxes_ignore)
            label_assign = self.student.bbox_head.get_kd_value_maps(*teacher_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)
        else:
            raise NotImplementedError

        cls_value_dict = dict()
        reg_value_dict = dict()
        back_value_dict = dict()
        for i, (cv, rv, bv, x) in enumerate(
                zip(label_assign['cls_value_maps'],
                    label_assign['reg_value_maps'],
                    label_assign['back_maps'],
                    x_student)):
            N, C, H, W = x.size()
            cls_value_dict[(H, W)] = cv.reshape(N, 1, H, W).to(dtype=x_student[0].dtype)
            reg_value_dict[(H, W)] = rv.reshape(N, 1, H, W).to(dtype=x_student[0].dtype)
            back_value_dict[(H,W)] = bv.reshape(N, 1, H, W).to(dtype=x_student[0].dtype)

        buffer_dict = dict(self.named_buffers())
        for item_loc in self.distill_cfg:

            student_module = 'student_' + item_loc.student_module.replace('.', '_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.', '_')

            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]

            for item_loss in item_loc.methods:
                loss_name = item_loss.name

                if type(self.distill_losses[loss_name]) == PGDClsLoss:
                    mask_fg = cls_value_dict[tuple(student_feat.shape[-2:])]
                elif type(self.distill_losses[loss_name]) == PGDRegLoss:
                    mask_fg = reg_value_dict[tuple(student_feat.shape[-2:])]
                else:
                    raise NotImplementedError
                mask_bg = back_value_dict[tuple(student_feat.shape[-2:])]
                student_loss[loss_name] = self.distill_losses[loss_name](student_feat, teacher_feat,
                                                                         mask_fg=mask_fg, mask_bg=mask_bg)
        return student_loss