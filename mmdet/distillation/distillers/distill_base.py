import torch.nn as nn
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint, load_state_dict
from mmdet.distillation.builder import DISTILLER, build_distill_loss
from collections import OrderedDict


@DISTILLER.register_module()
class DistillBaseDetector(BaseDetector):
    """Base distiller for detectors.
    It typically consists of teacher_model and student_model.
    """

    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=False):

        super(DistillBaseDetector, self).__init__()

        self.teacher = build_detector(teacher_cfg.model,
                                      train_cfg=teacher_cfg.get('train_cfg'),
                                      test_cfg=teacher_cfg.get('test_cfg'))
        self.teacher_pretrained = teacher_pretrained
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.student = build_detector(student_cfg.model,
                                      train_cfg=student_cfg.get('train_cfg'),
                                      test_cfg=student_cfg.get('test_cfg'))

        self.init_student = init_student


        self.distill_losses = nn.ModuleDict()

        self.distill_cfg = distill_cfg
        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())

        def regitster_hooks(student_module, teacher_module):
            def hook_teacher_forward(module, input, output):
                self.register_buffer(teacher_module, output)

            def hook_student_forward(module, input, output):
                self.register_buffer(student_module, output)

            return hook_teacher_forward, hook_student_forward

        if type(distill_cfg) is list:
            for item_loc in distill_cfg:
                if type(item_loc) is not None:
                    student_module = 'student_' + item_loc.student_module.replace('.', '_')
                    teacher_module = 'teacher_' + item_loc.teacher_module.replace('.', '_')

                    self.register_buffer(student_module, None)
                    self.register_buffer(teacher_module, None)

                    hook_teacher_forward, hook_student_forward = regitster_hooks(student_module, teacher_module)
                    teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
                    student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

                    for item_loss in item_loc.methods:
                        loss_name = item_loss.name
                        self.distill_losses[loss_name] = build_distill_loss(item_loss)


    def base_parameters(self):
        return nn.ModuleList([self.student, self.distill_losses])

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self):
        """Load the pretrained model in teacher detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        device_id = torch.cuda.current_device()
        checkpoint = load_checkpoint(self.teacher, self.teacher_pretrained, map_location=torch.device('cuda', device_id))
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        if self.init_student:
            all_name = []
            for name, v in checkpoint["state_dict"].items():
                if not name.startswith("backbone."):
                    all_name.append((name, v))
            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        pass

    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)
