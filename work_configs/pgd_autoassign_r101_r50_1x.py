_base_ = "base/1x_setting.py"

temperature = 0.4
alpha       = 0.08
delta       = 0.0008
beta        = alpha * 0.5
gamma       = alpha * 1.6

distiller = dict(
    type='PredictionGuidedDistiller',
    teacher_pretrained = 'work_dirs/autoassign_r101_3x_ms/epoch_36.pth',
    init_student = True,
    distill_cfg=[dict(student_module='bbox_head.cls_identities.0',
                      teacher_module='bbox_head.cls_identities.0',
                      output_hook=True,
                      methods=[dict(type='PGDClsLoss',
                                    name='loss_kd_cls_0',
                                    temp_s=temperature,
                                    temp_c=temperature,
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    delta=delta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.cls_identities.1',
                      teacher_module='bbox_head.cls_identities.1',
                      output_hook=True,
                      methods=[dict(type='PGDClsLoss',
                                    name='loss_kd_cls_1',
                                    temp_s=temperature,
                                    temp_c=temperature,
                                    loss_weight=1.25,
                                    alpha=alpha,
                                    beta=beta,
                                    delta=delta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.cls_identities.2',
                      teacher_module='bbox_head.cls_identities.2',
                      output_hook=True,
                      methods=[dict(type='PGDClsLoss',
                                    name='loss_kd_cls_2',
                                    temp_s=temperature,
                                    temp_c=temperature,
                                    loss_weight=1.5,
                                    alpha=alpha,
                                    beta=beta,
                                    delta=delta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.cls_identities.3',
                      teacher_module='bbox_head.cls_identities.3',
                      output_hook=True,
                      methods=[dict(type='PGDClsLoss',
                                    name='loss_kd_cls_3',
                                    temp_s=temperature,
                                    temp_c=temperature,
                                    loss_weight=1.75,
                                    alpha=alpha,
                                    beta=beta,
                                    delta=delta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.cls_identities.4',
                      teacher_module='bbox_head.cls_identities.4',
                      output_hook=True,
                      methods=[dict(type='PGDClsLoss',
                                    name='loss_kd_cls_4',
                                    temp_s=temperature,
                                    temp_c=temperature,
                                    loss_weight=2.0,
                                    alpha=alpha,
                                    beta=beta,
                                    delta=delta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.reg_identities.0',
                      teacher_module='bbox_head.reg_identities.0',
                      output_hook=True,
                      methods=[dict(type='PGDRegLoss',
                                    name='loss_kd_reg_0',
                                    temp=temperature,
                                    gamma=gamma,
                                    delta=delta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.reg_identities.1',
                      teacher_module='bbox_head.reg_identities.1',
                      output_hook=True,
                      methods=[dict(type='PGDRegLoss',
                                    name='loss_kd_reg_1',
                                    temp=temperature,
                                    gamma=gamma,
                                    delta=delta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.reg_identities.2',
                      teacher_module='bbox_head.reg_identities.2',
                      output_hook=True,
                      methods=[dict(type='PGDRegLoss',
                                    name='loss_kd_reg_2',
                                    temp=temperature,
                                    gamma=gamma,
                                    delta=delta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.reg_identities.3',
                      teacher_module='bbox_head.reg_identities.3',
                      output_hook=True,
                      methods=[dict(type='PGDRegLoss',
                                    name='loss_kd_reg_3',
                                    temp=temperature,
                                    gamma=gamma,
                                    delta=delta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.reg_identities.4',
                      teacher_module='bbox_head.reg_identities.4',
                      output_hook=True,
                      methods=[dict(type='PGDRegLoss',
                                    name='loss_kd_reg_4',
                                    temp=temperature,
                                    gamma=gamma,
                                    delta=delta,
                                    )
                               ]
                      ),
                ]
    )
fp16 = dict(loss_scale=512.)
student_cfg = 'work_configs/detectors/autoassign_r50_distill_head.py'
teacher_cfg = 'work_configs/detectors/autoassign_r101_3x_ms.py'
