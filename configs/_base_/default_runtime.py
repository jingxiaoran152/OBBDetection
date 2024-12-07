checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50, #每隔50个epoch打印一次日志
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)] #训练多少个epoch
