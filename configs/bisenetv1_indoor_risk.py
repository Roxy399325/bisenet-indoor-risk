import os
import os.path as osp


_DATA_ROOT = os.environ.get(
    'INDOOR_RISK_ROOT',
    '../dataset/ADE20K_BiSeNet_IndoorRisk_IndoorOnly',
)
if not osp.exists(_DATA_ROOT):
    _DATA_ROOT = './ADE20K_BiSeNet_IndoorRisk_IndoorOnly'


cfg = dict(
    model_type='bisenetv1',
    n_cats=6,
    num_aux_heads=2,
    lr_start=5e-3,
    weight_decay=1e-4,
    warmup_iters=1000,
    max_iter=40000,
    dataset='IndoorRiskDataset',
    im_root=_DATA_ROOT,
    train_im_anns=osp.join(_DATA_ROOT, 'train.txt'),
    val_im_anns=osp.join(_DATA_ROOT, 'val.txt'),
    scales=[0.5, 2.],
    cropsize=[512, 512],
    eval_crop=[512, 512],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5],
    eval_start_shortside=512,
    ims_per_gpu=8,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res_indoor_risk_v1',
)
