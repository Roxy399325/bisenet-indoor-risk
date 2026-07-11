import os
import os.path as osp


_DATA_ROOT = os.environ.get(
    'INDOOR_RISK_ROOT',
    '../dataset/ADE20K_BiSeNet_IndoorRisk_IndoorOnly',
)
if not osp.exists(_DATA_ROOT):
    _DATA_ROOT = './ADE20K_BiSeNet_IndoorRisk_IndoorOnly'

_PRETRAINED = os.environ.get(
    'INDOOR_RISK_PRETRAINED',
    './pretrained/model_final_v2_city.pth',
)


cfg = dict(
    model_type='bisenetv2',
    n_cats=6,
    num_aux_heads=4,
    lr_start=2e-3,
    weight_decay=1e-4,
    warmup_iters=1000,
    max_iter=20000,
    dataset='IndoorRiskDataset',
    im_root=_DATA_ROOT,
    train_im_anns=osp.join(_DATA_ROOT, 'train.txt'),
    val_im_anns=osp.join(_DATA_ROOT, 'val.txt'),
    scales=[0.5, 2.],
    cropsize=[512, 512],
    eval_crop=[512, 512],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5],
    eval_start_shortside=512,
    ims_per_gpu=16,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=False,
    finetune_from=_PRETRAINED,
    respth='./res_indoor_risk_v2',
)
