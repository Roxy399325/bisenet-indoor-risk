#!/usr/bin/env python3
"""Visualize IndoorRisk segmentation predictions on one image.

When --img-path is omitted, a sample is selected from the validation split in
the supplied config. The output image contains the source image, ground truth,
prediction, and a prediction overlay.
"""

import argparse
import json
import math
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import lib.data.transform_cv2 as T
from configs import set_cfg_from_file
from lib.models import model_factory
from lib.risk.bisenet_features import (
    CLASS_COLORS,
    CLASS_NAMES,
    analyze_bisenet,
    render_analysis_overlay,
)
from lib.risk.yolo import UltralyticsYoloAdapter


DEFAULT_CLASS_NAMES = list(CLASS_NAMES)

INDOOR_RISK_MEAN = (0.52594033, 0.46734318, 0.41189465)
INDOOR_RISK_STD = (0.24811083, 0.24959911, 0.25970083)
SERVER_YOLO_WEIGHT = (
    '/data/users/jianfei/results/yolo_results/'
    'yolov8s_fall_hazard_indoor_v6_extended2/weights/best.pt'
)


def default_yolo_weight():
    configured = os.environ.get('YOLO_WEIGHT_PATH')
    if configured:
        return configured
    return SERVER_YOLO_WEIGHT if osp.isfile(SERVER_YOLO_WEIGHT) else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default=osp.join(PROJECT_ROOT, 'configs', 'bisenetv2_indoor_risk.py'),
    )
    parser.add_argument(
        '--weight-path',
        default=osp.join(PROJECT_ROOT, 'res_indoor_risk_v2', 'model_final.pth'),
    )
    parser.add_argument(
        '--img-path', '--image-path', dest='img_path',
        help='Image to segment. Defaults to a validation image.',
    )
    parser.add_argument('--label-path', help='Optional grayscale ground-truth mask.')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Validation image index when --img-path is omitted.')
    parser.add_argument('--output', help='Output comparison image path.')
    parser.add_argument('--mask-output', help='Raw uint8 class-id mask output path.')
    parser.add_argument('--features-output', help='JSON feature and risk output path.')
    parser.add_argument('--analysis-output', help='Risk overlay output path.')
    parser.add_argument(
        '--yolo-weight', '--yolo', dest='yolo_weight',
        default=default_yolo_weight(),
        help=(
            'Optional trained YOLO .pt checkpoint. Defaults to YOLO_WEIGHT_PATH '
            'or the documented /data/users/jianfei checkpoint when available.'
        ),
    )
    parser.add_argument('--yolo-confidence', type=float, default=0.25,
                        help='Minimum YOLO detection confidence (default: 0.25).')
    parser.add_argument('--yolo-iou', type=float, default=0.70,
                        help='YOLO NMS IoU threshold (default: 0.70).')
    parser.add_argument(
        '--yolo-device',
        help='Ultralytics device, for example 0, 1, or cpu. Defaults to BiSeNet device.',
    )
    return parser.parse_args()


def resolve_project_path(path):
    if osp.isabs(path):
        return path
    if osp.exists(path):
        return osp.abspath(path)
    return osp.join(PROJECT_ROOT, path)


def read_validation_pair(cfg, sample_index):
    with open(cfg.val_im_anns, 'r', encoding='utf-8') as handle:
        samples = [line.strip() for line in handle if line.strip()]
    if not samples:
        raise RuntimeError('The validation annotation file is empty: {}'.format(cfg.val_im_anns))
    if not 0 <= sample_index < len(samples):
        raise ValueError('--sample-index must be in [0, {}]'.format(len(samples) - 1))

    line = samples[sample_index]
    paths = line.split(',', 1) if ',' in line else line.split()
    if len(paths) != 2:
        raise RuntimeError('Invalid validation annotation line: {}'.format(line))
    img_path, label_path = (item.strip() for item in paths)
    if not osp.isabs(img_path):
        img_path = osp.join(cfg.im_root, img_path)
    if not osp.isabs(label_path):
        label_path = osp.join(cfg.im_root, label_path)
    return img_path, label_path


def load_class_names(data_root, n_classes):
    names = list(DEFAULT_CLASS_NAMES[:n_classes])
    label_file = osp.join(data_root, 'labels.txt')
    if not osp.exists(label_file):
        return names

    names = ['class_{}'.format(index) for index in range(n_classes)]
    with open(label_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            class_id = int(parts[0])
            if 0 <= class_id < n_classes:
                names[class_id] = parts[1]
    return names


def colorize(mask, n_classes):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = CLASS_COLORS[class_id % len(CLASS_COLORS)]
    return color_mask


def draw_title(image, title):
    canvas = cv2.copyMakeBorder(image, 34, 0, 0, 0, cv2.BORDER_CONSTANT,
                                value=(255, 255, 255))
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (25, 25, 25), 2, cv2.LINE_AA)
    return canvas


def draw_legend(width, class_names, n_classes):
    line_height = 28
    legend = np.full((line_height * n_classes + 12, width, 3), 255, dtype=np.uint8)
    for class_id, name in enumerate(class_names):
        y = 24 + class_id * line_height
        color = tuple(int(value) for value in CLASS_COLORS[class_id % len(CLASS_COLORS)])
        cv2.rectangle(legend, (12, y - 16), (32, y + 4), color, thickness=-1)
        cv2.putText(legend, '{}: {}'.format(class_id, name), (44, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
    return legend


def make_comparison(image, prediction, label, class_names, n_classes):
    prediction_color = colorize(prediction, n_classes)
    panels = [image]
    titles = ['Input image']
    if label is not None:
        panels.append(colorize(label, n_classes))
        titles.append('Ground truth')
    panels.extend([prediction_color, cv2.addWeighted(image, 0.60, prediction_color, 0.40, 0)])
    titles.extend(['Prediction', 'Prediction overlay'])

    max_width = 640
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width)
    panel_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    panels = [cv2.resize(panel, panel_size, interpolation=cv2.INTER_NEAREST)
              for panel in panels]
    panels = [draw_title(panel, title) for panel, title in zip(panels, titles)]
    row = cv2.hconcat(panels)
    legend = draw_legend(row.shape[1], class_names, n_classes)
    return cv2.vconcat([row, legend])


def print_single_image_metrics(prediction, label, class_names, n_classes):
    valid = label != 255
    if not np.any(valid):
        print('Ground truth contains only ignored pixels; no metrics were computed.')
        return

    pixel_accuracy = (prediction[valid] == label[valid]).mean()
    ious = []
    print('Single-image metrics: pixel_accuracy={:.4f}'.format(pixel_accuracy))
    for class_id, name in enumerate(class_names):
        pred_class = prediction == class_id
        label_class = label == class_id
        union = np.logical_and(valid, np.logical_or(pred_class, label_class)).sum()
        if union == 0:
            continue
        iou = np.logical_and(pred_class, label_class).sum() / union
        ious.append(iou)
        print('  {:<20} IoU={:.4f}'.format(name, iou))
    if ious:
        print('  mean IoU={:.4f}'.format(float(np.mean(ious))))


def main():
    args = parse_args()
    cfg = set_cfg_from_file(resolve_project_path(args.config))
    for field in ('im_root', 'train_im_anns', 'val_im_anns', 'respth'):
        value = getattr(cfg, field, None)
        if value:
            setattr(cfg, field, resolve_project_path(value))

    if args.img_path is None:
        img_path, default_label_path = read_validation_pair(cfg, args.sample_index)
    else:
        img_path, default_label_path = args.img_path, None
    label_path = args.label_path if args.label_path is not None else default_label_path

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError('Could not read image: {}'.format(img_path))
    label = None
    if label_path is not None:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError('Could not read label: {}'.format(label_path))
        if label.shape != image.shape[:2]:
            label = cv2.resize(label, (image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    # The requested checkpoint contains the trained backbone. Avoid an
    # unnecessary online backbone download during local/offline inference.
    os.environ.setdefault('BISENET_SKIP_BACKBONE_PRETRAIN', '1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
    weight_path = resolve_project_path(args.weight_path)
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    load_result = net.load_state_dict(state_dict, strict=False)
    expected_aux_prefixes = ('aux2.', 'aux3.', 'aux4.', 'aux5_4.')
    unexpected = [
        key for key in load_result.unexpected_keys
        if not key.startswith(expected_aux_prefixes)
    ]
    if load_result.missing_keys or unexpected:
        raise RuntimeError(
            'Checkpoint is incompatible. missing_keys={}, unexpected_keys={}'.format(
                load_result.missing_keys, unexpected
            )
        )
    net.eval().to(device)

    to_tensor = T.ToTensor(mean=INDOOR_RISK_MEAN, std=INDOOR_RISK_STD)
    image_rgb = image[:, :, ::-1].copy()
    tensor = to_tensor(dict(im=image_rgb, lb=None))['im'].unsqueeze(0).to(device)
    original_size = tensor.shape[-2:]
    padded_size = tuple(math.ceil(size / 32) * 32 for size in original_size)

    with torch.no_grad():
        pad_height = padded_size[0] - original_size[0]
        pad_width = padded_size[1] - original_size[1]
        tensor = F.pad(tensor, (0, pad_width, 0, pad_height), value=0.0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = net(tensor)[0]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        logits = F.interpolate(logits, size=padded_size, mode='bilinear', align_corners=False)
        logits = logits[:, :, :original_size[0], :original_size[1]]
        prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    class_names = load_class_names(cfg.im_root, cfg.n_cats)
    comparison = make_comparison(image, prediction, label, class_names, cfg.n_cats)

    output_path = args.output
    if output_path is None:
        output_dir = osp.join(cfg.respth, 'visualizations')
        output_path = osp.join(output_dir, '{}_comparison.png'.format(
            osp.splitext(osp.basename(img_path))[0]))
    output_dir = osp.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not cv2.imwrite(output_path, comparison):
        raise RuntimeError('Could not write output image: {}'.format(output_path))

    # Keep the colour comparison above for human inspection, but pass the
    # original integer prediction to the machine-readable risk pipeline.
    yolo_detections = None
    yolo_elapsed_ms = None
    if args.yolo_weight:
        yolo_weight = resolve_project_path(args.yolo_weight)
        yolo_device = args.yolo_device
        if yolo_device is None:
            yolo_device = '0' if device.type == 'cuda' else 'cpu'
        yolo = UltralyticsYoloAdapter(
            yolo_weight,
            device=yolo_device,
            confidence=args.yolo_confidence,
            iou=args.yolo_iou,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        yolo_start = time.perf_counter()
        yolo_detections = yolo.predict(image)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        yolo_elapsed_ms = (time.perf_counter() - yolo_start) * 1000

    analysis = analyze_bisenet(
        image,
        prediction,
        yolo_detections=yolo_detections,
    )
    sample_stem = osp.splitext(osp.basename(img_path))[0]
    analysis_dir = osp.join(cfg.respth, 'analysis')
    mask_output = args.mask_output or osp.join(analysis_dir, '{}_mask.png'.format(sample_stem))
    features_output = args.features_output or osp.join(
        analysis_dir, '{}_features.json'.format(sample_stem)
    )
    analysis_output = args.analysis_output or osp.join(
        analysis_dir, '{}_risk.png'.format(sample_stem)
    )
    for path in (mask_output, features_output, analysis_output):
        parent = osp.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    if not cv2.imwrite(mask_output, prediction):
        raise RuntimeError('Could not write raw mask: {}'.format(mask_output))
    with open(features_output, 'w', encoding='utf-8') as handle:
        json.dump(analysis.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    analysis_overlay = render_analysis_overlay(image, analysis)
    if not cv2.imwrite(analysis_output, analysis_overlay):
        raise RuntimeError('Could not write risk overlay: {}'.format(analysis_output))

    print('Model device: {}'.format(device))
    print('Input image: {}'.format(img_path))
    print('Inference time: {:.2f} ms'.format(elapsed_ms))
    if yolo_elapsed_ms is not None:
        print('YOLO inference time: {:.2f} ms'.format(yolo_elapsed_ms))
        print('YOLO detections: {}'.format(len(yolo_detections)))
    print('Saved comparison: {}'.format(output_path))
    print('Saved raw mask: {}'.format(mask_output))
    print('Saved features: {}'.format(features_output))
    print('Saved risk overlay: {}'.format(analysis_output))
    print('Risk result: {} ({:.2f}/100)'.format(
        analysis.risk.level, analysis.risk.score
    ))
    if label is not None:
        print_single_image_metrics(prediction, label, class_names, cfg.n_cats)


if __name__ == '__main__':
    main()
