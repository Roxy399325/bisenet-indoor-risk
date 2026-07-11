#!/usr/bin/env python3
"""Run BiSeNet mask post-processing without loading a segmentation model.

This is useful when a mask has already been produced by another inference
script. The mask must be a single-channel image containing class IDs 0..5;
255 is treated as ignored.
"""

import argparse
import json
import os
import os.path as osp
import sys

import cv2

sys.path.insert(0, '.')

from lib.risk.bisenet_features import analyze_bisenet, render_analysis_overlay


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Original BGR image path.')
    parser.add_argument('--mask', required=True, help='Single-channel class-id mask path.')
    parser.add_argument('--output-dir', default='res_indoor_risk_v2/analysis')
    parser.add_argument('--prefix', help='Output filename prefix; defaults to image stem.')
    return parser.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError('Could not read image: {}'.format(args.image))
    if mask is None:
        raise FileNotFoundError('Could not read mask: {}'.format(args.mask))

    analysis = analyze_bisenet(image, mask)
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.prefix or osp.splitext(osp.basename(args.image))[0]
    features_path = osp.join(args.output_dir, '{}_features.json'.format(prefix))
    overlay_path = osp.join(args.output_dir, '{}_risk.png'.format(prefix))

    with open(features_path, 'w', encoding='utf-8') as handle:
        json.dump(analysis.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    overlay = render_analysis_overlay(image, analysis)
    if not cv2.imwrite(overlay_path, overlay):
        raise RuntimeError('Could not write risk overlay: {}'.format(overlay_path))

    print('Saved features: {}'.format(features_path))
    print('Saved risk overlay: {}'.format(overlay_path))
    print('Risk result: {} ({:.2f}/100)'.format(
        analysis.risk.level, analysis.risk.score
    ))


if __name__ == '__main__':
    main()
