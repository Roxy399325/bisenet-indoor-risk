#!/usr/bin/env python3
"""Unit tests for the BiSeNet mask-to-feature pipeline."""

import json
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lib.risk.bisenet_features import (
    _perspective_envelope,
    analyze_bisenet,
    render_analysis_overlay,
)


class BisenetFeatureTests(unittest.TestCase):

    def setUp(self):
        self.image = np.full((100, 120, 3), 180, dtype=np.uint8)

    def test_empty_walkable_scene_is_low_risk(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        result = analyze_bisenet(self.image, mask)

        self.assertTrue(result.mask_valid)
        self.assertTrue(result.features['corridor_valid'])
        self.assertEqual(result.features['obstacle_count'], 0)
        self.assertEqual(result.risk.level, 'low')
        self.assertEqual(result.features['walkable_surface_area_ratio'], 1.0)

    def test_small_noise_is_removed_but_real_obstacle_remains(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[20:45, 45:75] = 3
        mask[3, 3] = 3
        result = analyze_bisenet(self.image, mask)

        self.assertEqual(result.features['obstacle_count'], 1)
        self.assertEqual(result.obstacle_instances[0]['bbox'], [45, 20, 30, 25])
        self.assertGreater(result.features['obstacle_area_ratio'], 0.0)
        self.assertGreaterEqual(len(result.obstacle_instances[0]['contour']), 4)
        self.assertNotEqual(result.cleaned_mask[3, 3], 3)

    def test_corridor_overlap_distinguishes_central_and_side_obstacles(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[55:80, 50:70] = 3
        mask[20:35, 5:20] = 3
        result = analyze_bisenet(self.image, mask)

        self.assertEqual(result.features['obstacle_count'], 2)
        self.assertEqual(result.features['corridor_obstacle_count'], 1)
        self.assertGreater(result.features['obstacle_channel_overlap_ratio'], 0.0)
        self.assertLess(result.features['obstacle_channel_overlap_ratio'], 1.0)

    def test_lower_side_obstacle_is_not_part_of_central_corridor(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[55:80, 5:20] = 3
        result = analyze_bisenet(self.image, mask)

        self.assertTrue(result.features['corridor_valid'])
        self.assertEqual(result.features['corridor_obstacle_count'], 0)
        self.assertEqual(result.features['obstacle_channel_overlap_ratio'], 0.0)
        self.assertFalse(result.obstacle_instances[0]['in_corridor'])

    def test_top_only_walkable_patch_does_not_form_valid_corridor(self):
        mask = np.ones((100, 120), dtype=np.uint8)
        mask[36:60, 20:100] = 0
        result = analyze_bisenet(self.image, mask)

        self.assertFalse(result.features['corridor_valid'])
        self.assertTrue(result.features['corridor_fallback_used'])
        self.assertIsNone(result.features['corridor_obstacle_occupancy'])
        self.assertNotIn('corridor_obstacle_occupancy', result.available_features)
        self.assertIn('corridor_obstacle_occupancy', result.unavailable_features)
        self.assertEqual(result.risk.level, 'medium')

    def test_fallback_envelope_has_narrow_top_and_wide_bottom(self):
        envelope = _perspective_envelope(100, 120)
        top_width = int(envelope[35].sum())
        bottom_width = int(envelope[-1].sum())

        self.assertGreater(bottom_width, top_width * 2)
        self.assertAlmostEqual(top_width / 120.0, 0.20, delta=0.03)
        self.assertAlmostEqual(bottom_width / 120.0, 0.60, delta=0.03)

    def test_slippery_step_and_low_light_features_are_reported(self):
        image = np.full((100, 120, 3), 30, dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[55:70, 45:75] = 4
        mask[35:50, 48:72] = 5
        result = analyze_bisenet(image, mask)

        self.assertGreater(result.features['corridor_slippery_ratio'], 0.0)
        self.assertGreater(result.features['slippery_surface_score'], 0.0)
        self.assertEqual(result.features['step_threshold_count'], 1)
        self.assertTrue(result.features['low_light_flag'])
        self.assertIn('low_light', result.risk.component_scores)
        self.assertTrue(result.risk.suggestions)

    def test_all_ignore_is_not_reported_as_low_risk(self):
        mask = np.full((100, 120), 255, dtype=np.uint8)
        result = analyze_bisenet(self.image, mask)

        self.assertFalse(result.mask_valid)
        self.assertFalse(result.features['corridor_valid'])
        self.assertNotEqual(result.risk.level, 'low')
        self.assertIn('depth_change_score', result.unavailable_features)

    def test_valid_mask_without_corridor_is_not_low_risk(self):
        mask = np.full((100, 120), 1, dtype=np.uint8)
        result = analyze_bisenet(self.image, mask)

        self.assertTrue(result.mask_valid)
        self.assertFalse(result.features['corridor_valid'])
        self.assertEqual(result.risk.level, 'medium')
        self.assertAlmostEqual(
            result.risk.score,
            sum(result.risk.component_scores.values()),
            places=6,
        )

    def test_tiny_valid_fraction_is_not_a_valid_mask(self):
        mask = np.full((100, 120), 255, dtype=np.uint8)
        mask[90:100, 40:80] = 0
        result = analyze_bisenet(self.image, mask)

        self.assertFalse(result.mask_valid)
        self.assertLess(result.features['valid_pixel_ratio'], 0.80)
        self.assertEqual(result.risk.level, 'medium')

    def test_json_contract_and_overlay(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[60:80, 50:70] = 3
        result = analyze_bisenet(self.image, mask)
        payload = result.to_dict()
        json.dumps(payload, ensure_ascii=False)
        overlay = render_analysis_overlay(self.image, result)

        self.assertEqual(overlay.shape, self.image.shape)
        self.assertIn('features', payload)
        self.assertIn('risk', payload)
        self.assertIn('obstacles', payload)

    def test_invalid_mask_class_is_rejected(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[0, 0] = 6
        with self.assertRaises(ValueError):
            analyze_bisenet(self.image, mask)


if __name__ == '__main__':
    unittest.main()
