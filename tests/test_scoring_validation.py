#!/usr/bin/env python3
"""Tests for controlled risk-score validation and weight injection."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lib.risk.bisenet_features import (
    RISK_COMPONENT_MAX_POINTS,
    analyze_bisenet,
)
from lib.risk.scoring_validation import (
    DEFAULT_WEIGHT_DELTAS,
    perturb_component_weights,
    run_monotonicity_validation,
    run_weight_sensitivity,
)


class ScoringValidationTests(unittest.TestCase):

    def test_controlled_sweeps_are_monotonic(self):
        report = run_monotonicity_validation()

        self.assertTrue(report['summary']['passed'])
        self.assertGreaterEqual(report['summary']['series_count'], 8)
        self.assertEqual(report['summary']['violation_count'], 0)
        self.assertTrue(all(item['passed'] for item in report['series']))

    def test_weight_perturbation_preserves_total(self):
        base = dict(RISK_COMPONENT_MAX_POINTS)
        changed = perturb_component_weights(base, 'slippery_surface', 0.20)

        self.assertAlmostEqual(sum(changed.values()), sum(base.values()))
        self.assertAlmostEqual(
            changed['slippery_surface'],
            base['slippery_surface'] * 1.20,
        )
        self.assertEqual(base, RISK_COMPONENT_MAX_POINTS)

    def test_weight_sensitivity_covers_every_component_and_delta(self):
        report = run_weight_sensitivity()
        expected_variants = len(RISK_COMPONENT_MAX_POINTS) * len(
            DEFAULT_WEIGHT_DELTAS
        )

        self.assertEqual(report['summary']['variant_count'], expected_variants)
        self.assertGreater(report['summary']['scenario_count'], 4)
        self.assertEqual(
            {item['level'] for item in report['baseline'].values()},
            {'low', 'medium', 'high', 'emergency'},
        )
        self.assertGreaterEqual(
            report['summary']['minimum_spearman_rank_correlation'],
            -1.0,
        )
        self.assertLessEqual(
            report['summary']['minimum_spearman_rank_correlation'],
            1.0,
        )
        self.assertTrue(report['acceptance']['passed'])
        self.assertEqual(report['summary']['total_non_boundary_level_flips'], 0)
        self.assertEqual(report['summary']['total_two_level_jumps'], 0)
        self.assertEqual(RISK_COMPONENT_MAX_POINTS['slippery_surface'], 20.0)

    def test_explicit_weights_change_only_the_requested_scoring_run(self):
        image = np.full((100, 120, 3), 180, dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[55:85, 40:80] = 4
        baseline = analyze_bisenet(image, mask)
        weights = perturb_component_weights(
            RISK_COMPONENT_MAX_POINTS,
            'slippery_surface',
            0.20,
        )

        changed = analyze_bisenet(
            image,
            mask,
            risk_component_max_points=weights,
        )
        repeated_baseline = analyze_bisenet(image, mask)

        self.assertGreater(
            changed.risk.component_scores['slippery_surface'],
            baseline.risk.component_scores['slippery_surface'],
        )
        self.assertTrue(changed.quality['risk_component_weights_overridden'])
        self.assertFalse(
            repeated_baseline.quality['risk_component_weights_overridden']
        )
        self.assertEqual(repeated_baseline.risk.score, baseline.risk.score)

    def test_invalid_explicit_weights_are_rejected(self):
        image = np.full((100, 120, 3), 180, dtype=np.uint8)
        mask = np.zeros((100, 120), dtype=np.uint8)

        with self.assertRaises(ValueError):
            analyze_bisenet(
                image,
                mask,
                risk_component_max_points={'slippery_surface': 100.0},
            )

        invalid = dict(RISK_COMPONENT_MAX_POINTS)
        invalid['low_light'] = -1.0
        with self.assertRaises(ValueError):
            analyze_bisenet(
                image,
                mask,
                risk_component_max_points=invalid,
            )


if __name__ == '__main__':
    unittest.main()
