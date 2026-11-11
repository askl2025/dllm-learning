"""CPU regression tests: python -m unittest discover -s tests -v"""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slow_fast_sampling import SlowFastSampler


class HorizonModel:
    """Predict token 1, with scripted horizons and one cautious fill per step."""

    device = torch.device("cpu")

    def __init__(self, horizons, cfg_scale=0.0, predictions=None):
        self.horizons = horizons
        self.predictions = predictions
        self.calls_per_step = 2 if cfg_scale > 0 else 1
        self.calls = 0

    def __call__(self, x, attention_mask=None):
        step = self.calls // self.calls_per_step
        self.calls += 1
        logits = torch.zeros((*x.shape, 5))
        logits[..., 1] = 0.1  # Confidence below the 0.3 horizon threshold.
        for row, horizons in enumerate(self.horizons):
            horizon = horizons[min(step, len(horizons) - 1)]
            if horizon < x.shape[1]:
                logits[row, horizon, 1] = math.log(16)  # Confidence 0.8.
            masked = (x[row, 1:] == 4).nonzero(as_tuple=True)[0]
            if masked.numel():
                logits[row, 1 + masked[0], 1] = math.log(36)  # Confidence 0.9.
        if self.predictions is not None:
            tokens = torch.tensor(self.predictions[min(step, len(self.predictions) - 1)])
            scores = logits[:, 1:, 1].clone()
            logits[:, 1:] = 0
            logits[:, 1:].scatter_(-1, tokens.unsqueeze(-1), scores.unsqueeze(-1))
        return SimpleNamespace(logits=logits)


class SlowPhaseTest(unittest.TestCase):
    def run_slow(self, horizons, *, cfg_scale=0.0, predictions=None, **kwargs):
        model = HorizonModel(horizons, cfg_scale, predictions)
        sampler = SlowFastSampler(
            model, {"block_length": 8, "gen_length": 8,
                    "high_confidence_threshold": 0.99, **kwargs},
            mask_id=4, cfg_scale=cfg_scale,
        )
        x = torch.full((len(horizons), 9), 4, dtype=torch.long)
        x[:, 0] = 2
        lengths = torch.full((len(horizons),), 8, dtype=torch.long)
        sampler.slow_phase(
            x, 1, 0, torch.zeros_like(lengths), lengths,
            x[:, 1:] == 4, x != 4, None,
        )
        return model, x, lengths

    def test_stable_horizon_exits_after_current_fill(self):
        for cfg_scale in (0.0, 1.0):
            with self.subTest(cfg_scale=cfg_scale):
                model, x, lengths = self.run_slow([[8]], cfg_scale=cfg_scale)
                self.assertEqual(model.calls, 2 * model.calls_per_step)
                self.assertEqual(lengths.tolist(), [8])
                self.assertEqual(x.tolist(), [[2, 1, 1, 4, 4, 4, 4, 4, 4]])

    def test_batch_waits_until_every_item_is_stable(self):
        model, x, lengths = self.run_slow([[8], [8, 4, 4]])
        self.assertEqual(model.calls, 3)
        self.assertEqual(lengths.tolist(), [8, 4])
        self.assertEqual((x[:, 1:] != 4).sum(dim=1).tolist(), [3, 3])

    def test_full_history_window_is_required(self):
        model, x, lengths = self.run_slow(
            [[8]], cycle_length_stability_window=3,
        )
        self.assertEqual(model.calls, 3)
        self.assertEqual(lengths.tolist(), [8])
        self.assertEqual((x[:, 1:] != 4).sum().item(), 3)

    def test_unstable_horizon_uses_budget_and_final_window_mean(self):
        model, x, lengths = self.run_slow(
            [[8, 4, 8, 4]], k_exploration_steps=4,
        )
        self.assertEqual(model.calls, 4)
        self.assertEqual(lengths.tolist(), [6])
        self.assertEqual((x[:, 1:] != 4).sum().item(), 4)

    def test_short_budget_keeps_history_fallback(self):
        model, x, lengths = self.run_slow(
            [[8, 4]], k_exploration_steps=2, cycle_length_stability_window=3,
        )
        self.assertEqual(model.calls, 2)
        self.assertEqual(lengths.tolist(), [6])
        self.assertEqual((x[:, 1:] != 4).sum().item(), 2)

    def test_consistency_is_disabled_by_default(self):
        predictions = [[[1] * 8], [[2] * 8]]
        for kwargs in ({}, {"prediction_consistency_threshold": None}):
            with self.subTest(kwargs=kwargs):
                model, _, _ = self.run_slow([[8]], predictions=predictions, **kwargs)
                self.assertEqual(model.calls, 2)

    def test_consistency_waits_until_predictions_stop_changing(self):
        for cfg_scale in (0.0, 1.0):
            with self.subTest(cfg_scale=cfg_scale):
                model, _, lengths = self.run_slow(
                    [[8]], predictions=[[[1] * 8], [[2] * 8]],
                    cfg_scale=cfg_scale, prediction_consistency_threshold=1.0,
                )
                self.assertEqual(model.calls, 3 * model.calls_per_step)
                self.assertEqual(lengths.tolist(), [8])

    def test_consistency_waits_for_each_batch_item(self):
        model, _, lengths = self.run_slow(
            [[8], [8]], predictions=[[[1] * 8, [1] * 8], [[1] * 8, [2] * 8]],
            prediction_consistency_threshold=1.0,
        )
        self.assertEqual(model.calls, 3)
        self.assertEqual(lengths.tolist(), [8, 8])

    def test_consistency_ignores_filled_positions_and_outside_region(self):
        model, _, lengths = self.run_slow(
            [[4]], predictions=[[[1] * 8], [[2, 1, 1, 1, 2, 2, 2, 2]]],
            prediction_consistency_threshold=1.0,
        )
        self.assertEqual(model.calls, 2)
        self.assertEqual(lengths.tolist(), [4])

    def test_consistency_threshold_uses_fraction_of_remaining_positions(self):
        for threshold, calls in ((0.5, 2), (0.75, 3)):
            with self.subTest(threshold=threshold):
                model, _, _ = self.run_slow(
                    [[5]], predictions=[[[1] * 8], [[2, 2, 2, 1, 1, 2, 2, 2]]],
                    prediction_consistency_threshold=threshold,
                )
                self.assertEqual(model.calls, calls)

    def test_consistency_still_requires_endpoint_stability(self):
        model, _, lengths = self.run_slow(
            [[8, 4, 8, 4]], k_exploration_steps=4,
            prediction_consistency_threshold=1.0,
        )
        self.assertEqual(model.calls, 4)
        self.assertEqual(lengths.tolist(), [6])

    def test_inconsistent_predictions_keep_updating_history_and_fall_back(self):
        model, _, lengths = self.run_slow(
            [[8, 8, 4, 8]], k_exploration_steps=4,
            predictions=[[[1] * 8], [[2] * 8], [[1] * 8], [[2] * 8]],
            prediction_consistency_threshold=1.0,
        )
        self.assertEqual(model.calls, 4)
        self.assertEqual(lengths.tolist(), [6])

    def test_consistency_requires_a_previous_prediction_even_at_zero_threshold(self):
        model, _, _ = self.run_slow(
            [[8]], cycle_length_stability_window=1,
            prediction_consistency_threshold=0.0,
        )
        self.assertEqual(model.calls, 2)

    def test_empty_region_does_not_wait_for_consistency(self):
        model, x, _ = self.run_slow(
            [[8]], cycle_length_stability_window=1, high_confidence_threshold=0.2,
            prediction_consistency_threshold=1.0,
        )
        self.assertEqual(model.calls, 2)
        self.assertFalse((x == 4).any())

    def test_consistency_threshold_rejects_invalid_values(self):
        for threshold in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "prediction_consistency_threshold"):
                    SlowFastSampler(None, {"prediction_consistency_threshold": threshold})

    def test_generate_finishes_after_early_slow_exit(self):
        for threshold in (None, 1.0):
            with self.subTest(threshold=threshold):
                model = HorizonModel([[8]])
                sampler = SlowFastSampler(
                    model, {"block_length": 8, "gen_length": 8,
                            "high_confidence_threshold": 0.99,
                            "prediction_consistency_threshold": threshold}, mask_id=4,
                )
                out = sampler.generate(torch.tensor([[2]]), attention_mask=None)
                self.assertEqual(out.tolist(), [[1] * 8])


if __name__ == "__main__":
    unittest.main()
