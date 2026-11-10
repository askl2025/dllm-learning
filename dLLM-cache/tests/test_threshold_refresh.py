import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from dllm_cache.hooks.cache_hook_LLaDA import refresh_index


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ThresholdRefreshTest(unittest.TestCase):
    def setUp(self):
        similarities = [0.5, 0.8, 0.3, 0.9]
        self.current = torch.tensor(
            [[[s, math.sqrt(1 - s * s)] for s in similarities]], dtype=torch.float32
        )
        self.cached = torch.tensor([[[1.0, 0.0]] * 4], dtype=torch.float32)

    def select(self, ratio, threshold=None, metric="cosine"):
        return refresh_index(
            self.current, self.cached, transfer_ratio=ratio,
            similarity_threshold=threshold, change_metric=metric,
        )[0].tolist()

    def test_default_keeps_fixed_ratio(self):
        self.assertEqual(self.select(0.5), [2, 0])

    def test_threshold_is_capped_by_ratio(self):
        self.assertEqual(self.select(0.25, 0.7), [2])
        self.assertEqual(self.select(0.75, 0.7), [2, 0])

    def test_no_candidates_still_refreshes_one_token(self):
        self.assertEqual(self.select(0.25, 0.1), [2])

    def test_threshold_rejects_multi_sample_batch(self):
        current = self.current.expand(2, -1, -1)
        cached = self.cached.expand(2, -1, -1)
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            refresh_index(current, cached, transfer_ratio=0.25,
                          similarity_threshold=0.7)

    def test_relative_l2_detects_magnitude_change(self):
        self.current[0, 3] = self.cached[0, 3] * 3
        self.assertEqual(self.select(0.25), [2])
        self.assertEqual(self.select(0.25, metric="relative_l2"), [3])

    def test_relative_l2_rejects_cosine_threshold(self):
        with self.assertRaisesRegex(ValueError, "only supports cosine"):
            self.select(0.25, threshold=0.7, metric="relative_l2")


if __name__ == "__main__":
    unittest.main()
