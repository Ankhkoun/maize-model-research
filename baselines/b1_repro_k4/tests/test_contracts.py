"""Focused regression checks for alignment, supervision, migrations and queue rules."""
from __future__ import annotations

import json
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from af2_usa_a4_rf_20261005.data import (patch_targets, tile_grid_to_windows, windows_to_tile_grid,
                   _augment_sample, spatial_transform, _windows_to_tile_frames)
from af2_usa_a4_rf_20261005.reliability import A4WindowReliability
from af2_usa_a4_rf_20261005.objective import classification_loss
from af2_usa_a4_rf_20261005.targets import build_targets
from af2_usa_a4_rf_20261005.train import _target_batch, _load_init_checkpoint


class Contracts(unittest.TestCase):
    def test_batch4_delayed_target_shapes(self):
        torch.manual_seed(7)
        logits = torch.randn(4, 2, 256, 256)
        labels = torch.randint(0, 2, (4, 256, 256))
        labels[:, :2, :2] = 255
        target, valid, report = build_targets(logits, labels)
        self.assertEqual(tuple(target.shape), (4, 128, 128))
        self.assertEqual(tuple(valid.shape), (4, 128, 128))
        self.assertEqual(len(report["per_sample"]), 4)
        previous = target.clone()
        previous_valid = valid.clone()
        target_ema, valid_ema, _ = build_targets(
            logits, labels, previous, previous_valid, ema=0.95
        )
        self.assertEqual(tuple(target_ema.shape), (4, 128, 128))
        self.assertTrue(torch.isfinite(target_ema).all())
        self.assertTrue(torch.equal(valid_ema, valid))

    def test_all_window_and_patch_coordinates(self):
        grid = torch.arange(128 * 128).reshape(1, 128, 128)
        windows = tile_grid_to_windows(grid)
        self.assertTrue(torch.equal(windows_to_tile_grid(windows), grid))
        cube = torch.arange(26 * 10 * 256 * 256, dtype=torch.float32).reshape(1, 26, 10, 256, 256)
        for window in range(16):
            patches = A4WindowReliability._window_patches(cube, window)
            self.assertEqual(tuple(patches.shape), (1, 1024, 26, 40))
            for token in (0, 31, 32, 1023):
                row = window // 4 * 64 + token // 32 * 2
                col = window % 4 * 64 + token % 32 * 2
                expected = cube[0, :, :, row:row + 2, col:col + 2].reshape(26, 40)
                self.assertTrue(torch.equal(patches[0, token], expected))

    def test_patch_purity_fraction_and_ignore(self):
        label = np.full((256, 256), 255, dtype=np.uint8)
        label[:2, :2] = [[1, 1], [1, 0]]
        label[:2, 2:4] = [[0, 0], [255, 255]]
        label[:2, 4:6] = [[0, 1], [1, 0]]
        cls, valid, purity, fraction = patch_targets(label)
        self.assertEqual((cls[0, 0], valid[0, 0], purity[0, 0], fraction[0, 0]), (1, True, 0.75, 1))
        self.assertEqual((valid[0, 1], purity[0, 1], fraction[0, 1]), (False, 1, 0.5))
        self.assertEqual((valid[0, 2], purity[0, 2], fraction[0, 2]), (False, 0.5, 1))
        self.assertEqual((valid[2, 2], purity[2, 2], fraction[2, 2]), (False, 0, 0))
        logits = torch.randn(1, 2, 2, 2, requires_grad=True)
        target = torch.tensor([[[1, 255], [0, 255]]])
        loss = classification_loss(logits, target)
        loss.backward()
        self.assertTrue(bool((logits.grad[0, :, :, 1] == 0).all()))
        empty = classification_loss(logits.detach(), torch.full_like(target, 255))
        self.assertEqual(float(empty), 0)

    def test_delayed_target_augmentation_alignment(self):
        values = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
        valid = (values % 3) == 0
        bank = {"sample": (values.clone(), valid.clone())}
        for k, flip_h, flip_v in itertools.product(range(4), (False, True), (False, True)):
            transform = (k, flip_h, flip_v)
            target, mask = _target_batch({"sample_id": ["sample"], "spatial_transform": [transform]}, bank, torch.device("cpu"))
            expected = torch.rot90(values, k, (-2, -1))
            expected_mask = torch.rot90(valid, k, (-2, -1))
            if flip_h:
                expected, expected_mask = expected.flip(-1), expected_mask.flip(-1)
            if flip_v:
                expected, expected_mask = expected.flip(-2), expected_mask.flip(-2)
            self.assertTrue(torch.equal(target[0], expected))
            self.assertTrue(torch.equal(mask[0], expected_mask))
        self.assertTrue(torch.equal(bank["sample"][0], values))

    def test_synchronous_cube_support_and_metadata_augmentation(self):
        base = torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256)
        label = (base.long() % 3 == 0).long()
        label[30:100, 50:60] = 255
        source = ((base.long() % 4) + 1).to(torch.uint8).repeat(26, 1, 1)
        support_grid = (source == 1).float().reshape(26, 128, 2, 128, 2).mean((2, 4))
        support = tile_grid_to_windows(support_grid).permute(1, 2, 0)
        for seed in range(8):
            sample = {"agrifm_cube": base[None,None].clone(), "a4_cube": (base * 2)[None,None],
                      "target": label.clone(), "valid_pixel_mask": label != 255,
                      "source_map": source.clone(), "observed_support": support.clone()}
            transformed = _augment_sample(sample, generator=torch.Generator().manual_seed(seed))
            transform = transformed["spatial_transform"]
            self.assertTrue(torch.equal(transformed["agrifm_cube"][0,0], spatial_transform(base, transform)))
            self.assertTrue(torch.equal(transformed["a4_cube"][0,0], spatial_transform(base * 2, transform)))
            actual_support = _windows_to_tile_frames(transformed["observed_support"].permute(2,0,1))
            expected_support = (transformed["source_map"] == 1).float().reshape(26,128,2,128,2).mean((2,4))
            self.assertTrue(torch.equal(actual_support, expected_support))
            _, _, purity, fraction = patch_targets(transformed["target"].numpy())
            self.assertTrue(torch.equal(transformed["pseudo_purity"], torch.from_numpy(tile_grid_to_windows(purity))))
            self.assertTrue(torch.equal(transformed["pseudo_valid_fraction"], torch.from_numpy(tile_grid_to_windows(fraction))))

    def test_continuous_support_and_metadata_reach_experts(self):
        torch.manual_seed(42)
        head = A4WindowReliability().eval()
        cube = torch.randn(1, 4, 26, 40)
        context = torch.randn(1, 4, 4, 128)
        doy = torch.arange(26)[None].float() * 7 + 90
        observed_inputs = []
        hook = head.support_embedding.register_forward_pre_hook(lambda _, args: observed_inputs.append(args[0].clone()))
        args = [cube, context, doy, torch.full((1,4,26), 0.25), torch.ones(1,4).long(), torch.ones(1,4).bool(), torch.full((1,4),0.75), torch.ones(1,4)]
        with torch.no_grad():
            low = head._encode_window(*args)["reliability"]
            args[3] = torch.full((1,4,26), 0.75)
            high = head._encode_window(*args)["reliability"]
            args[6], args[7] = torch.full((1,4),0.5), torch.full((1,4),0.5)
            mixed = head._encode_window(*args)["reliability"]
            args[3] = torch.zeros(1,4,26)
            missing = head._encode_window(*args)["reliability"]
        hook.remove()
        self.assertTrue(bool((observed_inputs[0] == 0.25).all()))
        self.assertTrue(bool((observed_inputs[1] == 0.75).all()))
        self.assertGreater(float((low - high).abs().max()), 1e-7)
        self.assertGreater(float((high - mixed).abs().max()), 1e-7)
        self.assertTrue(bool(torch.isfinite(missing).all()) and bool((missing == 0).all()))

    def test_expert_counts_one_through_six_preserve_temporal_group_axis(self):
        torch.manual_seed(42)
        cube = torch.randn(1, 4, 4, 40)
        context = torch.randn(1, 4, 4, 16)
        doy = torch.tensor([[90.0, 150.0, 220.0, 280.0]])
        support = torch.ones(1, 4, 4)
        pseudo_class = torch.ones(1, 4).long()
        pseudo_valid = torch.ones(1, 4).bool()
        for count in range(1, 7):
            head = A4WindowReliability(dim=16, frames=4, num_experts=count).eval()
            with torch.no_grad():
                result = head._encode_window(
                    cube, context, doy, support, pseudo_class, pseudo_valid,
                    torch.ones(1, 4), torch.ones(1, 4),
                )
            self.assertEqual(tuple(result["route"].shape), (1, 4, 4, count))
            self.assertEqual(tuple(result["importance"].shape), (1, 4, count))
            self.assertEqual(tuple(result["expert_features"].shape), (1, 4, count, 4, 128))
            self.assertTrue(bool(torch.isfinite(result["reliability"]).all()))
            self.assertTrue(bool(torch.all(torch.diff(result["centers"]) > 0)))
        legacy = A4WindowReliability().ordered_centers().detach()
        self.assertTrue(torch.allclose(legacy, torch.tensor([0.15, 0.40, 0.65, 0.85]), atol=1e-5))

    def test_detached_weighted_classification(self):
        logits = torch.randn(1,2,4,4, requires_grad=True)
        weights = torch.rand(1,4,4, requires_grad=True)
        target = torch.zeros(1,4,4).long()
        loss = classification_loss(logits, target, weights, weighted=True, weight_floor=0.25)
        grad_logits, grad_weights = torch.autograd.grad(loss, (logits, weights), allow_unused=True)
        self.assertIsNotNone(grad_logits)
        self.assertIsNone(grad_weights)

    def test_migration_resets_whole_changed_experts(self):
        class MiniModel(nn.Module):
            def __init__(self, metadata):
                super().__init__()
                self.classifier = nn.Linear(3,2)
                self.reliability = A4WindowReliability(use_patch_metadata=metadata)
        old, new = MiniModel(False), MiniModel(True)
        before = new.reliability.experts[0][3].weight.detach().clone()
        report = _load_init_checkpoint(new, {"model": old.state_dict(), "config": {"model": {"use_patch_metadata": False}}})
        self.assertTrue(torch.equal(old.classifier.weight, new.classifier.weight))
        self.assertTrue(torch.equal(before, new.reliability.experts[0][3].weight))
        self.assertEqual(len([x for x in report["reinitialized_modules"] if "experts" in x]), 4)
        state = old.state_dict()
        state["classifier.weight"] = torch.zeros(7,7)
        with self.assertRaises(RuntimeError):
            _load_init_checkpoint(new, {"model": state})



def run_checks() -> dict:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
        result = unittest.TestResult()
        suite.run(result)
        if result.errors or result.failures:
            raise AssertionError(str(result.errors + result.failures))
        return {"tests_run": result.testsRun, "passed": True}
    finally:
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    print(json.dumps(run_checks()))
