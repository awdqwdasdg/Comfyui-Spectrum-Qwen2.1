"""End-to-end test of the Spectrum wrapper against a fake Qwen-Image-2.1.

Runs a simulated sampling loop over the fake transformer replica from
``fake_model.py`` and verifies:

* per-step outputs of the patched model stay close to the unpatched model,
* the schedule matches the controller's expectations (real-forward count
  observed through a norm_out call counter),
* the run-completion bookkeeping and new-run reset work,
* forecast failures degrade to real forwards,
* non-Qwen-Image-2.1 cores pass through untouched,
* determinism (two identical runs give identical results).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_model import FakeQwenImage21Model, NotAQwenModel  # noqa: E402
from spectrum_qwen21.config import SpectrumConfig  # noqa: E402
from spectrum_qwen21.patcher import create_spectrum_wrapper  # noqa: E402


def _make_options(sigmas: torch.Tensor, with_wrapper=None) -> dict:
    options = {
        "sample_sigmas": sigmas,
        "cond_or_uncond": [0],
        "sigmas": sigmas,
    }
    if with_wrapper is not None:
        options["wrappers"] = {
            "diffusion_model": {"spectrum_qwen21": [with_wrapper]}
        }
    return options


class SpectrumWrapperIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeQwenImage21Model(seed=0)
        self.model.eval()
        self.config = SpectrumConfig()
        self.wrapper = create_spectrum_wrapper(self.config)
        self.root = self.wrapper.root_state

        self.steps = 20
        self.sigmas = torch.linspace(1.0, 0.0, self.steps + 1)
        self.context = torch.randn(1, self.model.txt_len, 16)
        self.x0 = torch.randn(1, self.model.in_channels, 8, 8)

        # Count real block-stack executions (the forecast head never runs
        # transformer_blocks, unlike norm_out which both paths execute).
        self.norm_calls = 0

        def counter(module, args):
            self.norm_calls += 1

        self._handle = self.model.transformer_blocks[0].register_forward_pre_hook(counter)

    def tearDown(self) -> None:
        self._handle.remove()

    def _run(self, x: torch.Tensor, options: dict):
        outputs = []
        current = x.clone()
        for i in range(self.steps):
            t = self.sigmas[i].reshape(1)
            out = self.model(
                current, t, self.context, transformer_options=options
            )
            outputs.append(out)
            # arbitrary but stable latent update
            current = current + 0.05 * out
        return outputs, current

    def test_forecast_outputs_stay_close_to_real_outputs(self) -> None:
        real_outputs, _ = self._run(self.x0, _make_options(self.sigmas))
        real_calls = self.norm_calls
        self.assertEqual(real_calls, self.steps)

        self.norm_calls = 0
        patched_outputs, _ = self._run(
            self.x0, _make_options(self.sigmas, self.wrapper)
        )

        # Schedule check: warmup 5 + window actuals + tail 2 for 20 steps.
        # Simulate the controller expectation for 20 steps:
        expected_actual = self._simulate_schedule(self.steps)
        self.assertEqual(self.norm_calls, expected_actual)
        self.assertLess(self.norm_calls, self.steps)

        max_rel = 0.0
        for i, (real, patched) in enumerate(zip(real_outputs, patched_outputs)):
            self.assertEqual(patched.shape, real.shape)
            self.assertTrue(torch.isfinite(patched).all())
            if i >= self.config.warmup_steps:
                rel = (
                    (patched - real).norm() / (real.norm() + 1e-8)
                ).item()
                max_rel = max(max_rel, rel)
        # forecast steps must stay within a few percent of the real output
        self.assertLess(max_rel, 0.05, f"max relative error too high: {max_rel}")

    def _simulate_schedule(self, total: int) -> int:
        from spectrum_qwen21.controller import (
            decide_actual_or_forecast,
            note_decision,
        )
        from spectrum_qwen21.state import SpectrumBranchState

        state = SpectrumBranchState(config=self.config)
        forecaster = state.forecaster
        forecaster._feature_shape = torch.Size((1, 1))
        forecaster._feature_dtype = torch.float32
        forecaster._storage_device = torch.device("cpu")
        from spectrum_qwen21.chebyshev import _Anchor

        count = 0
        recorded = 0
        for step in range(total):
            # mimic anchors recorded on real steps
            if recorded < self.config.chebyshev_degree + 1:
                forecaster._anchors.append(_Anchor(0.0, torch.zeros(1, 1)))
                recorded += 1
            actual, reason = decide_actual_or_forecast(
                state=state, step_index=step, total_steps=total,
                control_present=False, config=self.config,
            )
            note_decision(state, actual, reason, self.config)
            if actual:
                count += 1
                if recorded < self.config.history_points:
                    forecaster._anchors.append(_Anchor(0.0, torch.zeros(1, 1)))
                    recorded += 1
        return count

    def test_determinism(self) -> None:
        out_a, _ = self._run(self.x0, _make_options(self.sigmas, self.wrapper))
        wrapper_b = create_spectrum_wrapper(self.config)
        out_b, _ = self._run(self.x0, _make_options(self.sigmas, wrapper_b))
        for a, b in zip(out_a, out_b):
            self.assertTrue(torch.equal(a, b))

    def test_new_run_resets_state(self) -> None:
        self._run(self.x0, _make_options(self.sigmas, self.wrapper))
        first_branch = self.root.branch_states.get((0,))
        self.assertIsNotNone(first_branch)
        self.assertGreater(first_branch.actual_count, 0)

        # a new sigma tensor (new run) must reset the branch state
        new_sigmas = torch.linspace(0.9, 0.0, 15)
        self.norm_calls = 0
        self._run(self.x0, _make_options(new_sigmas, self.wrapper))
        second_branch = self.root.branch_states.get((0,))
        self.assertIsNotNone(second_branch)
        self.assertEqual(len(self.root.branch_states), 1)
        # history was released at the end of the first run
        self.assertEqual(first_branch.forecaster.history_size, 0)

    def test_forecast_failure_degrades_to_real(self) -> None:
        from unittest.mock import patch

        from spectrum_qwen21.chebyshev import HistoryWeightChebyshevForecaster

        def broken_predict(self, coord):
            raise RuntimeError("boom")

        self.norm_calls = 0
        with patch.object(
            HistoryWeightChebyshevForecaster, "predict", broken_predict
        ):
            outputs, _ = self._run(
                self.x0, _make_options(self.sigmas, self.wrapper)
            )
        # every forecast attempt degraded to a real forward
        self.assertEqual(self.norm_calls, self.steps)
        for out in outputs:
            self.assertEqual(tuple(out.shape), (1, self.model.out_channels, 8, 8))
            self.assertTrue(torch.isfinite(out).all())
        branch = self.root.branch_states.get((0,))
        self.assertIsNotNone(branch)
        self.assertEqual(branch.actual_count, self.steps)
        self.assertEqual(branch.forecast_count, 0)

    def test_control_present_forces_real(self) -> None:
        config = SpectrumConfig()
        wrapper = create_spectrum_wrapper(config)
        x = self.x0
        calls = 0

        def counter(module, args):
            nonlocal calls
            calls += 1

        handle = self.model.transformer_blocks[0].register_forward_pre_hook(counter)
        try:
            for i in range(self.steps):
                t = self.sigmas[i].reshape(1)
                opts = _make_options(self.sigmas, wrapper)
                opts["cond_or_uncond"] = [0]
                out = self.model(
                    x, t, self.context, transformer_options=opts, control=object()
                )
                x = x + 0.05 * out
        finally:
            handle.remove()
        self.assertEqual(calls, self.steps)

    def test_non_qwen_model_passthrough(self) -> None:
        # feed the wrapper a model object that fails the structural check
        model = NotAQwenModel()
        from fake_model import FakeExecutor

        captured = {}

        def executor(*args, **kwargs):
            captured["args"] = (args, kwargs)
            return model(args[0])

        wrapper = create_spectrum_wrapper(SpectrumConfig())
        opts = _make_options(self.sigmas, wrapper)
        opts["wrappers"]["diffusion_model"]["spectrum_qwen21"] = [wrapper]

        x = torch.randn(2, 4)
        out = FakeExecutor(model.forward, model, [wrapper]).execute(
            x, self.sigmas[0].reshape(1), None, None, None, opts
        )
        self.assertTrue(torch.equal(out, model(x)))
        # the model never became qwen-detected: branch state stays empty
        self.assertEqual(len(wrapper.root_state.branch_states), 0)

    def test_missing_sample_sigmas_runs_real(self) -> None:
        self.norm_calls = 0
        options = {"cond_or_uncond": [0]}
        for i in range(3):
            t = self.sigmas[i].reshape(1)
            self.model(
                self.x0, t, self.context, transformer_options={
                    **options, "wrappers": {"diffusion_model": {"spectrum_qwen21": [self.wrapper]}}
                }
            )
        self.assertEqual(self.norm_calls, 3)

    def test_debug_mode_runs_without_error(self) -> None:
        config = SpectrumConfig(debug=True)
        wrapper = create_spectrum_wrapper(config)
        outputs, _ = self._run(self.x0, _make_options(self.sigmas, wrapper))
        self.assertEqual(len(outputs), self.steps)


if __name__ == "__main__":
    unittest.main()
