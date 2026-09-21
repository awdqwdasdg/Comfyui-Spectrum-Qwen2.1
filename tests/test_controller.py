from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectrum_qwen21.chebyshev import _Anchor  # noqa: E402
from spectrum_qwen21.config import SpectrumConfig  # noqa: E402
from spectrum_qwen21.controller import (  # noqa: E402
    decide_actual_or_forecast,
    find_step_index,
    note_decision,
)
from spectrum_qwen21.state import SpectrumBranchState  # noqa: E402


def _ready_branch(config: SpectrumConfig) -> SpectrumBranchState:
    """Branch whose forecaster pretends to be fully seeded."""
    state = SpectrumBranchState(config=config)
    forecaster = state.forecaster
    forecaster._feature_shape = torch.Size((1, 1))
    forecaster._feature_dtype = torch.float32
    forecaster._storage_device = torch.device("cpu")
    for i in range(config.chebyshev_degree + 1):
        forecaster._anchors.append(
            _Anchor(coord=-1.0 + 2.0 * i / config.chebyshev_degree, feature=torch.zeros(1, 1))
        )
    return state


class FindStepIndexTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        sigmas = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.2, 0.0])
        self.assertEqual(find_step_index(sigmas, torch.tensor([0.8])), 2)
        self.assertEqual(find_step_index(sigmas, torch.tensor([1.0])), 0)
        self.assertEqual(find_step_index(sigmas, torch.tensor([0.0])), 5)

    def test_batched_timestep_uses_first_row(self) -> None:
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.assertEqual(find_step_index(sigmas, torch.tensor([0.5, 0.5])), 1)

    def test_bracketing_fallback(self) -> None:
        sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4])
        # 0.7 lies between 0.8 and 0.6 -> index 1
        self.assertEqual(find_step_index(sigmas, torch.tensor([0.7])), 1)

    def test_nearest_fallback(self) -> None:
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.assertEqual(find_step_index(sigmas, torch.tensor([-5.0])), 2)

    def test_degenerate_inputs(self) -> None:
        self.assertEqual(find_step_index(torch.tensor([]), torch.tensor([1.0])), -1)
        self.assertEqual(find_step_index(torch.tensor([1.0]), torch.tensor([])), -1)


class ScheduleTest(unittest.TestCase):
    def _simulate(self, config: SpectrumConfig, total_steps: int):
        state = _ready_branch(config)
        decisions = []
        for step in range(total_steps):
            actual, reason = decide_actual_or_forecast(
                state=state,
                step_index=step,
                total_steps=total_steps,
                control_present=False,
                config=config,
            )
            decisions.append((actual, reason))
            note_decision(state, actual, reason, config)
        return decisions, state

    def test_default_schedule_40_steps_matches_paper_pattern(self) -> None:
        config = SpectrumConfig()
        decisions, state = self._simulate(config, total_steps=40)

        actual_steps = {i for i, (a, _) in enumerate(decisions) if a}
        # warmup 0-4, window-rule actuals, protected tail 38-39
        expected = {0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 25, 31, 38, 39}
        self.assertEqual(actual_steps, expected)

        forecasts = [i for i, (a, _) in enumerate(decisions) if not a]
        self.assertEqual(len(forecasts), 40 - len(expected))
        # first forecast happens right after warmup
        self.assertEqual(forecasts[0], 5)
        self.assertEqual(state.actual_count, len(expected))
        self.assertEqual(state.forecast_count, len(forecasts))
        # window grew by 0.75 per window-actual (7 window actuals after warmup)
        self.assertAlmostEqual(
            state.current_window,
            min(2.0 + 7 * 0.75, config.window_growth_cap),
            places=5,
        )
        # 26 forecasts / 40 steps ~= 2.9x transformer-pass skipping
        self.assertEqual(state.forecast_count, 26)

    def test_max_consecutive_cap_binds(self) -> None:
        config = SpectrumConfig(
            warmup_steps=0,
            tail_actual_steps=0,
            window_size=16.0,
            flex_window=0.0,
            max_consecutive_forecasts=1,
        )
        decisions, _ = self._simulate(config, total_steps=12)
        pattern = [a for a, _ in decisions]
        self.assertEqual(pattern.count(False), pattern.count(True))
        self.assertFalse(pattern[0])
        self.assertTrue(pattern[1])

    def test_history_missing_forces_actual(self) -> None:
        config = SpectrumConfig(warmup_steps=0, tail_actual_steps=0, window_size=2.0)
        state = SpectrumBranchState(config=config)  # forecaster not ready
        actual, reason = decide_actual_or_forecast(
            state=state, step_index=0, total_steps=10,
            control_present=False, config=config,
        )
        self.assertTrue(actual)
        self.assertEqual(reason, "insufficient_history")

    def test_control_guard(self) -> None:
        config = SpectrumConfig()
        state = _ready_branch(config)
        actual, reason = decide_actual_or_forecast(
            state=state, step_index=10, total_steps=20,
            control_present=True, config=config,
        )
        self.assertTrue(actual)
        self.assertEqual(reason, "control")

    def test_tail_protects_final_steps(self) -> None:
        config = SpectrumConfig(warmup_steps=0, tail_actual_steps=3)
        state = _ready_branch(config)
        for step in (7, 8, 9):
            actual, reason = decide_actual_or_forecast(
                state=state, step_index=step, total_steps=10,
                control_present=False, config=config,
            )
            self.assertTrue(actual)
            self.assertEqual(reason, "tail")

    def test_unknown_step_is_actual(self) -> None:
        config = SpectrumConfig()
        state = SpectrumBranchState(config=config)
        actual, _ = decide_actual_or_forecast(
            state=state, step_index=-1, total_steps=10,
            control_present=False, config=config,
        )
        self.assertTrue(actual)


class ConfigValidationTest(unittest.TestCase):
    def test_defaults(self) -> None:
        config = SpectrumConfig()
        config.validate()
        self.assertEqual(config.warmup_steps, 5)
        self.assertEqual(config.tail_actual_steps, 2)
        self.assertEqual(config.window_size, 2.0)
        self.assertEqual(config.flex_window, 0.75)
        self.assertEqual(config.max_consecutive_forecasts, 8)
        self.assertEqual(config.history_points, 8)
        self.assertEqual(config.chebyshev_degree, 4)
        self.assertEqual(config.ridge_lambda, 0.1)
        self.assertEqual(config.blend_weight, 0.5)
        self.assertEqual(config.cache_device, "main_device")
        self.assertTrue(config.force_actual_on_control)
        self.assertFalse(config.debug)

    def test_rejects_degree_larger_than_history(self) -> None:
        config = SpectrumConfig(chebyshev_degree=5, history_points=5)
        with self.assertRaises(ValueError):
            config.validate()

    def test_rejects_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            SpectrumConfig(warmup_steps=-1).validate()
        with self.assertRaises(ValueError):
            SpectrumConfig(blend_weight=1.5).validate()
        with self.assertRaises(ValueError):
            SpectrumConfig(window_size=0.5).validate()
        with self.assertRaises(ValueError):
            SpectrumConfig(cache_device="gpu").validate()

    def test_window_growth_cap(self) -> None:
        config = SpectrumConfig()
        self.assertEqual(config.window_growth_cap, 8.0)
        config = SpectrumConfig(window_size=12.0, history_points=4)
        self.assertEqual(config.window_growth_cap, 12.0)


if __name__ == "__main__":
    unittest.main()
