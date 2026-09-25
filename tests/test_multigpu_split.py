"""Simulates ComfyUI MultiGPU CFG Split: cond on device A (stable
sample_sigmas object), uncond on device B (fresh .to(device) copy each call),
both sharing one wrapper closure, run in parallel threads."""
from __future__ import annotations
import sys, threading, unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fake_model import FakeQwenImage21Model  # noqa: E402
from spectrum_qwen21.config import SpectrumConfig  # noqa: E402
from spectrum_qwen21.patcher import create_spectrum_wrapper  # noqa: E402


class MultiGPUSplitTest(unittest.TestCase):
    def test_forecasts_happen_on_both_devices(self):
        torch.manual_seed(0)
        models = {"cuda:0": FakeQwenImage21Model(seed=0).eval(),
                  "cuda:1": FakeQwenImage21Model(seed=0).eval()}
        wrapper = create_spectrum_wrapper(SpectrumConfig(debug=False))
        steps = 25
        sigmas = torch.linspace(1.0, 0.0, steps + 1)
        ctx = torch.randn(1, models["cuda:0"].txt_len, 16)
        x = {"cuda:0": torch.randn(1, models["cuda:0"].in_channels, 8, 8)}
        x["cuda:1"] = x["cuda:0"].clone()
        wrappers = {"diffusion_model": {"spectrum_qwen21": [wrapper]}}

        def call(dev, branch, i):
            s = sigmas if dev == "cuda:0" else sigmas.clone()  # new id each call
            opts = {"sample_sigmas": s, "sigmas": s, "cond_or_uncond": [branch],
                    "multigpu_thread_device": dev, "wrappers": wrappers}
            out = models[dev](x[dev], sigmas[i].reshape(1), ctx, transformer_options=opts)
            x[dev] = x[dev] + 0.05 * out

        for i in range(steps):
            ts = [threading.Thread(target=call, args=("cuda:0", 0, i)),
                  threading.Thread(target=call, args=("cuda:1", 1, i))]
            [t.start() for t in ts]; [t.join() for t in ts]

        roots = list(getattr(wrapper, 'roots', {'d': wrapper.root_state}).values())
        fc = {k: b.forecast_count for r in {id(r): r for r in roots}.values()
              for k, b in r.branch_states.items()}
        print("forecast counts per branch:", fc)
        self.assertGreater(fc.get((0,), 0), 0)
        self.assertGreater(fc.get((1,), 0), 0)


if __name__ == "__main__":
    unittest.main()
