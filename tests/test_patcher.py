"""Tests for apply_spectrum registration via stubbed comfy modules."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_model import FakeQwenImage21Model  # noqa: E402

_PKG_DIR = Path(__file__).resolve().parent.parent
_PKG_NAME = "spectrum_qwenimage21_root"


def _load_node_package():
    """Load the custom-node folder the way ComfyUI loads custom nodes.

    The folder (with hyphens in its name) is loaded as a package via
    importlib with a synthetic module name; ``__init__.py`` then imports
    ``nodes.py`` with its relative imports, mirroring the real runtime.
    """
    if _PKG_NAME in sys.modules:
        return sys.modules[_PKG_NAME]
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME,
        _PKG_DIR / "__init__.py",
        submodule_search_locations=[str(_PKG_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_NAME] = module
    spec.loader.exec_module(module)
    return module


class _StubModelPatcher:
    """Minimal ModelPatcher stand-in."""

    def __init__(self, core):
        self.core = core
        self.model_options = {}

    def clone(self, *args, **kwargs):
        clone = _StubModelPatcher(self.core)
        # ComfyUI clones carry over model_options
        clone.model_options = {
            "transformer_options": {
                k: v for k, v in self.model_options.get("transformer_options", {}).items()
            }
        }
        return clone

    def get_model_object(self, name):
        return self.core


def _install_comfy_stubs() -> None:
    comfy = types.ModuleType("comfy")
    patcher_extension = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"
        APPLY_MODEL = "apply_model"

    def add_wrapper_with_key(wrapper_type, key, wrapper, options, is_model_options=False):
        if is_model_options:
            options = options.setdefault("transformer_options", {})
        wrappers = options.setdefault("wrappers", {})
        wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    patcher_extension.WrappersMP = WrappersMP  # type: ignore[attr-defined]
    patcher_extension.add_wrapper_with_key = add_wrapper_with_key  # type: ignore[attr-defined]
    comfy.patcher_extension = patcher_extension  # type: ignore[attr-defined]

    sys.modules.setdefault("comfy", comfy)
    sys.modules.setdefault("comfy.patcher_extension", patcher_extension)


class ApplySpectrumTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_comfy_stubs()

    def test_registers_wrapper_on_clone_only(self) -> None:
        from spectrum_qwen21.config import SpectrumConfig
        from spectrum_qwen21.constants import WRAPPER_KEY
        from spectrum_qwen21.patcher import apply_spectrum

        core = FakeQwenImage21Model()
        patcher = _StubModelPatcher(core)
        patcher.model_options["transformer_options"] = {"existing": True}

        patched = apply_spectrum(patcher, SpectrumConfig())

        self.assertIsNot(patched, patcher)
        wrappers = (
            patched.model_options["transformer_options"]
            .get("wrappers", {})
            .get("diffusion_model", {})
            .get(WRAPPER_KEY)
        )
        self.assertIsNotNone(wrappers)
        self.assertEqual(len(wrappers), 1)
        self.assertTrue(callable(wrappers[0]))

        # original patcher untouched
        self.assertNotIn(
            "wrappers", patcher.model_options.get("transformer_options", {})
        )
        # unrelated transformer_options survive the clone
        self.assertTrue(
            patched.model_options["transformer_options"].get("existing")
        )

    def test_rejects_non_qwen21_model(self) -> None:
        from spectrum_qwen21.config import SpectrumConfig
        from spectrum_qwen21.patcher import apply_spectrum

        class OldQwen:
            # Qwen-Image 1.x exposes txt_norm; must be rejected
            txt_norm = None
            txt_in = None
            img_in = None
            transformer_blocks = [object()]
            norm_out = None
            proj_out = None
            time_text_embed = None
            inner_dim = 8
            out_channels = 8

        patcher = _StubModelPatcher(OldQwen())
        with self.assertRaises(ValueError):
            apply_spectrum(patcher, SpectrumConfig())

    def test_config_validated_before_patch(self) -> None:
        from spectrum_qwen21.config import SpectrumConfig
        from spectrum_qwen21.patcher import apply_spectrum

        core = FakeQwenImage21Model()
        patcher = _StubModelPatcher(core)
        bad = SpectrumConfig(chebyshev_degree=9, history_points=4)
        with self.assertRaises(ValueError):
            apply_spectrum(patcher, bad)


class NodeDefinitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_comfy_stubs()

    def test_node_mappings_importable(self) -> None:
        pkg = _load_node_package()
        nodes = importlib.import_module(f"{_PKG_NAME}.nodes")

        self.assertIn("SpectrumQwenImage21", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("SpectrumQwenImage21", nodes.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(
            nodes.NODE_DISPLAY_NAME_MAPPINGS["SpectrumQwenImage21"],
            "Spectrum (Qwen-Image-2.1)",
        )

    def test_input_types_defaults(self) -> None:
        _load_node_package()
        nodes = importlib.import_module(f"{_PKG_NAME}.nodes")
        SpectrumQwenImage21 = nodes.SpectrumQwenImage21

        spec = SpectrumQwenImage21.INPUT_TYPES()
        required = spec["required"]
        self.assertIn("model", required)
        defaults = {
            "warmup_steps": 5,
            "tail_actual_steps": 2,
            "window_size": 2.0,
            "flex_window": 0.75,
            "max_consecutive_forecasts": 8,
            "history_points": 8,
            "chebyshev_degree": 4,
            "ridge_lambda": 0.1,
            "blend_weight": 0.5,
            "cache_device": "main_device",
            "force_actual_on_control": True,
            "debug": False,
        }
        for key, value in defaults.items():
            self.assertIn(key, required, f"missing input: {key}")
            self.assertEqual(required[key][1]["default"], value, f"bad default: {key}")
        # return signature
        self.assertEqual(SpectrumQwenImage21.RETURN_TYPES, ("MODEL",))
        self.assertEqual(SpectrumQwenImage21.FUNCTION, "patch")

    def test_patch_returns_model_tuple(self) -> None:
        _load_node_package()
        nodes = importlib.import_module(f"{_PKG_NAME}.nodes")
        SpectrumQwenImage21 = nodes.SpectrumQwenImage21

        core = FakeQwenImage21Model()
        patcher = _StubModelPatcher(core)
        node = SpectrumQwenImage21()
        result = node.patch(
            model=patcher,
            warmup_steps=5,
            tail_actual_steps=2,
            window_size=2.0,
            flex_window=0.75,
            max_consecutive_forecasts=8,
            history_points=8,
            chebyshev_degree=4,
            ridge_lambda=0.1,
            blend_weight=0.5,
            cache_device="main_device",
            force_actual_on_control=True,
            debug=False,
        )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], _StubModelPatcher)


if __name__ == "__main__":
    unittest.main()
