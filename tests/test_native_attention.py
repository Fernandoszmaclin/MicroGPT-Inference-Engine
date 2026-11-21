import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from microgpt import GPTModel
from microgpt.backends import causal_attention_numpy, resolve_attention
from microgpt.native import causal_attention, is_available
from test_microgpt import sample_model


class TestBackendSelection(unittest.TestCase):
    def test_unavailable_native_backend_falls_back_only_in_auto(self):
        with patch("microgpt.native.is_available", return_value=False):
            self.assertEqual(resolve_attention("auto"), ("numpy", causal_attention_numpy))
            self.assertEqual(resolve_attention("numpy"), ("numpy", causal_attention_numpy))
            with self.assertRaisesRegex(ValueError, "unavailable"):
                resolve_attention("c")

    def test_available_native_backend_is_selected(self):
        with patch("microgpt.native.is_available", return_value=True):
            self.assertEqual(resolve_attention("auto"), ("c", causal_attention))
            self.assertEqual(resolve_attention("c"), ("c", causal_attention))
            self.assertEqual(resolve_attention("numpy"), ("numpy", causal_attention_numpy))


@unittest.skipUnless(is_available(), "optional C attention library is not built")
class TestNativeAttention(unittest.TestCase):
    def test_native_attention_matches_numpy(self):
        rng = np.random.default_rng(617)
        for shape, scale in (((1, 1, 1), 1), ((2, 4, 3), 1), ((4, 9, 8), 20)):
            with self.subTest(shape=shape, scale=scale):
                q, k = (rng.normal(0, scale, shape).astype(np.float32) for _ in range(2))
                v = rng.standard_normal(shape, dtype=np.float32)
                result = causal_attention(q, k, v)
                self.assertEqual(result.dtype, np.float32)
                np.testing.assert_array_equal(result[:, 0], v[:, 0])
                np.testing.assert_allclose(
                    result, causal_attention_numpy(q, k, v), rtol=3e-5, atol=3e-6
                )

    def test_native_logits_match_numpy(self):
        config, tokenizer, parameters = sample_model()
        ids = [tokenizer.bos_token_id, *tokenizer.encode("abça")]
        reference = GPTModel(config, parameters, backend="numpy").logits(ids)
        self.assertEqual(resolve_attention("auto")[0], "c")
        for backend in ("c", "auto"):
            with self.subTest(backend=backend):
                model = GPTModel(config, parameters, backend=backend)
                self.assertEqual(model.backend, "c")
                logits = model.logits(ids)
                np.testing.assert_allclose(logits, reference, rtol=3e-5, atol=3e-6)

    def test_native_cli_matches_numpy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "demo.npz")
            command = [sys.executable, "-B", "-m", "microgpt"]
            subprocess.run([*command, "make-demo", path], capture_output=True, check=True)
            generate = [*command, "generate", path, "--prompt", "MiniApps"]
            expected = subprocess.check_output([*generate, "--backend", "numpy"])
            self.assertEqual(expected.strip(), b"MiniApps: viva o codigo!")
            for backend in ("c", "auto"):
                with self.subTest(backend=backend):
                    self.assertEqual(
                        subprocess.check_output([*generate, "--backend", backend]), expected
                    )
