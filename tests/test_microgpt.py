import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from microgpt import (
    CharacterTokenizer,
    GPTConfig,
    GPTModel,
    load_checkpoint,
    save_checkpoint,
)


def sample_model():
    tokenizer = CharacterTokenizer("abç")
    config = GPTConfig(tokenizer.vocab_size, 5, 4, 2, 1, 6)
    rng = np.random.default_rng(28)

    def weights(*shape):
        return rng.normal(0.0, 0.25, shape).astype(np.float32)

    parameters = {
        "token_embedding": weights(5, 4),
        "position_embedding": weights(5, 4),
        "final_norm_weight": weights(4) + 1.0,
        "final_norm_bias": weights(4),
        "lm_head_weight": weights(4, 5),
        "lm_head_bias": weights(5),
    }
    for name in ("attention_norm", "mlp_norm"):
        parameters[f"layers.0.{name}_weight"] = weights(4) + 1.0
        parameters[f"layers.0.{name}_bias"] = weights(4)
    for name, dimensions in (
        ("qkv", (4, 12)),
        ("attention_out", (4, 4)),
        ("mlp_in", (4, 6)),
        ("mlp_out", (6, 4)),
    ):
        parameters[f"layers.0.{name}_weight"] = weights(*dimensions)
        parameters[f"layers.0.{name}_bias"] = weights(dimensions[-1])
    return config, tokenizer, parameters


def decoder_oracle(ids, parameters):
    """Reference for the test model, using float64 and explicit attention loops."""
    params = {name: array.astype(np.float64) for name, array in parameters.items()}

    def linear(x, name):
        return x @ params[f"{name}_weight"] + params[f"{name}_bias"]

    def normalize(x, name):
        centered = x - x.mean(axis=-1, keepdims=True)
        variance = (centered ** 2).mean(axis=-1, keepdims=True)
        return centered / np.sqrt(variance + 1e-5) * params[f"{name}_weight"] + params[
            f"{name}_bias"
        ]

    x = np.stack([
        params["token_embedding"][token] + params["position_embedding"][pos]
        for pos, token in enumerate(ids)
    ])
    q, k, v = np.split(linear(normalize(x, "layers.0.attention_norm"), "layers.0.qkv"), 3, axis=1)
    attended = np.zeros_like(x)
    for token in range(len(ids)):
        for head in (slice(0, 2), slice(2, 4)):
            scores = np.array([
                np.dot(q[token, head], k[previous, head]) / np.sqrt(2)
                for previous in range(token + 1)
            ])
            weights = np.exp(scores - scores.max())
            attended[token, head] = weights @ v[:token + 1, head] / weights.sum()
    x += linear(attended, "layers.0.attention_out")
    hidden = linear(normalize(x, "layers.0.mlp_norm"), "layers.0.mlp_in")
    gelu = 0.5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
    x += linear(gelu, "layers.0.mlp_out")
    return linear(normalize(x, "final_norm"), "lm_head")


class TestMicroGPT(unittest.TestCase):
    def setUp(self):
        self.config, self.tokenizer, self.parameters = sample_model()
        self.model = GPTModel(self.config, self.parameters, backend="numpy")

    def test_decoder_matches_independent_float64_oracle(self):
        ids = self.tokenizer.encode("abç")
        np.testing.assert_allclose(
            self.model.logits(ids),
            decoder_oracle(ids, self.parameters),
            rtol=3e-5,
            atol=3e-6,
        )

    def test_prefix_is_invariant_to_future_tokens(self):
        prefix = self.tokenizer.encode("ab")
        left = self.model.logits(prefix + self.tokenizer.encode("aç"))
        right = self.model.logits(prefix + self.tokenizer.encode("çb"))
        np.testing.assert_allclose(left[:2], right[:2], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(left[:2], self.model.logits(prefix), rtol=1e-6, atol=1e-6)
        self.assertGreater(float(np.max(np.abs(left[2:] - right[2:]))), 1e-3)

    def test_tokenizer_round_trip_and_unknown_character(self):
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode("çaab")), "çaab")
        with self.assertRaises(ValueError):
            self.tokenizer.encode("z")

    def test_checkpoint_round_trip_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_checkpoint(path, self.config, self.tokenizer, self.parameters)
            config, tokenizer, parameters = load_checkpoint(path)
        self.assertEqual(config, self.config)
        self.assertEqual(tokenizer.encode("çaab"), self.tokenizer.encode("çaab"))
        self.assertEqual(parameters.keys(), self.parameters.keys())
        for name in parameters:
            np.testing.assert_array_equal(parameters[name], self.parameters[name])
        ids = tokenizer.encode("çaab")
        reloaded = GPTModel(config, parameters, backend="numpy")
        np.testing.assert_array_equal(reloaded.logits(ids), self.model.logits(ids))

    def test_checkpoint_rejects_malformed_parameter_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_checkpoint(path, self.config, self.tokenizer, self.parameters)
            with np.load(path, allow_pickle=False) as archive:
                content = dict(archive)
            content["token_embedding"] = content["token_embedding"][:-1]
            np.savez(path, **content)
            with self.assertRaisesRegex(ValueError, "token_embedding"):
                load_checkpoint(path)

    def test_generation_greedy_seed_and_context_limit(self):
        prompt = self.tokenizer.encode("ab")
        expected = []
        for _ in range(3):
            expected.append(int(np.argmax(self.model.logits(prompt + expected)[-1])))
        self.assertEqual(self.model.generate(prompt, max_new_tokens=10), expected)
        options = {"max_new_tokens": 3, "temperature": 1.1, "seed": 42}
        first = self.model.generate(prompt, **options)
        self.assertEqual(self.model.generate(prompt, **options), first)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(0 <= token < self.config.vocab_size for token in first))

    def test_generation_stops_before_returning_eos(self):
        prompt = self.tokenizer.encode("ab")
        predicted = int(np.argmax(self.model.logits(prompt)[-1]))
        self.assertEqual(
            self.model.generate(prompt, max_new_tokens=3, eos_token_id=predicted),
            [],
        )

    def test_cli_builds_checkpoint_and_generates_demo(self):
        def run(*args):
            return subprocess.run(
                [sys.executable, "-B", "-m", "microgpt", *args],
                capture_output=True,
                check=True,
                text=True,
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "demo.npz")
            run("make-demo", path)
            for prompt, expected in (
                ("MicroGPT", "MicroGPT: ola, mundo!"),
                ("MiniApps", "MiniApps: viva o codigo!"),
            ):
                with self.subTest(prompt=prompt):
                    result = run(
                        "generate", path, "--prompt", prompt, "--max-new-tokens", "24",
                        "--backend", "numpy",
                    )
                    self.assertEqual(result.stdout.strip(), expected)


if __name__ == "__main__":
    unittest.main()
