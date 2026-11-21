"""Build the tiny, deterministic checkpoint used by the CLI demonstration."""

from pathlib import Path

import numpy as np

from .checkpoint import save_checkpoint
from .config import GPTConfig
from .model import GPTModel
from .tokenizer import CharacterTokenizer


DEMO_CORPUS = ("MicroGPT: ola, mundo!", "MiniApps: viva o codigo!")


def make_demo_checkpoint(path: str | Path) -> None:
    tokenizer = CharacterTokenizer(sorted(set("".join(DEMO_CORPUS))))
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=max(map(len, DEMO_CORPUS)) + 2,
        model_dim=96,
        num_heads=4,
        num_layers=1,
        mlp_dim=128,
    )
    rng = np.random.default_rng(2027)
    parameters = {}
    for name, shape in config.parameter_shapes().items():
        if name.endswith("norm_weight"):
            values = np.ones(shape, dtype=np.float32)
        elif name in ("token_embedding", "position_embedding"):
            values = rng.standard_normal(shape, dtype=np.float32)
        elif name.endswith("_bias") or name == "lm_head_weight":
            values = np.zeros(shape, dtype=np.float32)
        else:
            values = rng.standard_normal(shape, dtype=np.float32) * np.float32(0.08)
        parameters[name] = values

    model = GPTModel(config, parameters, backend="numpy")
    hidden = []
    targets = []
    for sentence in DEMO_CORPUS:
        ids = [tokenizer.bos_token_id, *tokenizer.encode(sentence)]
        hidden.append(model._hidden_states(np.asarray(ids, dtype=np.intp)))
        targets.extend([*ids[1:], tokenizer.eos_token_id])

    # Fit only the vocabulary projection with ridge regression on the two strings.
    states = np.concatenate(hidden).astype(np.float64)
    labels = np.eye(config.vocab_size)[targets]
    gram = states @ states.T + 0.01 * np.eye(states.shape[0])
    parameters["lm_head_weight"] = (12.0 * states.T @ np.linalg.solve(gram, labels)).astype(np.float32)
    save_checkpoint(path, config, tokenizer, parameters)
