"""A single-sequence, pre-norm GPT decoder and autoregressive generation."""

from collections.abc import Mapping, Sequence
from math import isfinite

import numpy as np

from .backends import resolve_attention
from .checkpoint import validate_parameters
from .config import GPTConfig


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    variance = (centered * centered).mean(axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5)) * weight + bias


def _gelu(x: np.ndarray) -> np.ndarray:
    return np.float32(0.5) * x * (
        np.float32(1) + np.tanh(np.float32(0.7978845608) * (x + np.float32(0.044715) * x**3))
    )


class GPTModel:
    def __init__(
        self,
        config: GPTConfig,
        parameters: Mapping[str, np.ndarray],
        backend: str = "auto",
    ):
        validate_parameters(config, parameters)
        self.config = config
        self._parameters = {
            name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in parameters.items()
        }
        self.backend, self._attention = resolve_attention(backend)

    def _token_array(self, token_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or not 1 <= ids.size <= self.config.context_size:
            raise ValueError(f"expected between 1 and {self.config.context_size} token IDs")
        if ids.dtype.kind not in "iu" or (ids < 0).any() or (ids >= self.config.vocab_size).any():
            raise ValueError("token IDs must be integers within the vocabulary")
        return ids.astype(np.intp, copy=False)

    def _hidden_states(self, ids: np.ndarray) -> np.ndarray:
        params = self._parameters
        config = self.config
        x = params["token_embedding"][ids] + params["position_embedding"][:ids.size]
        head_dim = config.model_dim // config.num_heads

        for index in range(config.num_layers):
            prefix = f"layers.{index}."
            normalized = _layer_norm(
                x, params[prefix + "attention_norm_weight"], params[prefix + "attention_norm_bias"]
            )
            qkv = normalized @ params[prefix + "qkv_weight"] + params[prefix + "qkv_bias"]
            q, k, v = (
                np.ascontiguousarray(part.reshape(ids.size, config.num_heads, head_dim).transpose(1, 0, 2))
                for part in np.split(qkv, 3, axis=-1)
            )
            attended = self._attention(q, k, v).transpose(1, 0, 2).reshape(ids.size, config.model_dim)
            x = x + attended @ params[prefix + "attention_out_weight"] + params[prefix + "attention_out_bias"]

            normalized = _layer_norm(
                x, params[prefix + "mlp_norm_weight"], params[prefix + "mlp_norm_bias"]
            )
            inner = _gelu(normalized @ params[prefix + "mlp_in_weight"] + params[prefix + "mlp_in_bias"])
            x = x + inner @ params[prefix + "mlp_out_weight"] + params[prefix + "mlp_out_bias"]

        return _layer_norm(x, params["final_norm_weight"], params["final_norm_bias"])

    def logits(self, token_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return (tokens, vocabulary) logits, with a prediction for each input position."""
        hidden = self._hidden_states(self._token_array(token_ids))
        return hidden @ self._parameters["lm_head_weight"] + self._parameters["lm_head_bias"]

    def generate(
        self,
        token_ids: Sequence[int] | np.ndarray,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        seed: int | None = None,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Generate new IDs, stopping before EOS or at the context limit."""
        context = self._token_array(token_ids).tolist()
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if not isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if eos_token_id is not None and (
            type(eos_token_id) is not int or not 0 <= eos_token_id < self.config.vocab_size
        ):
            raise ValueError("eos_token_id must be an integer within the vocabulary")
        rng = np.random.default_rng(seed) if temperature else None
        generated = []
        limit = min(max_new_tokens, self.config.context_size - len(context))
        for _ in range(limit):
            scores = self.logits(context)[-1]
            if not np.isfinite(scores).all():
                raise ValueError("model produced non-finite logits")
            if rng is None:
                next_id = int(scores.argmax())
            else:
                shifted = scores.astype(np.float64) - float(scores.max())
                with np.errstate(over="ignore"):
                    weights = np.exp(shifted / temperature)
                next_id = int(rng.choice(scores.size, p=weights / weights.sum()))
            if next_id == eos_token_id:
                break
            generated.append(next_id)
            context.append(next_id)
        return generated
