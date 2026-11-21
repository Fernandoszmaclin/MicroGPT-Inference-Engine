"""Versioned NumPy checkpoints with JSON metadata and float32 parameters."""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import GPTConfig
from .tokenizer import CharacterTokenizer


_FORMAT_VERSION = 1
_METADATA = "metadata"
_MAX_METADATA_BYTES = 1_048_576


def validate_parameters(config: GPTConfig, parameters: Mapping[str, np.ndarray]) -> None:
    shapes = config.parameter_shapes()
    if parameters.keys() != shapes.keys():
        missing = sorted(shapes.keys() - parameters.keys())
        extra = sorted(parameters.keys() - shapes.keys())
        raise ValueError(f"invalid parameter names (missing={missing}, extra={extra})")
    for name, shape in shapes.items():
        value = parameters[name]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype.kind != "f"
            or value.dtype.itemsize != 4
        ):
            raise ValueError(f"{name} must be a float32 array of shape {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains a non-finite value")


def save_checkpoint(
    path: str | Path,
    config: GPTConfig,
    tokenizer: CharacterTokenizer,
    parameters: Mapping[str, np.ndarray],
) -> None:
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer size does not match config.vocab_size")
    validate_parameters(config, parameters)
    metadata = json.dumps({
        "version": _FORMAT_VERSION,
        "config": asdict(config),
        "chars": tokenizer.chars,
    }).encode("utf-8")
    if len(metadata) > _MAX_METADATA_BYTES:
        raise ValueError("checkpoint metadata is too large")
    # Passing a file object preserves the exact path supplied by the caller.
    with Path(path).open("wb") as file:
        np.savez(file, metadata=np.frombuffer(metadata, dtype=np.uint8), **parameters)


def load_checkpoint(path: str | Path) -> tuple[GPTConfig, CharacterTokenizer, dict[str, np.ndarray]]:
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("expected an NPZ checkpoint")
    with archive:
        if _METADATA not in archive.files:
            raise ValueError("checkpoint is missing metadata")
        metadata = archive[_METADATA]
        if metadata.ndim != 1 or metadata.dtype != np.uint8 or metadata.size > _MAX_METADATA_BYTES:
            raise ValueError("invalid checkpoint metadata array")
        try:
            data = json.loads(metadata.tobytes().decode("utf-8"))
            if type(data["version"]) is not int or data["version"] != _FORMAT_VERSION:
                raise ValueError("unsupported checkpoint version")
            config = GPTConfig(**data["config"])
            if not isinstance(data["chars"], list):
                raise ValueError("invalid tokenizer vocabulary")
            tokenizer = CharacterTokenizer(data["chars"])
            if tokenizer.vocab_size != config.vocab_size:
                raise ValueError("tokenizer size does not match config.vocab_size")
        except (KeyError, TypeError, UnicodeError) as error:
            raise ValueError("invalid checkpoint metadata") from error

        expected_names = {*config.parameter_shapes(), _METADATA}
        if len(archive.files) != len(expected_names) or set(archive.files) != expected_names:
            raise ValueError("checkpoint contains missing, duplicate, or unexpected entries")
        parameters = {name: archive[name] for name in archive.files if name != _METADATA}
    validate_parameters(config, parameters)
    return config, tokenizer, parameters
