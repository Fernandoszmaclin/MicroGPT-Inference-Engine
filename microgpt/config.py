"""Architecture metadata and parameter layout for MicroGPT."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_size: int
    model_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field.name} must be a positive integer")
        if self.vocab_size < 3:
            raise ValueError("vocab_size must allow BOS, EOS and at least one character")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")

    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        width = self.model_dim
        shapes = {
            "token_embedding": (self.vocab_size, width),
            "position_embedding": (self.context_size, width),
            "final_norm_weight": (width,),
            "final_norm_bias": (width,),
            "lm_head_weight": (width, self.vocab_size),
            "lm_head_bias": (self.vocab_size,),
        }
        for index in range(self.num_layers):
            prefix = f"layers.{index}."
            for name in ("attention_norm", "mlp_norm"):
                shapes[prefix + name + "_weight"] = (width,)
                shapes[prefix + name + "_bias"] = (width,)
            for name, input_size, output_size in (
                ("qkv", width, 3 * width),
                ("attention_out", width, width),
                ("mlp_in", width, self.mlp_dim),
                ("mlp_out", self.mlp_dim, width),
            ):
                shapes[prefix + name + "_weight"] = (input_size, output_size)
                shapes[prefix + name + "_bias"] = (output_size,)
        return shapes
