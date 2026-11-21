"""Command-line checkpoint creation and text generation."""

import argparse
from pathlib import Path
from zipfile import BadZipFile

from .checkpoint import load_checkpoint
from .demo import make_demo_checkpoint
from .model import GPTModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="microgpt", description="Run a small GPT checkpoint.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("make-demo", help="create the reproducible example checkpoint")
    demo.add_argument("checkpoint", type=Path)

    generate = commands.add_parser("generate", help="generate text from a checkpoint")
    generate.add_argument("checkpoint", type=Path)
    generate.add_argument("--prompt", default="")
    generate.add_argument("--max-new-tokens", type=int, default=32)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--backend", choices=("auto", "numpy", "c"), default="auto")

    args = parser.parse_args(argv)
    try:
        if args.command == "make-demo":
            make_demo_checkpoint(args.checkpoint)
            print(f"Created {args.checkpoint}")
        else:
            config, tokenizer, parameters = load_checkpoint(args.checkpoint)
            model = GPTModel(config, parameters, backend=args.backend)
            prompt_ids = [tokenizer.bos_token_id, *tokenizer.encode(args.prompt)]
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
                eos_token_id=tokenizer.eos_token_id,
            )
            print(tokenizer.decode(prompt_ids + output_ids))
    except (OSError, ValueError, BadZipFile) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0
