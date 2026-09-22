from __future__ import annotations

import argparse

from .constants import DIRECTIONS
from .models import generate_translations, load_saved_model


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Translate with a Turaco-NLLB checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--direction", choices=sorted(DIRECTIONS), required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-source-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    tokenizer, model, metadata = load_saved_model(args.model)
    if args.direction not in metadata.get("directions", []):
        raise ValueError(
            f"Checkpoint supports {metadata.get('directions', [])}, not {args.direction}"
        )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    result = generate_translations(
        model,
        tokenizer,
        [args.text],
        metadata["backend"],
        args.direction,
        batch_size=1,
        num_beams=args.num_beams,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
    )[0]
    print(result)


if __name__ == "__main__":
    main()
