from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and optionally upload a Turaco checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-card", default="README.md")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Upload after validation")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    model_card = Path(args.model_card)
    required = {
        "config.json",
        "generation_config.json",
        "turaco_config.json",
        "sentencepiece.bpe.model",
        "tokenizer_config.json",
    }
    missing = [name for name in sorted(required) if not (checkpoint / name).exists()]
    weight_files = list(checkpoint.glob("*.safetensors"))
    if missing or not weight_files:
        raise FileNotFoundError(
            json.dumps({"missing": missing, "safetensors_found": len(weight_files)}, indent=2)
        )
    if not model_card.exists():
        raise FileNotFoundError(model_card)
    card_text = model_card.read_text(encoding="utf-8")
    unresolved = [token for token in ("Pending", "REPLACE", "[HASH]", "[N]") if token in card_text]
    if unresolved:
        raise ValueError(f"Model card still contains release placeholders: {unresolved}")

    plan = {
        "checkpoint": str(checkpoint.resolve()),
        "repo_id": args.repo_id,
        "private": args.private,
        "weights": [path.name for path in weight_files],
        "execute": args.execute,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute after the release checklist is complete.")
        return

    from huggingface_hub import HfApi, create_repo

    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api = HfApi()
    api.upload_folder(
        folder_path=str(checkpoint),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload frozen Turaco-NLLB checkpoint",
    )
    api.upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add reviewed model card",
    )


if __name__ == "__main__":
    main()
