"""Compare mysglang benchmark token IDs with Transformers greedy decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        help="let Transformers/Accelerate place a checkpoint that does not fit one GPU",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _fixed_prompt_ids(tokenizer: object, target: int) -> list[int]:
    seed = tokenizer.encode(
        "性能测试：请简洁介绍大语言模型推理。",
        add_special_tokens=False,
    )
    return (seed * ((target + len(seed) - 1) // len(seed)))[:target]


def main() -> None:
    args = _parser().parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt_ids = _fixed_prompt_ids(tokenizer, args.prompt_tokens)
    load_kwargs = {
        "local_files_only": True,
        "dtype": torch.bfloat16,
    }
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.device_map is None:
        model = model.to("cuda")
    model.eval()

    input_device = model.get_input_embeddings().weight.device
    current = torch.tensor([prompt_ids], device=input_device)
    past_key_values = None
    reference: list[int] = []
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            outputs = model(
                input_ids=current,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1].argmax(dim=-1)
            reference.append(int(next_token.item()))
            current = next_token.to(input_device)[:, None]
            past_key_values = outputs.past_key_values

    rows = [json.loads(line) for line in args.results.read_text().splitlines()]
    selected = {row["name"]: row for row in rows if row["name"] in args.names}
    comparisons = {}
    for name in args.names:
        actual = selected[name]["requests"][0]["token_ids"]
        mismatch = next(
            (index for index, pair in enumerate(zip(reference, actual)) if pair[0] != pair[1]),
            None,
        )
        comparisons[name] = {
            "exact_match": actual == reference,
            "first_mismatch_index": mismatch,
            "mysglang_token_ids": actual,
        }

    result = {
        "model": str(args.model.resolve()),
        "dtype": "bfloat16",
        "device_map": args.device_map,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "transformers_token_ids": reference,
        "comparisons": comparisons,
        "all_exact_match": all(item["exact_match"] for item in comparisons.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
