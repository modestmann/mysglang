"""Convert a BF16 Qwen3.5 checkpoint into the official block-FP8 layout.

Qwen publishes FP8 builds only for large models. To test the FP8 path end to end on a
small GPU, this script reproduces the same format for any BF16 checkpoint: every Linear
weight becomes e4m3 with one float32 ``weight_scale_inv`` per 128x128 tile, while the
modules the official Qwen3.6 FP8 export keeps in BF16 stay unchanged (embeddings,
lm_head, norms, MoE router/shared-expert gates, DeltaNet ``in_proj_a/b``/conv/A_log/
dt_bias, ``mtp.fc`` and the vision tower). Those are listed in ``modules_to_not_convert``.

    PYTHONPATH=src python benchmarks/make_fp8_checkpoint.py \\
        --model ~/models/Qwen3.5-0.8B --output ~/models/Qwen3.5-0.8B-FP8
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from mysglang.modeling.fp8 import quantize_block_fp8

_KEEP_SUFFIXES = (
    "mlp.gate",
    "shared_expert_gate",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "mtp.fc",
)


def _quantizable(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight") or tensor.ndim != 2:
        return False
    module = name.removesuffix(".weight")
    if module.startswith(("model.visual.", "visual.")) or module.endswith(_KEEP_SUFFIXES):
        return False
    if "norm" in module.rsplit(".", 1)[-1] or module.endswith(("embed_tokens", "lm_head")):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = args.model.expanduser()
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    block = tuple(args.block_size)
    files = sorted(source.glob("*.safetensors"))
    weight_map: dict[str, str] = {}
    excluded: list[str] = []
    quantized_count = 0
    for file in files:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                tensor = checkpoint.get_tensor(name)
                if _quantizable(name, tensor):
                    weight, scale = quantize_block_fp8(tensor.to(args.device), block)
                    tensors[name] = weight.cpu()
                    tensors[name + "_scale_inv"] = scale.cpu()
                    quantized_count += 1
                else:
                    tensors[name] = tensor
                    if name.endswith(".weight") and tensor.ndim == 2:
                        excluded.append(name.removesuffix(".weight"))
        save_file(tensors, str(output / file.name), metadata={"format": "pt"})
        weight_map.update({name: file.name for name in tensors})

    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": list(block),
        "modules_to_not_convert": sorted(excluded),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2)
    )
    for extra in source.iterdir():
        if extra.suffix in {".json", ".jinja", ".txt"} and not (output / extra.name).exists():
            shutil.copy(extra, output / extra.name)
    print(f"quantized {quantized_count} Linear weights; kept {len(excluded)} in BF16 -> {output}")


if __name__ == "__main__":
    main()
